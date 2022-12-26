
import sys
import os
print(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gym
import pickle as pkl
from typing import Any, Dict, Callable, Tuple
from offline_baselines_jax.common.buffers import ReplayBuffer, TaskDictReplayBuffer, DictReplayBuffer, ModelBasedDictReplayBuffer
from offline_baselines_jax.common.type_aliases import Params
from offline_baselines_jax import SAC
import tqdm

from metaworld import MT50
import numpy as np
import jax.numpy as jnp
import jax
import torch
import random
import copy
import argparse
import functools

LOG_STD_MAX = 2
LOG_STD_MIN = -10


@jax.jit
def normal_sampling(key:Any, task_latents_mu:jnp.ndarray, task_latents_log_std:jnp.ndarray):
    return task_latents_mu + jax.random.normal(key, shape=(task_latents_log_std.shape[-1], )) * jnp.exp(0.5 * task_latents_log_std)

@functools.partial(jax.jit, static_argnames=('task_embedding_fn'))
def inference_task_embeddings(rng: int, task_embedding_fn: Callable[..., Any], encoder_params: Params,
                    traj: jnp.ndarray) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    task_embeddings = task_embedding_fn({'params': encoder_params}, traj)
    task_embeddings_log_std = jnp.clip(task_embeddings[1], LOG_STD_MIN, LOG_STD_MAX)
    task_latents = task_embeddings[0] + jax.random.normal(key, shape=task_embeddings[1].shape) * jnp.exp(0.5 * task_embeddings[1])
    return rng, task_embeddings[0], task_embeddings_log_std, task_latents

@functools.partial(jax.jit, static_argnames=('task_embedding_fn'))
def inference_task_embeddings_AE(task_embedding_fn: Callable[..., Any], encoder_params: Params,
                    traj: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    task_embeddings = task_embedding_fn({'params': encoder_params}, traj)
    return task_embeddings

@functools.partial(jax.jit, static_argnames=('task_embedding_fn'))
def inference_task_embeddings_BE(rng: int, task_embedding_fn: Callable[..., Any], encoder_params: Params,
                    ps: jnp.ndarray, pa: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    task_embeddings = task_embedding_fn({'params': encoder_params}, ps, pa)
    return rng, task_embeddings



@functools.partial(jax.jit, static_argnames=('reward_fn'))
def reward_transition_function(reward_fn: Callable[..., Any], encoder_params: Params, state: jnp.ndarray, action: jnp.ndarray, task_latents:jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    out = reward_fn({'params': encoder_params}, state, action, task_latents)
    return out

@functools.partial(jax.jit, static_argnames=('reward_fn'))
def transition_function(reward_fn: Callable[..., Any], encoder_params: Params, state: jnp.ndarray, action: jnp.ndarray, task_latents:jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    out = reward_fn({'params': encoder_params}, state, action, task_latents)
    return out

@functools.partial(jax.jit, static_argnames=('reward_fn'))
def policy_function(reward_fn: Callable[..., Any], encoder_params: Params, state: jnp.ndarray, policy_latents: jnp.ndarray, seq:jnp.ndarray) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    out = reward_fn({'params': encoder_params}, state, policy_latents, seq)
    return out


class OfflineDatasets(object):
    def __init__(self):
        super().__init__()
        self.trajectories = []
        self.num_episodes = 0
        self.num_success = 0
        self.num_timesteps = 0
        self.temp_trajectory = []

    def get_episodic_rewards(self):
        episodic_rewards = []
        for traj in self.trajectories:
            sum_reward = 0
            for t in traj:
                sum_reward += t['reward']
            episodic_rewards.append(sum_reward)
        return episodic_rewards

    def add(self, state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.float32, done: bool,
            infos: Dict[str, Any], success=False) -> None:
        data = {'state': state, 'next_state': next_state, 'action': action, 'reward': reward, 'done': done, 'info': infos}
        self.temp_trajectory.append(data)
        if done:
            self.num_episodes += 1
            self.num_timesteps += len(self.temp_trajectory)
            self.trajectories.append(self.temp_trajectory)
            self.temp_trajectory = []
            if success:
                self.num_success += 1
        return

    def add_traj(self, traj, success=False):
        self.trajectories.append(traj)
        self.num_episodes += 1
        self.num_timesteps += len(traj)
        if success:
            self.num_success += 1

    @property
    def state_size(self):
        if len(self.trajectories) > 0:
            return self.trajectories[0][0]['state'].shape[0]
        elif len(self.temp_trajectory) > 0:
            return self.temp_trajectory[0]['state'].shape[0]
        else:
            return 0

    @property
    def action_size(self):
        if len(self.trajectories) > 0:
            return self.trajectories[0][0]['action'].shape[0]
        elif len(self.temp_trajectory) > 0:
            return self.temp_trajectory[0]['action'].shape[0]
        else:
            return 0


    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump({'trajectories': self.trajectories, 'num_episodes': self.num_episodes, 'num_timesteps': self.num_timesteps, 'num_success': self.num_success}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)

        self.trajectories = data['trajectories']
        self.num_episodes = data['num_episodes']
        self.num_timesteps = data['num_timesteps']
        self.num_success = data['num_success']

    def get_replay_buffer(self, buffer_size):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return
        obs_space = gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['state'].shape)
        action_space = gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['action'].shape)

        replay_buffer = ReplayBuffer(buffer_size, observation_space=obs_space, action_space=action_space)
        for trajectory in tqdm.tqdm(self.trajectories):
            for t in trajectory:
                replay_buffer.add(np.array([t['state']]), np.array([t['next_state']]), np.array([t['action']]),
                                  np.array([t['reward']]), np.array([t['done']]), [t['info']])

        return replay_buffer

    def get_task_ID_replay_buffer(self, buffer_size, task, num_tasks, replay_buffer=None, data_augmentation=False):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return

        task_labels = torch.zeros(num_tasks)
        task_labels[task] = 1

        obs_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['state'].shape),
                                     'task': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_tasks, ))})

        action_space = gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['action'].shape)
        if replay_buffer is None:
            if data_augmentation:
                replay_buffer = ModelBasedDictReplayBuffer(buffer_size, use_embeddings=False, observation_space=obs_space, action_space=action_space)
            else:
                replay_buffer = TaskDictReplayBuffer(buffer_size, observation_space=obs_space, action_space=action_space, num_tasks=num_tasks)

        for trajectory in tqdm.tqdm(self.trajectories):
            for t in trajectory:
                observation = {'obs': t['state'], 'task': task_labels}
                next_observation = {'obs': t['next_state'], 'task': task_labels}
                info = copy.deepcopy(t['info'])
                info['task'] = task
                replay_buffer.add(observation, next_observation, np.array([t['action']]),
                                  np.array([t['reward']]), np.array([t['done']]), [info])

        return replay_buffer

    def get_task_label_replay_buffer(self, buffer_size, task_encoder, latent_space, task, num_tasks, replay_buffer=None):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return

        task_labels = torch.zeros(num_tasks)
        task_labels[task] = 1

        task_embeddings = task_encoder(task_labels.clone())[0].detach().numpy()

        obs_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['state'].shape),
                                     'task': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_space, ))})

        action_space = gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['action'].shape)
        if replay_buffer is None:
            replay_buffer = TaskDictReplayBuffer(buffer_size, observation_space=obs_space, action_space=action_space, num_tasks=num_tasks)

        for trajectory in tqdm.tqdm(self.trajectories):
            for t in trajectory:
                observation = {'obs': t['state'], 'task': task_embeddings}
                next_observation = {'obs': t['next_state'], 'task': task_embeddings}
                info = copy.deepcopy(t['info'])
                info['task'] = task
                replay_buffer.add(observation, next_observation, np.array([t['action']]),
                                  np.array([t['reward']]), np.array([t['done']]), [info])

        return replay_buffer

    def get_transitions_replay_buffer(self, buffer_size, task_encoder, latent_space, task, num_tasks, replay_buffer=None):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return

        task_labels = torch.zeros(num_tasks)
        task_labels[task] = 1

        task_embeddings = task_encoder(task_labels.clone())[0].detach().numpy()

        obs_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['state'].shape),
                                     'task': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_space, ))})

        action_space = gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['action'].shape)
        if replay_buffer is None:
            replay_buffer = TaskDictReplayBuffer(buffer_size, observation_space=obs_space, action_space=action_space, num_tasks=num_tasks)

        for trajectory in tqdm.tqdm(self.trajectories):
            for t in trajectory:
                observation = {'obs': t['state'], 'task': task_embeddings}
                next_observation = {'obs': t['next_state'], 'task': task_embeddings}
                info = copy.deepcopy(t['info'])
                info['task'] = task
                replay_buffer.add(observation, next_observation, np.array([t['action']]),
                                  np.array([t['reward']]), np.array([t['done']]), [info])

        return replay_buffer

    def sample_traj_embeddings(self, task_encoder, mode='te', n_steps=4):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return

        if mode == 'te':
            trajectory = np.random.randint(0, len(self.trajectories))
            trajectory = self.trajectories[trajectory]
            task_embeddings_list = []

            for idx, t in enumerate(trajectory):
                traj = []
                for i in range(n_steps):
                    transitions = np.zeros((t['action'].shape[0] + t['state'].shape[0] + 1))
                    history_idx = idx - n_steps + i
                    if history_idx >= 0:
                        transitions[:t['action'].shape[0]] = trajectory[history_idx]['action']
                        transitions[t['action'].shape[0]: t['action'].shape[0] + t['state'].shape[0]] = \
                        trajectory[history_idx]['next_state']
                        transitions[t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['reward']
                    traj.append(transitions)

                traj = np.concatenate(traj)

                rng = jax.random.PRNGKey(0)
                task_latents = inference_task_embeddings_AE(task_encoder.apply_fn, task_encoder.params, traj)

                task_embeddings_list.append(task_latents)

        elif mode == 'be':
            trajectory = np.random.randint(0, len(self.trajectories))
            trajectory = self.trajectories[trajectory]
            task_embeddings_list = []

            for idx, t in enumerate(trajectory):
                ps = []
                pa = []
                for i in range(n_steps - 1):
                    history_idx = idx - n_steps + i + 1
                    if history_idx >= 0:
                        ps.append(trajectory[history_idx]['state'])
                        pa.append(trajectory[history_idx]['action'])
                    else:
                        ps.append(np.zeros_like(t['state']))
                        pa.append(np.zeros_like(t['action']))

                ps = np.concatenate(ps)
                pa = np.concatenate(pa)
                rng = jax.random.PRNGKey(0)
                key, task_latents = inference_task_embeddings_BE(rng, task_encoder.apply_fn, task_encoder.params, ps, pa)

                task_embeddings_list.append(task_latents)


        return trajectory, np.array(task_embeddings_list)

    def sample_transition_embeddings(self, task_encoder):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return

        trajectory = np.random.choice(self.trajectories)
        task_embeddings_list = []

        for idx, t in enumerate(trajectory):
            transition = np.array([np.concatenate([t['state'], t['action'], np.array([t['reward']]), t['next_state']])])
            with torch.no_grad():
                task_embeddings = task_encoder(torch.from_numpy(transition).float())[1].cpu().detach().numpy()
            task_embeddings_list.append(task_embeddings[0])

        return trajectory, np.array(task_embeddings_list)

    def sample_states(self, n_episode=100):
        sampled_states = []
        a = np.random.choice(np.arange(len(self.trajectories)), n_episode)
        for i in a:
            trajectory = self.trajectories[i]
            for t in trajectory:
                sampled_states.append(np.concatenate([t['state'], t['action']]))
        return sampled_states

        # for trajectory in tqdm.tqdm(self.trajectories):
        #     for idx, t in enumerate(trajectory):


    def get_trajectories_replay_buffer(self, key, buffer_size, task_encoder, latent_space, task, num_tasks, obs_space, replay_buffer=None, n_steps=4, AE=True,
                                       data_augmentation=False, behavior_decoder=None, transition_decoder=None, reward_decoder=None):
        if self.num_timesteps == 0:
            print('buffer is empty!')
            return
        skill_data = []
        task_labels = torch.zeros(num_tasks)
        task_labels[task] = 1
        state_set = []
        for trajectory in self.trajectories:
            for t in trajectory:
                state_set.append(t['state'])

        state_set = []

        a = np.random.choice(np.arange(len(self.trajectories)), 25)

        obs_space = gym.spaces.Dict({'obs': obs_space, 'task': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_space, ))})
        action_space = gym.spaces.Box(low=-1, high=1, shape=self.trajectories[0][0]['action'].shape)
        if replay_buffer is None:
            task_embedding_lambda = not AE
            replay_buffer = DictReplayBuffer(buffer_size, observation_space=obs_space, action_space=action_space,
                                                 task_embedding_lambda=task_embedding_lambda, use_embeddings=False)

        for trajectory in tqdm.tqdm(self.trajectories):
            sum_reward = 0
            for idx_t in range(len(trajectory)):
                sum_reward += (0.99 ** (idx_t)) * trajectory[idx_t]['reward']
            for idx, t in enumerate(trajectory):
                if task_encoder is None:
                    task_latents = np.zeros(latent_space)
                    next_task_latents = task_latents
                else:
                    traj = []
                    for i in range(n_steps):
                        transitions = np.zeros((t['action'].shape[0] + t['state'].shape[0] + 1))
                        history_idx = idx - n_steps + i
                        if history_idx >= 0:
                            transitions[:t['action'].shape[0]] = trajectory[history_idx]['action']
                            transitions[t['action'].shape[0]: t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['next_state']
                            transitions[t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['reward']
                        traj.append(transitions)

                    traj = np.concatenate(traj)
                    task_latents = inference_task_embeddings_AE(task_encoder.apply_fn, task_encoder.params, traj)

                    if not t['done'] and idx + 1 < len(trajectory):
                        traj = []
                        for i in range(n_steps):
                            transitions = np.zeros((t['action'].shape[0] + t['state'].shape[0] + 1))
                            history_idx = idx + 1 - n_steps + i
                            if history_idx >= 0:
                                transitions[:t['action'].shape[0]] = trajectory[history_idx]['action']
                                transitions[t['action'].shape[0]: t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['next_state']
                                transitions[t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['reward']
                            traj.append(transitions)

                        _traj = np.concatenate(traj)
                        next_task_latents = inference_task_embeddings_AE(task_encoder.apply_fn, task_encoder.params, _traj)
                    else:
                        next_task_latents = task_latents

                next_observation = {'obs': t['next_state'], 'task': next_task_latents}
                observation = {'obs': t['state'], 'task': task_latents}
                info = copy.deepcopy(t['info'])
                reward = np.array([t['reward']])
                replay_buffer.add(observation, next_observation, np.array([t['action']]), reward, np.array([t['done']]), [info])

                if data_augmentation and not t['done'] and np.random.random() <= 1:
                    obs = t['state']
                    del traj[-1]
                    key, policy_key = jax.random.split(key)
                    policy_latents = task_latents.copy() + normal_sampling(policy_key, jnp.zeros_like(task_latents), jnp.ones_like(task_latents) * -4.6)
                    for i in range(4):
                        seq = np.zeros((8, ))
                        seq[i+4] = 1
                        action = policy_function(behavior_decoder.apply_fn, behavior_decoder.params, obs, policy_latents, seq)
                        action = np.clip(action, a_min=-1, a_max=1)

                        state_set.append(np.concatenate([obs, action]))

                        reward = reward_transition_function(reward_decoder.apply_fn, reward_decoder.params, obs, action, task_latents)
                        skill_data.append([sum_reward, action, t['action'], reward])
                        next_state = transition_function(transition_decoder.apply_fn, transition_decoder.params, obs, action, task_latents)
                        # remove noise for standardization
                        next_state = np.where(np.abs(next_state) < 1e-3, 0, next_state)
                        transitions = np.zeros((t['action'].shape[0] + t['state'].shape[0] + 1))
                        transitions[:t['action'].shape[0]] = action
                        transitions[t['action'].shape[0]: t['action'].shape[0] + t['state'].shape[0]] = next_state
                        transitions[t['action'].shape[0] + t['state'].shape[0]] = reward

                        traj.append(transitions)
                        _traj = np.concatenate(traj)
                        next_task_latents = inference_task_embeddings_AE(task_encoder.apply_fn, task_encoder.params, _traj)

                        # if i == 0:
                        next_observation = {'obs': obs.copy(), 'task': next_task_latents.copy()}
                        observation = {'obs': obs.copy(), 'task': task_latents.copy()}
                        info = copy.deepcopy(t['info'])
                        info['task'] = task
                        replay_buffer.add(observation, next_observation, np.array([action]), np.array([reward]), np.array([t['done']]), [info])

                        obs = next_state
                        task_latents = next_task_latents
                        del traj[0]

        return replay_buffer

    def export_dataset(self, mode='transitions', n_steps=None):
        if mode == 'transitions':
            states = []
            actions = []
            rewards = []
            next_states = []
            for trajectory in tqdm.tqdm(self.trajectories):
                for t in trajectory:
                    states.append(t['state'])
                    actions.append(t['action'])
                    rewards.append(np.array([t['reward']]))
                    next_states.append(t['next_state'])

            return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

        elif mode == 'trajectories':
            trajectories = []
            rewards = []
            actions = []
            states = []
            prev_states = []
            prev_actions = []
            sum_rewards = []
            next_states = []
            for trajectory in tqdm.tqdm(self.trajectories):
                sum_reward = 0
                for idx_t in range(len(trajectory)):
                    sum_reward += (0.99 ** (idx_t)) * trajectory[idx_t]['reward']
                for idx, t in enumerate(trajectory):
                    traj = []
                    prev_state = []
                    prev_action = []
                    s = []
                    a = []
                    r = []
                    n_s = []
                    # sum_reward = 0
                    for i in range(n_steps):
                        transitions = np.zeros((t['action'].shape[0] + t['state'].shape[0] + 1))
                        ps = np.zeros(t['state'].shape)
                        ns = np.zeros(t['state'].shape)
                        pa = np.zeros(t['action'].shape)
                        _r = 0
                        history_idx = idx - n_steps + i
                        if history_idx >= 0:
                            transitions[:t['action'].shape[0]] = trajectory[history_idx]['action'].copy()
                            transitions[t['action'].shape[0] : t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['next_state'].copy()
                            transitions[t['action'].shape[0] + t['state'].shape[0]] = trajectory[history_idx]['reward']
                            pa = trajectory[history_idx]['action'].copy()
                            ps = trajectory[history_idx]['state'].copy()
                            ns = trajectory[history_idx]['next_state'].copy()
                            _r = trajectory[history_idx]['reward']
                            # sum_reward += t['reward']
                        prev_state.append(ps)
                        prev_action.append(pa)
                        s.append(ps.copy())
                        a.append(pa.copy())
                        n_s.append(ns)
                        r.append(np.array([_r]))
                        traj.append(transitions)

                    for i in range(n_steps):
                        ps = np.zeros(t['state'].shape)
                        pa = np.zeros(t['action'].shape)
                        # sum_reward += t['reward']
                        behavior_idx = idx + i
                        if behavior_idx < len(self.trajectories):
                            pa = trajectory[behavior_idx]['action'].copy()
                            ps = trajectory[behavior_idx]['state'].copy()
                        prev_state.append(ps)
                        prev_action.append(pa)

                    traj = np.concatenate(traj)
                    p_s = np.array(prev_state)
                    p_a = np.array(prev_action)
                    s = np.array(s)
                    a = np.array(a)
                    r = np.array(r)
                    n_s = np.array(n_s)

                    trajectories.append(traj)
                    prev_states.append(p_s)
                    prev_actions.append(p_a)

                    states.append(s)
                    actions.append(a)
                    rewards.append(r)
                    sum_rewards.append(sum_reward)
                    next_states.append(n_s)

            return np.array(trajectories), np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(sum_rewards), np.array(prev_states), np.array(prev_actions)

    def get_info(self):
        self.update_params()
        print('Num of Episode: {}'.format(self.num_episodes))
        print('Total timesteps: {}'.format(self.num_timesteps))
        print('Timesteps/Episode: {}'.format(int(self.num_timesteps / self.num_episodes)))
        print('Success Episode: {}'.format(self.num_success))
        print('Success Rate %.3d'%(self.num_success / self.num_episodes * 100))
        print('Timesteps/Episode (Success): {}'.format(int((self.num_timesteps - 200 * (self.num_episodes - self.num_success)) / self.num_success)))
        return

    def extend(self, data):
        self.trajectories.extend(data.trajectories)
        self.num_success += data.num_success
        self.num_episodes += data.num_episodes
        self.num_timesteps += data.num_timesteps

    def success_traj(self):
        return_data = OfflineDatasets()
        for traj in self.trajectories:
            if traj[-1]['info']['success']:
                return_data.trajectories.append(copy.deepcopy(traj))
                return_data.num_success += 1
                return_data.num_episodes += 1
                return_data.num_timesteps += len(traj)
        return return_data

    def failed_traj(self):
        return_data = OfflineDatasets()
        for traj in self.trajectories:
            if not traj[-1]['info']['success']:
                return_data.trajectories.append(copy.deepcopy(traj))
                return_data.num_episodes += 1
                return_data.num_timesteps += len(traj)
        return return_data

    def _shuffle(self):
        np.random.shuffle(self.trajectories)

    def sample(self, num_episode):
        if num_episode > self.num_episodes:
            num_episode = self.num_episodes
        self._shuffle()

        return_data = OfflineDatasets()
        return_data.trajectories = copy.deepcopy(self.trajectories[:num_episode])
        return_data.update_params()

        return return_data

    def add_task_label(self, task, num_tasks):
        one_hot_task = np.zeros(num_tasks)
        one_hot_task[task] = 1
        for traj in self.trajectories:
            for i in range(len(traj)):
                traj[i]['state'] = np.concatenate([traj[i]['state'], one_hot_task])
                traj[i]['next_state'] = np.concatenate([traj[i]['next_state'], one_hot_task])
        return

    def make_goal_mdp(self):
        for traj in self.trajectories:
            for i in range(len(traj)):
                if traj[i]['info']['success']:
                    traj[i]['reward'] = 1
                else:
                    traj[i]['reward'] = 0

    def update_params(self):
        self.num_episodes = len(self.trajectories)
        self.num_success = 0
        self.num_timesteps = 0
        for traj in self.trajectories:
            self.num_timesteps += len(traj)
            if traj[-1]['info']['success']:
                self.num_success += 1

    def add_torchrl_replay_buffer(self, replay_buffer, task_idx):
        for traj in self.trajectories:
            for i in range(len(traj)):
                sample_dict = {
                    "obs": traj[i]['state'],
                    "next_obs": traj[i]['next_state'],
                    "acts": traj[i]['action'],
                    "task_idxs": [task_idx],
                    "rewards": [traj[i]['reward']],
                    "terminals": [traj[i]['done']]
                }
                replay_buffer.add_sample(sample_dict, task_idx)
        return replay_buffer


class OfflineDataCollector(gym.Wrapper):
    def __init__(self, env, datasets):
        super().__init__(env)
        self.datasets = datasets
        self.state = None

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        self.state = state
        return state

    def step(self, action):
        next_state, rewards, done, info = super().step(action)
        self.datasets.add(self.state, next_state, action, rewards, done, info)
        return next_state, rewards, done, info


def GetOfflineData(task_name: str, policy_path:str, path: str, model_save_freq:int, episodes: int, max_timesteps:int, num_model:int):
    print('Get expert data for {}'.format(task_name))

    from envs.env_dict import TimeLimitRewardMDP
    from envs.meta_world import SingleTask

    env = SingleTask(seed, task_name)
    test_env = TimeLimitRewardMDP(env)

    save_path = os.path.join(path, task_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(path, task_name)
    policy_model_path = os.path.join(policy_path, task_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(num_model):
        expert_replay_buffer = OfflineDatasets()
        success = 0
        timesteps = 0
        progress_bar = tqdm.tqdm(range(episodes))
        policy = SAC.load(os.path.join(policy_model_path, 'model_{}_steps.zip'.format(model_save_freq * (i + 1))))
        sum_reward = 0
        for epochs in progress_bar:
            obs = test_env.reset()
            done = False
            # success_check = False
            while not done:
                action = policy.predict(np.expand_dims(obs, axis=0))[0][0]

                next_obs, reward, done, info = test_env.step(action)
                expert_replay_buffer.add(obs, next_obs, action, reward, done, info, info['success'])
                timesteps += 1
                obs = next_obs

                sum_reward += reward
                if info['success'] == 1:
                    success += 1
            if max_timesteps is not None and timesteps > max_timesteps:
                break
            progress_bar.set_description("TIMESTEP %d MODEL:: Success rate %.3f%% Timesteps %d Reward: %.3f"%(model_save_freq * (i + 1),
                                                                                                 success / (epochs+1) * 100, timesteps, sum_reward/(epochs+1)))
        expert_replay_buffer.save(os.path.join(save_path, 'offline_data_{}.pkl'.format(model_save_freq * (i + 1))))

def GetRandomOfflineData(task_name: str, path: str, episodes: int, max_timesteps:int):
    print('Get expert data for {}'.format(task_name))
    # make environment

    from envs.env_dict import TimeLimitRewardMDP
    from envs.meta_world import SingleTask

    env = SingleTask(seed, task_name)
    test_env = TimeLimitRewardMDP(env)

    save_path = os.path.join(path, task_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # set meta-world environment to goal MDP

    expert_replay_buffer = OfflineDatasets()
    success = 0
    timesteps = 0


    progress_bar = tqdm.tqdm(range(episodes))
    for epochs in progress_bar:
        obs = test_env.reset()
        done = False
        # success_check = False
        while not done:
            action = test_env.action_space.sample()
            next_obs, reward, done, info = test_env.step(action)
            expert_replay_buffer.add(obs, next_obs, action, reward, done, info, info['success'])
            timesteps += 1
            obs = next_obs
            if info['success'] == 1:
                success += 1
        if max_timesteps is not None and timesteps > max_timesteps:
            break
        progress_bar.set_description("RANDOM MODEL:: Success rate %.3f%% Timesteps %d"%(success / (epochs+1) * 100, timesteps))
    expert_replay_buffer.save(os.path.join(save_path, 'offline_data_random.pkl'))


if __name__ == '__main__':

    seed = 777
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default=None)
    parser.add_argument('--model-save-freq', type=int, default=50_000)
    parser.add_argument('--num-model', type=int, default=40)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max-timesteps', type=int, default=2_000_000)
    parser.add_argument('--path', type=str, default=os.getcwd() + '/single_task/offline_data')
    parser.add_argument('--policy-path', type=str, default=os.getcwd() + '/single_task')
    parser.add_argument('--random-policy', action='store_true', default=True)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        os.mkdir(args.path)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from envs.env_dict import TimeLimitRewardMDP, MT50_TASK, MT10_TASK
    if args.task_name is None:
        print("Input task name for generating expert data")
        print(MT50_TASK)
    elif args.task_name == 'mt10':
        for task_name in MT10_TASK:
            if args.random_policy:
                GetRandomOfflineData(task_name=task_name, path=args.path, episodes=args.episodes, max_timesteps=args.max_timesteps)
            else:
                GetOfflineData(task_name=task_name, episodes=args.episodes, path=args.path, num_model=args.num_model,
                           model_save_freq=args.model_save_freq, max_timesteps=args.max_timesteps, policy_path=args.policy_path)
    else:

        if args.random_policy:
            GetRandomOfflineData(task_name=args.task_name, path=args.path, episodes=args.episodes, max_timesteps=args.max_timesteps)
        else:
            GetOfflineData(task_name=args.task_name, episodes=args.episodes, path=args.path, num_model=args.num_model,
                       model_save_freq=args.model_save_freq, max_timesteps=args.max_timesteps, policy_path=args.policy_path)