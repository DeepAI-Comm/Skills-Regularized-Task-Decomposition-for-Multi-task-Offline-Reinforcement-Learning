import gym
from typing import Any, Dict, Callable, Tuple
from offline_baselines_jax.common.buffers import ReplayBuffer, TaskDictReplayBuffer
from offline_baselines_jax.common.type_aliases import Params

import jax.numpy as jnp
import jax
import functools
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
import torch
from collections import deque

from metaworld.envs.mujoco.env_dict import MT10_V2

@functools.partial(jax.jit, static_argnames=('task_embedding_fn'))
def inference_task_embeddings(task_embedding_fn: Callable[..., Any], encoder_params: Params,
                    traj: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    task_embeddings = task_embedding_fn({'params': encoder_params}, traj)
    return task_embeddings

@functools.partial(jax.jit, static_argnames=('task_embedding_fn'))
def behavior_decoding_AE(task_embedding_fn: Callable[..., Any], encoder_params: Params, state: jnp.ndarray, latent: jnp.ndarray, seq:jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    task_embeddings = task_embedding_fn({'params': encoder_params}, state, latent, seq)
    return task_embeddings

EASY_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2']
INTERMEDIATE_TASK = ['peg-insert-side-v2', 'push-v2']
HARD_TASK = ['button-press-topdown-v2', 'door-open-v2', 'drawer-open-v2', 'pick-place-v2']

CDS_TASK = ['door-open-v2', 'door-close-v2', 'drawer-open-v2', 'drawer-close-v2']

MT50_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
             'door-open-v2', 'drawer-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2',
             'assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2',  'button-press-topdown-wall-v2',
             'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2',
             'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-unlock-v2', 'hand-insert-v2',
             'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2',
             'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2',  'pick-place-wall-v2', 'pick-out-of-hole-v2',
             'push-back-v2',   'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
             'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2',
             'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2']

MT10_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
             'door-open-v2', 'drawer-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2']

MT8_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2',
             'door-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2']

AIRSIM_TASK = ['indoor_complex', 'indoor_pyramid', 'indoor_gt', 'indoor_cloud_wind', 'indoor_complex_wind', 'indoor_gt_wind']

def meta_world_task(name, mode='GoalMDP'):
    meta_world_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]()
    if mode == 'GoalMDP':
        return TimeLimitGoalMDP(meta_world_env)
    elif mode == 'TimeLimitMDP':
        return TimeLimitMDP(meta_world_env)

def meta_world_item():
    return ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()

class DummyEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, action):
        pass

    def reset(self):
        return self.observation_space.sample()


class TrajectoriesEncoderEnv(gym.Wrapper):
    def __init__(self, env, task_encoder, latent_space, seed, n_steps=4, mean=0, std=1):
        super().__init__(env)
        self.n_steps = n_steps
        self.key = jax.random.PRNGKey(seed)
        self.latent_space = latent_space
        self.task_encoder = task_encoder

        self.state_history = list()
        self.action_history = list()
        self.reward_history = list()

        self.mean = mean
        self.std = std
        self.observation_space = gym.spaces.Dict({
            'obs': self.env.observation_space['obs'],
            'task': gym.spaces.Box(shape=(latent_space,), low=-np.inf, high=np.inf)
        })

        self.action_shape = self.action_space.shape[0]
        self.state_shape = self.observation_space['obs'].shape[0]

    def reset(self):
        state = self.env.reset()
        self.state_history = list()
        self.action_history = list()
        self.reward_history = list()
        self.state_history.append(state['obs'])
        obs = {'obs': (state['obs'] - self.mean) / (self.std + 1e-3), 'task': np.zeros(self.latent_space)}
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.action_history.append(action)
        self.state_history.append(state['obs'])
        self.reward_history.append(reward)

        traj = []
        for i in range(self.n_steps):
            transitions = np.zeros((self.action_shape + self.state_shape + 1, ))
            idx = len(self.action_history) - self.n_steps + i
            if idx >= 0:
                transitions[:self.action_shape] = self.action_history[idx]
                transitions[self.action_shape: self.action_shape + self.state_shape] = self.state_history[idx]
                transitions[self.action_shape + self.state_shape] = self.reward_history[idx]
            traj.append(transitions)

        traj = np.concatenate(traj)
        task_latents = inference_task_embeddings(self.task_encoder.apply_fn, self.task_encoder.params, traj)
        obs = {'obs': (state['obs'] - self.mean) / (self.std + 1e-3), 'task': task_latents}
        return obs, reward, done, info


class NormalizeEnv(gym.Wrapper):
    def __init__(self, env, mean=None, std=None):
        super().__init__(env)
        if mean is None:
            mean = np.zeros((10, ))
        if std is None:
            std = np.ones((10, ))

        self.mean = mean
        self.std = std

    def reset(self):
        state = self.env.reset()
        a = np.argmax(state['task'])
        obs = {'obs': (state['obs'] - self.mean[a]) / (self.std[a] + 1e-3), 'task': state['task']}
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        a = np.argmax(state['task'])
        obs = {'obs': (state['obs'] - self.mean[a]) / (self.std[a] + 1e-3), 'task': state['task']}
        return obs, reward, done, info


class TimeLimitMDP(gym.Wrapper):
    def __init__(self, env, max_episode_steps=200):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.success = False
        self.reward = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True

        return state, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.success = False
        self.reward = 0
        return self.env.reset(**kwargs)


class TimeLimitRewardMDP(TimeLimitMDP):
    def step(self, action):
        state, reward, done, info = super().step(action)
        info['reward'] = reward
        shaping_reward = reward - self.reward
        self.reward = reward

        return state, shaping_reward, done, info


class TimeLimitGoalMDP(TimeLimitMDP):
    def step(self, action):
        state, reward, done, info = super().step(action)
        info['reward'] = reward
        if info['success'] and not self.success:
            reward = 1
            self.success = True
        else:
            reward = 0

        return state, reward, done, info
