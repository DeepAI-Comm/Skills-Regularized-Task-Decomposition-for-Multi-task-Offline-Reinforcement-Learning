import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from offline_baselines_jax import TD3
from offline_baselines_jax.td3.policies import MultiInputPolicy
TD3Policy = MultiInputPolicy
from offline_baselines_jax.sac.policies import MultiInputPolicy
SACPolicy = MultiInputPolicy

from envs.env_dict import TimeLimitRewardMDP, MT10_TASK, TrajectoriesEncoderEnv, AIRSIM_TASK, DummyEnv
from envs.meta_world import MetaWorldIndexedMultiTaskTester
from offline_data.offline_data_collector import OfflineDatasets
from cfg.sampling_policy_cfg import sampling_policy, default_dynamic_num_data
from stable_baselines3.common.evaluation import evaluate_policy
from collections import deque

from typing import List
from offline_baselines_jax.common.callbacks import CheckpointCallback
from offline_baselines_jax.common.buffers import ModelBasedDictReplayBuffer
from jax_models import Model, TaskEncoderAE, RewardDecoder, TransitionDecoder, BehaviorDecoder

import jax
import numpy as np
import random
import gym
import torch
import argparse
import flax.linen as nn
import pickle as pkl
import optax

def GenerateOfflineMultiTaskPolicy(task_embeddings_model:str, model_path:str, task_name:str, timesteps: int, algos:str,
                                   path: str, buffer_size:int, mode:str, n_steps:int, seed: int, policy_quality: List[str], latent_dim:int,
                                   reward_path: str=None, transition_path: str=None, behavior_path: str=None, data_augmentation:bool=False):
    if task_name == 'Multitask':
        task_name_list = MT10_TASK
    elif task_name == 'airsim':
        task_name_list = AIRSIM_TASK
    else:
        task_name_list = [task_name]

    if algos == 'TD3':
        model_algo = TD3
        policy = TD3Policy
    else:
        NotImplementedError()

    np.random.seed(seed)
    random.seed(seed)

    offline_datasets = OfflineDatasets()
    offline_datasets.load(os.getcwd() + '/single_task/offline_data/{}/{}.pkl'.format(task_name_list[0], policy_quality[0]))
    latents = np.zeros((latent_dim, ))
    key = jax.random.PRNGKey(seed)
    AE = True
    if task_name == 'airsim':
        action_space = gym.spaces.Box(low=np.full((3, ), -1), high=np.full((3, ), 1), dtype=np.float32)
        observation_space = gym.spaces.Dict({'obs': gym.spaces.Box(low=np.full(140, -1), high=np.full(140, -1), dtype=np.float32),
                                             'task': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim, ))})
        train_env = DummyEnv(observation_space=observation_space, action_space=action_space)
        test_env = DummyEnv(observation_space=observation_space, action_space=action_space)
    else:
        train_env = MetaWorldIndexedMultiTaskTester(mode='dict', seed=seed)
    trajectories = np.zeros((n_steps * (offline_datasets.state_size + offline_datasets.action_size + 1), ))
    states = np.zeros((offline_datasets.state_size, ))
    actions = np.zeros((offline_datasets.action_size, ))
    seq = np.zeros((8, ))

    if task_embeddings_model == 'TE':
        task_encoder_def = TaskEncoderAE(net_arch=[256, 256], latent_dim=latent_dim)
        task_encoder = Model.create(task_encoder_def, inputs=[key, trajectories])
        task_encoder = task_encoder.load(model_path)

    elif task_embeddings_model == 'PGTE':
        policy_task_encoder_def = TaskEncoderAE(net_arch=[256, 256], latent_dim=latent_dim)
        task_encoder = Model.create(policy_task_encoder_def, inputs=[key, trajectories])
        task_encoder = task_encoder.load(model_path)

        reward_decoder_def = RewardDecoder(net_arch=[256, 256, 1])
        transition_decoder_def = TransitionDecoder(net_arch=[256, 256, offline_datasets.state_size])

        reward_decoder = Model.create(reward_decoder_def, inputs=[key, states, actions, latents])
        transition_decoder = Model.create(transition_decoder_def, inputs=[key, states, actions, latents])

        reward_decoder = reward_decoder.load(reward_path)
        transition_decoder = transition_decoder.load(transition_path)
        if data_augmentation:
            behavior_decoder_def = BehaviorDecoder(net_arch=[256, 256, offline_datasets.action_size])
            behavior_decoder = Model.create(behavior_decoder_def, inputs=[key, states, latents, seq])
            behavior_decoder = behavior_decoder.load(behavior_path)
    else:
        NotImplementedError()
        return


    num_data_dict = default_dynamic_num_data
    replay_buffer = None

    offline_datasets = OfflineDatasets()
    for idx, t_n in enumerate(task_name_list):
        _offline_datasets = OfflineDatasets()
        _offline_datasets.load(os.getcwd() + '/single_task/offline_data/{}/{}.pkl'.format(t_n, policy_quality[idx]))
        _offline_datasets = _offline_datasets.sample(num_data_dict[policy_quality[idx]])
        offline_datasets.extend(_offline_datasets)

    key, rng = jax.random.split(key, 2)
    if task_embeddings_model == 'PGTE' and data_augmentation:
        replay_buffer = offline_datasets.get_trajectories_replay_buffer(rng, buffer_size, task_encoder, latent_dim, idx, 10, replay_buffer=replay_buffer, n_steps=n_steps, AE=AE, obs_space=train_env.observation_space['obs'],
                                                                        data_augmentation=True, transition_decoder=transition_decoder, reward_decoder=reward_decoder, behavior_decoder=behavior_decoder)
    else:
        replay_buffer = offline_datasets.get_trajectories_replay_buffer(rng, buffer_size, task_encoder, latent_dim, idx, 10, replay_buffer=replay_buffer, n_steps=n_steps, AE=AE, obs_space=train_env.observation_space['obs'],)
    if isinstance(replay_buffer, ModelBasedDictReplayBuffer):
        replay_buffer.set_data_point()
        replay_buffer.clean_model()

    mean, std = replay_buffer.normalize_states()
    print(mean, std)
    if task_name != 'arisim':
        train_env = TrajectoriesEncoderEnv(TimeLimitRewardMDP(train_env), task_encoder, latent_dim, seed=seed, mean=mean, std=std)
        test_env = TrajectoriesEncoderEnv(TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict')), task_encoder, latent_dim, n_steps=n_steps, seed=seed, mean=mean, std=std)
    if task_embeddings_model == 'PGTE' and data_augmentation:
        da = 'da'
    else:
        da = ''
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=os.path.join(path, 'models/{}_seed_{}_'.format('{}{}'.format(task_embeddings_model, algos), seed) + mode + '_' + da),
                                             name_prefix='model')


    # Generate RL Model
    policy_kwargs = dict(net_arch=[256, 256, 256], activation_fn=nn.relu)
    min_parameter = dict(alpha=2.5)
    kwargs = min_parameter

    model = model_algo(policy, train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=buffer_size,
                  train_freq=1, policy_kwargs=policy_kwargs, without_exploration=True,  learning_rate=1e-4, gradient_steps=1000,
                  tensorboard_log=os.path.join(path, 'tensorboard/{}'.format('{}{}_seed_{}_'.format(task_embeddings_model, algos, seed) + mode + '_' + da)),
                       **kwargs)

    model.reload_buffer = False
    model.replay_buffer = replay_buffer

    if task_name == 'airsim':
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=1000, tb_log_name='exp')
    else:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, eval_freq=200000, log_interval=1000,
                n_eval_episodes=500, eval_log_path=os.path.join(path, 'models/{}_seed_{}_'.format('{}{}'.format(task_embeddings_model, algos), seed) + mode+ '_' + da),
                eval_env=test_env, tb_log_name='exp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--algos', type=str, default='MTCQL')
    parser.add_argument('--model', type=str, default='PGTE')
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--buffer-size', type=int, default=2_000_000)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--path', type=str, default=os.getcwd() + '/results')
    parser.add_argument('--mode', type=str, default='replay_50')
    parser.add_argument('--n-steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num-data', type=str, default='static')
    args = parser.parse_args()

    # policy_quality = sampling_policy[args.mode]
    # np.random.shuffle(policy_quality)

    policy_quality = ['random']*10

    GenerateOfflineMultiTaskPolicy(task_embeddings_model=args.model, model_path=args.model_path, seed=args.seed, algos=args.algos,
                                   task_name=args.task_name, timesteps=args.timesteps, path=args.path,
                                   buffer_size=args.buffer_size, mode=args.mode, policy_quality=policy_quality,n_steps=args.n_steps,
                                   data_augmentation=args.data_augmentaiton)