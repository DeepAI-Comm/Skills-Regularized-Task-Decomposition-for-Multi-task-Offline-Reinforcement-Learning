import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import numpy as np
import random
import torch

from cfg.sampling_policy_cfg import sampling_policy, default_dynamic_num_data
from envs.env_dict import MT10_TASK, MT8_TASK, AIRSIM_TASK
from offline_data.offline_data_collector import OfflineDatasets
from offline_data.dataloader import Trajectories
from task_embeddings_trainer_jax import TETrainer
from policy_task_embedding_trainer_jax import PGTETrainer
from policy_train import GenerateOfflineMultiTaskPolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment setting')
    parser.add_argument('--model', type=str, default='PGTE')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-batch', type=int, default=2048)
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--buffer-size', type=int, default=2_000_000)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--latent-dim', type=int, default=4)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--mode', type=str, default='replay')
    parser.add_argument('--path', type=str, default='./results')
    parser.add_argument('--data-augmentation', action='store_true')
    parser.add_argument('--n-steps', type=int, default=4)
    parser.add_argument('--embedding-skip', action='store_true')
    parser.add_argument('--algos', type=str, default='TD3')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    policy_quality = sampling_policy[args.mode]
    np.random.shuffle(policy_quality)
    num_data = default_dynamic_num_data


    reward_path = None
    transition_path = None
    behavior_path = None

    if args.task_name == 'Multitask':
        task_list = MT10_TASK
    elif args.task_name == 'mt8':
        task_list = MT8_TASK
    elif args.task_name == 'airsim':
        task_list = AIRSIM_TASK
    else:
        task_list = [args.task_name]

    if not args.embedding_skip:
        datasets = []
        for idx, task_name in enumerate(task_list):
            path = './single_task/offline_data/{}/{}.pkl'.format(task_name, policy_quality[idx])
            # Load Dataloader
            task_replay_buffer = OfflineDatasets()
            task_replay_buffer.load(path)
            task_replay_buffer = task_replay_buffer.sample(num_data[policy_quality[idx]])
            datasets.append(task_replay_buffer)

        datasets = Trajectories(datasets, n_steps=args.n_steps, jax=True)

        if args.model == 'PGTE':
            trainer = PGTETrainer(datasets, args)
            trainer.train(args.epochs)
            model_path, reward_path, transition_path, behavior_path = trainer.save('./results/models/PGTE', seed)
        elif args.model == 'TE':
            trainer = TETrainer(datasets, args)
            trainer.train(args.epochs)
            model_path = trainer.save('./results/models/TE', seed)
        else:
            NotImplementedError()
            model_path = None
    else:
        if args.model == 'TE':
            model_path = os.path.join('./results/models/TE', 'task_encoder_{}_seed_{}.jax'.format(args.mode, args.seed))
        elif args.model == 'PGTE':
            model_path =  os.path.join('./results/models/PGTE', 'policy_task_encoder_{}_seed_{}.jax'.format(args.mode, args.seed))
            reward_path =  os.path.join('./results/models/PGTE', 'reward_decoder_{}_seed_{}.jax'.format(args.mode, args.seed))
            transition_path =  os.path.join('./results/models/PGTE', 'transition_decoder_{}_seed_{}.jax'.format(args.mode, args.seed))
            behavior_path =  os.path.join('./results/models/PGTE', 'behavior_decoder_{}_seed_{}.jax'.format(args.mode, args.seed))
        else:
            model_path = None


    GenerateOfflineMultiTaskPolicy(task_embeddings_model=args.model, model_path=model_path, seed=args.seed, algos=args.algos,
                                   task_name=args.task_name, timesteps=args.timesteps, path=args.path, policy_quality=policy_quality,
                                   buffer_size=args.buffer_size, mode=args.mode, n_steps=args.n_steps, latent_dim=args.latent_dim, reward_path=reward_path,
                                   transition_path=transition_path, behavior_path=behavior_path, data_augmentation=args.data_augmentation)