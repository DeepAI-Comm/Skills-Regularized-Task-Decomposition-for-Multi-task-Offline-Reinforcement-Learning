import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import gym
import numpy as np
from metaworld import MT10
import random
from envs.env_dict import MT10_TASK


class SingleTask(gym.Env):
    def __init__(self, seed: int, task_name: str):
        random.seed(seed)
        self.mt = MT10(seed=seed)

        self.env_cls = self.mt.train_classes[task_name]
        self.task_name = task_name

        self.env = self.env_cls()
        task = random.choice([task for task in self.mt.train_tasks if task.env_name == self.task_name])
        self.env.set_task(task)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward / 10, done, info

    def reset(self):
        self.env = self.env_cls()
        task = random.choice([task for task in self.mt.train_tasks if task.env_name == self.task_name])
        self.env.set_task(task)
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)


class MetaWorldIndexedMultiTaskTester(gym.Env):
    def __init__(self, max_episode_length=200, task_name_list=MT10_TASK, mode='concat', seed=777):
        self.MTenv = MT10(seed=seed)
        self.env = None
        self._task_idx = 0
        self._epi_index = 0

        self.max_episode_length = max_episode_length
        self.mode = mode
        self.task_name_list = task_name_list

        env_cls = self.MTenv.train_classes[task_name_list[0]]
        env = env_cls()
        task = random.choice([task for task in self.MTenv.train_tasks if task.env_name == task_name_list[0]])
        env.set_task(task)

        if mode == 'concat':
            self.observation_space = gym.spaces.Box(low=np.concatenate([env.observation_space.low, np.zeros(len(self.task_name_list))]),
                                                high=np.concatenate([env.observation_space.high, np.ones(len(self.task_name_list))]))
        elif mode == 'dict':
            self.observation_space = gym.spaces.Dict({'obs': env.observation_space,
                                                      'task': gym.spaces.Box(low=np.zeros(len(self.task_name_list)),
                                                                             high=np.ones(len(self.task_name_list)))})
        else:
            NotImplementedError()

        self.action_space = env.action_space
        self.task_change = True

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        context = np.zeros(len(self.task_name_list))
        context[self._task_idx % len(self.task_name_list)] = 1

        if self.mode == 'concat':
            observation = np.concatenate([state, context])
        elif self.mode == 'dict':
            observation = {'obs': state, 'task': context}
        else:
            NotImplementedError()
            return

        info['task_name'] = self.task_name_list[self._task_idx % len(self.task_name_list)]
        info['task'] = self._task_idx % len(self.task_name_list)

        return observation, reward / 10, done, info

    def set_task(self, task_num):
        self._task_idx = task_num
        self.task_change = False

    def reset(self):
        if self.task_change:
            self._task_idx = self._task_idx + 1
            self._epi_index = self._task_idx // len(self.task_name_list)

        _task_idx = self._task_idx % len(self.task_name_list)
        task_name = self.task_name_list[_task_idx]
        env_cls = self.MTenv.train_classes[task_name]
        self.env = env_cls()
        tasks = [task for task in self.MTenv.train_tasks if task.env_name == task_name]
        self.env.set_task(tasks[self._epi_index % 50])

        state = self.env.reset()
        context = np.zeros(len(self.task_name_list))
        context[_task_idx] = 1

        if self.mode == 'concat':
            observation = np.concatenate([state, context])
        elif self.mode == 'dict':
            observation = {'obs': state, 'task': context}
        else:
            NotImplementedError()
            return

        return observation

    def render(self, mode="human"):
        self.env.render(mode)

    @property
    def num_tasks(self):
        return len(self.task_name_list)