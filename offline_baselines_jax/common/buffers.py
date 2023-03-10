import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

import numpy as np
from gym import spaces
import tqdm

from offline_baselines_jax.common.preprocessing import get_action_dim, get_obs_shape
from offline_baselines_jax.common.type_aliases import (
    DictReplayBufferSamples,
    ReplayBufferSamples,
    Params,
)

import functools
import jax
import jax.numpy as jnp
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

@functools.partial(jax.jit, static_argnames=('reward_fn'))
def reward_transition_function(reward_fn: Callable[..., Any], encoder_params: Params, state: jnp.ndarray,
                               action: jnp.ndarray, task_latents: jnp.ndarray) -> jnp.ndarray:
    out = reward_fn({'params': encoder_params}, state, action, task_latents)
    return out

@functools.partial(jax.jit, static_argnames=('reward_fn'))
def policy_function(reward_fn: Callable[..., Any], encoder_params: Params, state: jnp.ndarray,
                    task_latents: jnp.ndarray) -> jnp.ndarray:
    out = reward_fn({'params': encoder_params}, state, task_latents)
    return out

@jax.jit
def normal_sampling(key:Any, task_latents_mu:jnp.ndarray, task_latents_log_std:jnp.ndarray):
    rng, key = jax.random.split(key)
    return task_latents_mu + jax.random.normal(key, shape=(task_latents_log_std.shape[-1], )) * jnp.exp(0.5 * task_latents_log_std)

@jax.jit
def task_sampling(key: Any):
    task_latents = jax.random.randint(key, shape=(1,), minval=0, maxval=10)
    task_latents = jax.nn.one_hot(task_latents, 10)
    return task_latents

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(data))


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        task_embedding_lambda: bool = False,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        use_embeddings: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.key = jax.random.PRNGKey(0)
        self.task_embedding_lambda = task_embedding_lambda
        self.use_embeddings = use_embeddings

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        if task_embedding_lambda:
            self.task_embeddings = {'task_latents_mu': np.zeros_like(self.observations['task']),
                                    'task_latents_log_std': np.zeros_like(self.observations['task']),
                                    'next_task_latents_mu': np.zeros_like(self.next_observations['task']),
                                    'next_task_latents_log_std': np.zeros_like(self.next_observations['task'])}

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        task_embeddings: Dict[str, np.ndarray] = None,
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        if self.task_embedding_lambda:
            for key in self.task_embeddings.keys():
                self.task_embeddings[key][self.pos] = np.array(task_embeddings[key]).copy()

        # Same reshape, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])


        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)


    def normalize_states(self, eps=1e-3):
        key = 'obs'
        mean = np.mean(self.observations[key][:self.pos - 1], axis=0)
        std = np.std(self.observations[key][:self.pos - 1], axis=0)
        self.observations[key] = (self.observations[key] - np.mean(self.observations[key][:self.pos - 1], axis=0)) / (np.std(self.observations[key][: self.pos - 1], axis=0) + eps)
        self.next_observations[key] = (self.next_observations[key] - np.mean(self.next_observations[key][:self.pos - 1], axis=0)) / (np.std(self.next_observations[key][: self.pos - 1], axis=0) + eps)

        return mean, std

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()})

        observations = {key: obs for key, obs in obs_.items()}
        next_observations = {key: obs for key, obs in next_obs_.items()}
        if self.task_embedding_lambda:
            self.key, task_key, next_key = jax.random.split(self.key, 3)
            task_latents_mu = self.task_embeddings['task_latents_mu'][batch_inds, env_indices, :]
            task_latents_log_std = self.task_embeddings['task_latents_log_std'][batch_inds, env_indices, :]
            next_task_latents_mu = self.task_embeddings['next_task_latents_mu'][batch_inds, env_indices, :]
            next_task_latents_log_std = self.task_embeddings['next_task_latents_log_std'][batch_inds, env_indices, :]
            observations['task'] = normal_sampling(task_key, task_latents_mu, task_latents_log_std)
            next_observations['task'] = normal_sampling(next_key, next_task_latents_mu, next_task_latents_log_std)
        elif self.use_embeddings:
            pass
            self.key, task_key, next_task_key= jax.random.split(self.key, 3)
            noise = jnp.clip(normal_sampling(task_key, jnp.zeros_like(observations['task']), jnp.ones_like(observations['task']) * -4.6), a_min=-0.5, a_max=0.5)
            next_noise = jnp.clip(normal_sampling(next_task_key, jnp.zeros_like(observations['task']), jnp.ones_like(observations['task']) * -4.6), a_min=-0.5, a_max=0.5)

            observations['task'] += noise
            next_observations['task'] += next_noise
        else:
            pass

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.actions[batch_inds, env_indices],
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices]).reshape(-1, 1),
            rewards=self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

    def _reload_task_latents(self):
        return


class ModelBasedDictReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        behavior_model = None,
        transition_model = None,
        reward_model = None,
        use_embeddings: bool = True,
        task_embedding_lambda: bool = False,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,

    ):
        super(ModelBasedDictReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs, task_embedding_lambda,
                                                         optimize_memory_usage, handle_timeout_termination)
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.behavior_model = behavior_model
        self.use_embeddings = use_embeddings
        self.data_point = 0
        self.key = jax.random.PRNGKey(0)

    def set_data_point(self):
        self.data_point = self.pos
        self.buffer_size = int(self.data_point * 1.2)

    def clean_model(self):
        self.behavior_model = None
        self.transition_model = None
        self.reward_model = None

    def get_model(self, transition_model, reward_model, behavior_model=None, seed=777):
        self.behavior_model = behavior_model
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.key = jax.random.PRNGKey(seed)

    def add_augmentation(self, num_of_data: int, env: Optional[VecNormalize] = None, verbose=0):
        if verbose:
            progress_bar = tqdm.tqdm(range(num_of_data))
        else:
            progress_bar = range(num_of_data)
        for _ in progress_bar:
            upper_bound = self.data_point
            batch_inds = np.random.randint(0, upper_bound, size=(1, ))
            env_indices = np.random.randint(0, high=self.n_envs, size=(1,))

            # Normalize if needed and remove extra dimension (we are using only one env for now)
            obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()})
            next_obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()})
            observations = {key: obs for key, obs in obs_.items()}
            next_observations = {key: obs for key, obs in next_obs_.items()}
            # rewards = self.rewards[batch_inds, env_indices]

            _obs = observations['obs']
            task_latents = observations['task']
            next_task_latents = next_observations['task']
            dones = self.dones[batch_inds, env_indices]

            if self.use_embeddings:
                self.key, task_rng, rew_rng, pol_rng = jax.random.split(self.key, 4)
                policy_latents = normal_sampling(pol_rng, jnp.zeros_like(task_latents), jnp.zeros_like(task_latents))
                # noise = normal_sampling(task_rng, jnp.zeros_like(task_latents), jnp.ones_like(task_latents) * -4.6)
                _task_latents = task_latents
                task_latents = _task_latents
                next_task_latents = next_task_latents


            infos = [dict()]
            for i in range(1):
                obs = _obs
                self.key, task_rng, rew_rng, pol_rng = jax.random.split(self.key, 4)
                if self.use_embeddings:
                    action = policy_function(self.behavior_model.apply_fn, self.behavior_model.params, obs, policy_latents)
                else:
                    _task_latents = task_sampling(task_rng)
                    action = self.actions[batch_inds, env_indices]
                    action = normal_sampling(pol_rng, jnp.zeros_like(action), jnp.zeros_like(action))

                reward, = reward_transition_function(self.reward_model.apply_fn, self.reward_model.params, obs, action, _task_latents)
                next_state = reward_transition_function(self.transition_model.apply_fn, self.transition_model.params, obs, action, _task_latents)
                # reward = jnp.clip(reward, a_max=rewards * 0.99)
                _obs = next_state

                if self.task_embedding_lambda:
                    task_embeddings = dict()
                    task_embeddings['task_latents_mu'] = _task_latents
                    task_embeddings['task_latents_log_std'] = jnp.ones_like(_task_latents) * -5
                    task_embeddings['next_task_latents_mu'] = _task_latents
                    task_embeddings['next_task_latents_log_std'] = jnp.ones_like(_task_latents) * -5
                else:
                    task_embeddings = None

                self.add({'obs': obs, 'task': task_latents}, {'obs': next_state, 'task': next_task_latents}, action, reward,
                         dones, infos, task_embeddings=task_embeddings)


    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        task_embeddings: Dict[str, np.ndarray] = None,
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        if self.task_embedding_lambda:
            for key in self.task_embeddings.keys():
                self.task_embeddings[key][self.pos] = np.array(task_embeddings[key]).copy()

        # Same reshape, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])


        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = self.data_point


    # def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    #     """
    #     Sample elements from the replay buffer.
    #     Custom sampling when using memory efficient variant,
    #     as we should not sample the element with index `self.pos`
    #     See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    #
    #     :param batch_size: Number of element to sample
    #     :param env: associated gym VecEnv
    #         to normalize the observations/rewards when sampling
    #     :return:
    #     """
    #     # if not self.optimize_memory_usage:
    #     #     return super().sample(batch_size=batch_size, env=env)
    #     # Do not sample the element with index `self.pos` as the transitions is invalid
    #     # (we use only one array to store `obs` and `next_obs`)
    #     if self.full:
    #         prob = np.ones(self.buffer_size)
    #         prob[:self.data_point] = prob[:self.data_point] * 5
    #         prob = prob / sum(prob)
    #         batch_inds = np.random.choice(np.arange(self.buffer_size), size=batch_size, p=prob, replace=False)
    #     else:
    #         prob = np.ones(self.pos)
    #         prob[:self.data_point] = prob[:self.data_point] * 5
    #         prob = prob / sum(prob)
    #         batch_inds = np.random.choice(np.arange(self.pos), size=batch_size, p=prob, replace=False)
    #     return self._get_samples(batch_inds, env=env)


class TaskDictReplayBuffer(object):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            task_embedding_lambda: bool = False,
            handle_timeout_termination: bool = True,
            num_tasks: int = 10,
    ):
        self.replay_buffers = []
        self.num_tasks = num_tasks
        for _ in range(num_tasks):
            self.replay_buffers.append(DictReplayBuffer(buffer_size//num_tasks, observation_space, action_space,
                                                        n_envs=n_envs, optimize_memory_usage=optimize_memory_usage,
                                                        handle_timeout_termination=handle_timeout_termination,
                                                        task_embedding_lambda=task_embedding_lambda, use_embeddings=False))

    def _reload_task_latents(self):
        for i in range(self.num_tasks):
            self.replay_buffers[i]._reload_task_latents()

    def normalize_states(self):
        list_mean = []
        list_std = []
        for buffer in self.replay_buffers:
            mean, std = buffer.normalize_states()
            list_mean.append(mean)
            list_std.append(std)
        return list_mean, list_std


    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            task_embeddings: Dict[str, np.ndarray] = None,
    ) -> None:
        self.replay_buffers[infos[0]['task']].add(obs, next_obs, action, reward, done, infos, task_embeddings)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        observations = {'task': [], 'obs': []}
        actions = []
        next_observations = {'task': [], 'obs': []}
        rewards = []
        dones = []
        for i in range(self.num_tasks):
            batch = self.replay_buffers[i].sample(batch_size//self.num_tasks, env)
            for key, data in batch.observations.items():
                observations[key].append(data)
            actions.append(batch.actions)
            for key, data in batch.next_observations.items():
                next_observations[key].append(data)
            rewards.append(batch.rewards)
            dones.append(batch.dones)

        for key, data in observations.items():
            observations[key] = np.concatenate(data, axis=0)
        for key, data in next_observations.items():
            next_observations[key] = np.concatenate(data, axis=0)

        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)

        return DictReplayBufferSamples(observations=observations, actions=actions, next_observations=next_observations,
                                       dones=dones, rewards=rewards)
