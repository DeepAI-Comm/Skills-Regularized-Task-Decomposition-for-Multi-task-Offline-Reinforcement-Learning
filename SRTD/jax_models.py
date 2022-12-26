import torch
# import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Dict, List, Tuple, Type, Union, Sequence, Callable, Optional, Any

import os
import jax
import flax
import optax
import jax.numpy as jnp
import flax.linen as nn

LOG_STD_MAX = 2
LOG_STD_MIN = -10

Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, float]

@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        assert (loss_fn is not None or grads is not None,
                'Either a loss function or grads must be specified.')
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert (has_aux,
                    'When grads are provided, expects no aux outputs.')

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


class MLP(nn.Module):
    net_arch: List[int]
    _eval: bool = False
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    last_layer_activation: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.net_arch):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.net_arch) or self.last_layer_activation:
                x = self.activation_fn(x)
        return x


def default_init():
    return nn.initializers.he_normal()

class SimpleLinearModel(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.latent_dim, kernel_init=default_init())(inputs)
        return x

class TaskEncoder(nn.Module):
    net_arch: List[int]
    latent_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        task_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(trajectories)

        task_latent_mu = nn.Dense(self.latent_dim)(task_hidden)
        task_latent_log_std = nn.Dense(self.latent_dim)(task_hidden)
        return task_latent_mu, task_latent_log_std

class TaskEncoderAE(nn.Module):
    net_arch: List[int]
    latent_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        task_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(trajectories)
        task_latent = nn.Dense(self.latent_dim)(task_hidden)
        return task_latent

class TaskEncoderPrior(nn.Module):
    net_arch: List[int]
    latent_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, task_id:jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([states, task_id], -1)
        task_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(states)
        task_latent = nn.Dense(self.latent_dim)(task_hidden)
        return task_latent

class PolicyEncoderAE(nn.Module):
    net_arch: List[int]
    latent_dim : int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([states, actions], -1)
        policy_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(inputs)
        policy_latent = nn.Dense(self.latent_dim)(policy_hidden)

        # policy_hidden_2 = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(actions[..., :-4])
        # policy_latent_2 = nn.Dense(self.latent_dim)(policy_hidden_2)
        return policy_latent

class PolicyTaskEncoderAE(nn.Module):
    net_arch: List[int]
    task_net_arch: List[int]
    policy_net_arch: List[int]
    latent_dim : int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        hidden_value = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(trajectories)
        task_hidden = MLP(net_arch=self.task_net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(hidden_value)
        policy_hidden = MLP(net_arch=self.policy_net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(hidden_value)

        task_latent= nn.Dense(self.latent_dim)(task_hidden)
        policy_latent = nn.Dense(self.latent_dim)(policy_hidden)

        return task_latent, policy_latent,

class PolicyEncoder(nn.Module):
    net_arch: List[int]
    latent_dim : int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([states, actions], -1)
        policy_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(inputs)

        policy_latent_mu = nn.Dense(self.latent_dim)(policy_hidden)
        policy_latent_log_std = nn.Dense(self.latent_dim)(policy_hidden)
        return  policy_latent_mu, policy_latent_log_std

class PolicyTaskEncoder(nn.Module):
    net_arch: List[int]
    task_net_arch: List[int]
    policy_net_arch: List[int]
    latent_dim : int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        hidden_value = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(trajectories)
        task_hidden = MLP(net_arch=self.task_net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(hidden_value)
        policy_hidden = MLP(net_arch=self.policy_net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(hidden_value)

        task_latent_mu = nn.Dense(self.latent_dim)(task_hidden)
        task_latent_log_std = nn.Dense(self.latent_dim)(task_hidden)
        policy_latent_mu = nn.Dense(self.latent_dim)(policy_hidden)
        #policy_latent_log_std = nn.Dense(self.latent_dim)(policy_hidden)

        return task_latent_mu, task_latent_log_std, policy_latent_mu#, policy_latent_log_std


class RewardDecoder(nn.Module):
    # [256, 256, 1]
    net_arch: List[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray, latents: jnp.ndarray):
        inputs = jnp.concatenate([states, actions, latents], -1)
        pred_rewards = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn)(inputs)
        return pred_rewards


class TransitionDecoder(nn.Module):
    # [256, 256, 8]
    net_arch: List[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray, latents: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([states, actions, latents], -1)
        next_states_diff = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn)(inputs)
        return next_states_diff


class BehaviorDecoder(nn.Module):
    # [256, 256, 8]
    net_arch: List[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, latents: jnp.ndarray, seq:jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([states, latents], -1)
        pred_actions = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn)(inputs)
        return pred_actions

class TaskReconstruction(nn.Module):
    # [256, 256, 10]
    net_arch: List[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray, latents: jnp.ndarray):
        inputs = jnp.concatenate([states, actions, latents], -1)
        pred_rewards = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn)(inputs)
        return pred_rewards

class SumRewardDecoder(nn.Module):
    # [256, 256, 1]
    net_arch: List[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, task_latents: jnp.ndarray, policy_latents: jnp.ndarray):
        inputs = jnp.concatenate([task_latents, policy_latents], -1)
        pred_rewards = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn)(inputs)
        return pred_rewards
