"""Policies: abstract base class and concrete implementations."""

from typing import Any, Optional, Tuple, Union, Callable,Sequence, List

import optax
import gym
import flax
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import os

from offline_baselines_jax.common.type_aliases import Params


def grad_flatten(grad:Params):
    return jnp.concatenate([g.flatten() for g in jax.tree_flatten(grad)[0]])


def grad_unflatten(grad:jnp.ndarray, original_grad:Params):
    unflatten_layer_value = []
    leaves, struct = jax.tree_flatten(original_grad)
    temp_idx = 0
    for leave in leaves:
        num = leave.size
        unflatten_layer_value.append(grad[temp_idx: temp_idx + num].reshape(leave.shape))
        temp_idx = temp_idx + num

    return jax.tree_unflatten(struct, unflatten_layer_value)


def pc_grad(key: Any, grads: jnp.ndarray):
    grads_2 = jnp.copy(grads)
    for idx, g_i in enumerate(grads):
        jax.random.permutation(key, grads_2, independent=True)
        for g_j in grads_2:
            g_i_g_j = jnp.dot(g_i, g_j)
            grads = grads.at[idx].set(g_i - (jnp.clip((g_i_g_j), a_max=0) / (jnp.linalg.norm(g_j, ord=2) ** 2)) * g_j)
    return jnp.mean(grads, axis=0)


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

        return cls(step=1, apply_fn=model_def.apply,
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

        updates, new_opt_state = self.tx.update(grads, self.opt_state,self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model


    def PCGrad(self,
               key:Any,
            loss_fn: Optional[List[Callable[[Params], Any]]] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:

        grad_list = []
        for loss in loss_fn:
            grad_fn = jax.grad(loss, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
            flattened_grad = jnp.expand_dims(grad_flatten(grads), axis=0)
            grad_list.append(flattened_grad)

        task_grad = jnp.concatenate(grad_list)
        grad_surgery = pc_grad(key, task_grad)
        real_grads = grad_unflatten(grad_surgery, grads)
        updates, new_opt_state = self.tx.update(real_grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model


    def save_dict(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))
        return self.params

    def load_dict(self, params: bytes) -> 'Model':
        params = flax.serialization.from_bytes(self.params, params)
        return self.replace(params=params)