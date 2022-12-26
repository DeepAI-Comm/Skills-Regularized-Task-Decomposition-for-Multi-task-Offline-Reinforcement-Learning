import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import jax
import jax.numpy as jnp
import functools
import optax

from typing import Tuple, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from jax_models import TaskEncoderAE, TransitionDecoder, RewardDecoder, Model, Params, InfoDict

LOG_STD_MAX = 2
LOG_STD_MIN = -10

@jax.jit
def normal_sampling(key:Any, task_latents_mu:jnp.ndarray, task_latents_log_std:jnp.ndarray):
    return task_latents_mu + jax.random.normal(key, shape=task_latents_log_std.shape) * jnp.exp(0.5 * task_latents_log_std)

def l2_loss(x):
    return (x ** 2).mean()

"""
  actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
"""

def compute_mmd(z: jnp.ndarray, z1:jnp.ndarray=None, reg_weight: float=100) -> jnp.ndarray:
    # Sample from prior (Gaussian) distribution
    key = jax.random.PRNGKey(0)
    batch_size = z.shape[0]
    reg_weight = reg_weight / (batch_size * (batch_size - 1))
    if z1 is None:
        prior_z =jax.random.normal(key, shape=z.shape)
    else:
        prior_z = z1

    prior_z__kernel = compute_inv_mult_quad(prior_z, prior_z)
    z__kernel = compute_inv_mult_quad(z, z)
    priorz_z__kernel = compute_inv_mult_quad(prior_z, z)

    mmd = reg_weight * prior_z__kernel.mean() + \
          reg_weight * z__kernel.mean() - \
          2 * reg_weight * priorz_z__kernel.mean()
    return mmd

def compute_inv_mult_quad(x1: jnp.ndarray, x2: jnp.ndarray, eps: float = 1e-7, latent_var: float = 2.) -> jnp.ndarray:
    D, N = x1.shape

    x1 = jnp.expand_dims(x1, axis=-2)  # Make it into a column tensor
    x2 = jnp.expand_dims(x2, axis=-3)  # Make it into a row tensor

    x1 = jnp.repeat(x1, D, axis=-2)
    x2 = jnp.repeat(x2, D, axis=-3)

    z_dim = x2.shape[-1]
    C = 2 * z_dim * latent_var
    kernel = C / (eps + C + jnp.sum((x1 - x2)**2, axis=-1))

    # Exclude diagonal elements
    result = jnp.sum(kernel) - jnp.sum(jnp.diag(kernel))

    return result


def task_encoder_update(key:Any, task_encoder: Model, reward_decoder: Model, transition_decoder: Model, trajectories: jnp.ndarray,
                         states: jnp.ndarray, actions: jnp.ndarray, next_state: jnp.array, rewards: jnp.ndarray):
    def task_encoder_loss(task_encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        task_latent = task_encoder.apply_fn({'params':task_encoder_params}, trajectories)
        pred_next_state = transition_decoder(states, actions, jnp.repeat(task_latent, repeats=4, axis=0))
        pred_rewards = reward_decoder(states, actions, jnp.repeat(task_latent, repeats=4, axis=0))

        reconstruction_loss = jnp.mean(jnp.sum(jnp.square(pred_next_state - next_state), axis=1), axis=0)\
                              + jnp.mean(jnp.sum(jnp.square(pred_rewards - rewards), axis=1), axis=0)

        reg_loss = compute_mmd(task_latent)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(task_encoder_params))
        loss = reconstruction_loss + reg_loss + 1e-3 * l2_reg

        return loss, {'reconstruction_loss': reconstruction_loss, 'reg_loss': reg_loss, 'task_latent': task_latent}

    new_task_encoder, info = task_encoder.apply_gradient(task_encoder_loss)
    return new_task_encoder, info

def decoder_update(task_latent:jnp.ndarray, reward_decoder: Model, transition_decoder: Model, trajectories: jnp.ndarray,
                   states: jnp.ndarray, actions: jnp.ndarray, next_states: jnp.array, rewards: jnp.ndarray):

    def reward_loss(reward_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_rewards = reward_decoder.apply_fn({'params': reward_decoder_params}, states, actions, jnp.repeat(task_latent, repeats=4, axis=0))
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(reward_decoder_params))
        loss = jnp.mean(jnp.sum(jnp.square(pred_rewards - rewards), axis=1), axis=0)+ 1e-3 * l2_reg
        return loss, {'reward_loss': loss}

    def transition_loss(transition_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_next_states = transition_decoder.apply_fn({'params': transition_decoder_params}, states, actions, jnp.repeat(task_latent, repeats=4, axis=0))
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(transition_decoder_params))
        loss = jnp.mean(jnp.sum(jnp.square(pred_next_states - next_states), axis=1), axis=0)+ 1e-3 * l2_reg
        return loss, {'transition_loss': loss}

    new_reward_decoder, reward_info = reward_decoder.apply_gradient(reward_loss)
    new_transition_decoder, transition_info = transition_decoder.apply_gradient(transition_loss)

    return new_reward_decoder, new_transition_decoder, {**reward_info, **transition_info}

@functools.partial(jax.jit)
def _update_te(rng:Any, task_encoder: Model, reward_decoder: Model, transition_decoder: Model, trajectories:jnp.ndarray,
               states:jnp.ndarray, actions:jnp.ndarray, rewards: jnp.ndarray, next_states:jnp.ndarray) \
        -> Tuple[Any, Model, Model, Model, InfoDict]:

    rng, key1, key2 = jax.random.split(rng, 3)
    actions = jnp.reshape(actions, (-1, actions.shape[-1]))
    states = jnp.reshape(states, (-1, states.shape[-1]))
    next_states = jnp.reshape(next_states, (-1, next_states.shape[-1]))
    rewards = jnp.reshape(rewards, (-1, rewards.shape[-1]))
    new_task_encoder, task_info = task_encoder_update(key1, task_encoder, reward_decoder, transition_decoder,
                                                      trajectories, states, actions, next_states, rewards)

    new_reward_decoder, new_transition_decoder, decoder_info = \
        decoder_update(task_info['task_latent'], reward_decoder, transition_decoder, trajectories, states, actions,
                       next_states, rewards)

    return rng, new_task_encoder, new_reward_decoder, new_transition_decoder, {**task_info, **decoder_info}


class TETrainer(object):
    def __init__(self, datasets, args):
        # dataset = ConcatDataset(*datasets)
        self.train_loader = DataLoader(datasets, batch_size=args.num_batch, shuffle=True, )
        self.num_task = len(datasets)
        self.num_batch = args.num_batch

        self.key = jax.random.PRNGKey(args.seed)
        self.mode = args.mode

        state_size = datasets.state_size
        action_size = datasets.action_size
        size = datasets.size

        states = jnp.zeros((state_size,))
        actions = jnp.zeros((action_size,))
        trajectories = jnp.zeros((size,))
        latents = jnp.zeros((args.latent_dim, ))

        # Define network
        task_encoder_def = TaskEncoderAE(net_arch=[256, 256], latent_dim=args.latent_dim)
        reward_decoder_def = RewardDecoder(net_arch=[256, 256, 1])
        transition_decoder_def = TransitionDecoder(net_arch=[256, 256, state_size])

        # create model
        self.key, task_encoder_key, reward_decoder_key, transition_decoder_key = jax.random.split(self.key, 4)
        self.task_encoder = Model.create(task_encoder_def, inputs=[task_encoder_key, trajectories],
                                         tx=optax.adam(learning_rate=args.lr))
        self.reward_decoder = Model.create(reward_decoder_def, inputs=[reward_decoder_key, states, actions, latents],
                                           tx=optax.adam(learning_rate=args.lr))
        self.transition_decoder = Model.create(transition_decoder_def, inputs=[transition_decoder_key, states, actions, latents],
                                               tx=optax.adam(learning_rate=args.lr))


    def train(self, epoch):
        for i in range(epoch):
            tbar = tqdm(self.train_loader)
            transition_loss, reward_loss, kld_loss = 0, 0, 0
            for idx, sample in enumerate(tbar):
                traj, states, actions, rewards, next_states, _, task_id, _, _ = sample

                self.key, key = jax.random.split(self.key, 2)
                self.key, new_task_encoder, new_reward_decoder, new_transition_decoder, info = \
                    _update_te(key, self.task_encoder, self.reward_decoder, self.transition_decoder, traj.numpy(),
                               states.numpy(), actions.numpy(), rewards.numpy(), next_states.numpy())

                self.task_encoder = new_task_encoder
                self.reward_decoder = new_reward_decoder
                self.transition_decoder = new_transition_decoder

                transition_loss += info['transition_loss']
                reward_loss += info['reward_loss']
                kld_loss += info['reg_loss']

                tbar.set_description('Epochs %d: Trans. loss: %.4f Rew. loss: %.4f Reg loss: %.4f'%(i, transition_loss / (idx + 1), reward_loss / (idx + 1), kld_loss / (idx + 1)))

    def save(self, path, seed, num_data):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.task_encoder.save(os.path.join(path, 'task_encoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        self.reward_decoder.save(os.path.join(path, 'reward_decoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        self.transition_decoder.save(os.path.join(path, 'transition_decoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        return os.path.join(path, 'task_encoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data))