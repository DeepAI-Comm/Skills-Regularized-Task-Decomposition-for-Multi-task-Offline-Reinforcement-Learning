import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import jax
import jax.numpy as jnp
import functools
import optax
import numpy as np

from typing import Tuple, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from jax_models import PolicyEncoderAE, TaskEncoderAE, TransitionDecoder, RewardDecoder, BehaviorDecoder, Model, Params, InfoDict

def l2_loss(x):
    return (x ** 2).mean()

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


def policy_task_encoder_update(task_encoder: Model, policy_encoder:Model, reward_decoder: Model, transition_decoder: Model,
                               behavior_decoder: Model, trajectories: jnp.ndarray, states: jnp.ndarray, actions: jnp.ndarray,
                               next_state: jnp.array, rewards: jnp.ndarray, sum_rewards:jnp.ndarray, prev_states: jnp.ndarray,
                               prev_actions: jnp.ndarray, seq: jnp.ndarray):

    def task_encoder_loss(task_encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        task_latent = task_encoder.apply_fn({'params': task_encoder_params}, trajectories)
        policy_latent = policy_encoder(jnp.reshape(prev_states, (prev_states.shape[0], -1)), jnp.reshape(prev_actions, (prev_actions.shape[0], -1)))

        pred_next_state = transition_decoder(states, actions, jnp.repeat(task_latent, repeats=4, axis=0))
        pred_rewards = reward_decoder(states, actions, jnp.repeat(task_latent, repeats=4, axis=0))

        reconstruction_loss = jnp.mean(jnp.sum(jnp.square(pred_next_state - next_state), axis=1), axis=0)\
                              + jnp.mean(jnp.sum(jnp.square(pred_rewards - rewards), axis=1), axis=0)

        reg_loss = compute_mmd(task_latent)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(task_encoder_params))
        policy_embedding_loss = (sum_rewards * jnp.sum(jnp.square(task_latent - jax.lax.stop_gradient(policy_latent)), axis=-1)).mean()
        loss = reconstruction_loss + reg_loss + policy_embedding_loss*1e-2 + 1e-3 * l2_reg
        return loss, {'task_reg_loss': reg_loss, 'policy_embedding_loss': policy_embedding_loss, 'task_latent': task_latent}

    def policy_encoder_loss(policy_encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        policy_latent = policy_encoder.apply_fn({'params': policy_encoder_params}, jnp.reshape(prev_states, (prev_states.shape[0], -1)), jnp.reshape(prev_actions, (prev_actions.shape[0], -1)))
        pred_actions = behavior_decoder(jnp.reshape(prev_states, (-1, prev_states.shape[-1])), jnp.repeat(policy_latent, repeats=8, axis=0), seq)

        reconstruction_loss = jnp.mean(jnp.sum(jnp.square(pred_actions - jnp.reshape(prev_actions, (-1, prev_actions.shape[-1]))), axis=1), axis=0)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(policy_encoder_params))
        reg_loss = compute_mmd(policy_latent)

        loss = reconstruction_loss + reg_loss + l2_reg * 1e-3
        return loss, {'policy_reg_loss': reg_loss, 'policy_latent': policy_latent}

    new_task_encoder, task_info = task_encoder.apply_gradient(task_encoder_loss)
    new_policy_encoder, policy_info = policy_encoder.apply_gradient(policy_encoder_loss)
    return new_task_encoder, new_policy_encoder, {**task_info, **policy_info}

def decoder_update(task_latent:jnp.ndarray, policy_latent: jnp.ndarray, reward_decoder: Model, transition_decoder: Model, behavior_decoder: Model,
                   states: jnp.ndarray, actions: jnp.ndarray, next_states: jnp.array, rewards: jnp.ndarray,
                   prev_states: jnp.ndarray, prev_actions: jnp.ndarray, seq: jnp.ndarray):

    def reward_loss(reward_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_rewards = reward_decoder.apply_fn({'params': reward_decoder_params}, states, actions, jnp.repeat(task_latent, repeats=4, axis=0))

        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(reward_decoder_params))
        loss = jnp.mean(jnp.sum(jnp.square(pred_rewards - rewards), axis=1), axis=0)+ 1e-3 * l2_reg
        return loss, {'reward_loss': loss}

    def transition_loss(transition_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_next_states = transition_decoder.apply_fn({'params': transition_decoder_params}, states, actions, jnp.repeat(task_latent, repeats=4, axis=0))
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(transition_decoder_params))
        loss = jnp.mean(jnp.sum(jnp.square(pred_next_states - next_states), axis=1), axis=0) + 1e-3 * l2_reg
        return loss, {'transition_loss': loss }

    def behavior_loss(behavior_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_actions = behavior_decoder.apply_fn({'params': behavior_decoder_params}, jnp.reshape(prev_states, (-1, prev_states.shape[-1])), jnp.repeat(policy_latent, repeats=8, axis=0), seq)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(behavior_decoder_params))
        loss = jnp.mean(jnp.sum(jnp.square(pred_actions - jnp.reshape(prev_actions, (-1, prev_actions.shape[-1]))), axis=1), axis=0) + 1e-3 * l2_reg
        return loss, {'behavior_loss': loss, 'pred_actions': pred_actions}

    new_reward_decoder, reward_info = reward_decoder.apply_gradient(reward_loss)
    new_transition_decoder, transition_info = transition_decoder.apply_gradient(transition_loss)
    new_behavior_decoder, behavior_info = behavior_decoder.apply_gradient(behavior_loss)
    return new_reward_decoder, new_transition_decoder, new_behavior_decoder, {**reward_info, **transition_info, **behavior_info}

@functools.partial(jax.jit)
def _update_pgte(rng:Any, task_encoder: Model, policy_encoder:Model, reward_decoder: Model, transition_decoder: Model, behavior_decoder:Model,
                 trajectories:jnp.ndarray, states:jnp.ndarray, actions:jnp.ndarray, rewards: jnp.ndarray, next_states:jnp.ndarray,
                 sum_rewards:jnp.ndarray, prev_states:jnp.ndarray, prev_actions:jnp.ndarray, seq:jnp.ndarray) \
        -> Tuple[Any, Model, Model, Model, Model, Model, InfoDict]:

    rng, task_key, decoder_key, key1, key2  = jax.random.split(rng, 5)

    actions = jnp.reshape(actions, (-1, actions.shape[-1]))
    states = jnp.reshape(states, (-1, states.shape[-1]))
    next_states = jnp.reshape(next_states, (-1, next_states.shape[-1]))
    rewards = jnp.reshape(rewards, (-1, rewards.shape[-1]))

    new_task_encoder, new_policy_encoder, task_info = policy_task_encoder_update(task_encoder, policy_encoder, reward_decoder, transition_decoder, behavior_decoder,
                                                       trajectories, states, actions, next_states, rewards, sum_rewards, prev_states, prev_actions, seq)

    new_reward_decoder, new_transition_decoder, new_behavior_decoder, decoder_info = \
        decoder_update(task_info['task_latent'], task_info['policy_latent'], reward_decoder, transition_decoder, behavior_decoder, states, actions,
                       next_states, rewards, prev_states, prev_actions,  seq)

    return rng, new_task_encoder, new_policy_encoder, new_reward_decoder, new_transition_decoder, new_behavior_decoder, {**task_info, **decoder_info}


class PGTETrainer(object):
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
        print(action_size, state_size)

        states = jnp.zeros((state_size, ))
        actions = jnp.zeros((action_size, ))
        trajectories = jnp.zeros((size, ))
        latents = jnp.zeros((args.latent_dim, ))
        prev_states = jnp.zeros((state_size * (args.n_steps * 2)))
        prev_actions = jnp.zeros((action_size * (args.n_steps * 2)))
        seq = jnp.zeros((args.n_steps * 2, ))

        # Define network
        task_encoder_def = TaskEncoderAE(net_arch=[256, 256],  latent_dim=args.latent_dim)
        policy_encoder_def = PolicyEncoderAE(net_arch = [256, 256], latent_dim=args.latent_dim)
        reward_decoder_def = RewardDecoder(net_arch=[256, 256, 1])
        transition_decoder_def = TransitionDecoder(net_arch=[256, 256, state_size])
        behavior_decoder_def = BehaviorDecoder(net_arch=[256, 256, action_size])

        # create model
        self.key, task_encoder_key, policy_encoder_key, reward_decoder_key, transition_decoder_key, behavior_decoder_key, task_to_policy_key = jax.random.split(self.key, 7)
        self.task_encoder = Model.create(task_encoder_def, inputs=[task_encoder_key, trajectories],
                                         tx=optax.adam(learning_rate=args.lr))
        self.policy_encoder = Model.create(policy_encoder_def, inputs=[policy_encoder_key, prev_states, prev_actions],
                                         tx=optax.adam(learning_rate=args.lr))
        self.reward_decoder = Model.create(reward_decoder_def, inputs=[reward_decoder_key, states, actions, latents],
                                           tx=optax.adam(learning_rate=args.lr))
        self.transition_decoder = Model.create(transition_decoder_def, inputs=[transition_decoder_key, states, actions, latents],
                                               tx=optax.adam(learning_rate=args.lr))
        self.behavior_decoder = Model.create(behavior_decoder_def, inputs=[behavior_decoder_key, states, latents, seq],
                                             tx=optax.adam(learning_rate=args.lr))


    def train(self, epoch):
        for i in range(epoch):
            tbar = tqdm(self.train_loader)
            transition_loss, reward_loss, behavior_loss, kld_loss, kld_loss_2, policy_embedding_loss = 0, 0, 0, 0, 0, 0
            for idx, sample in enumerate(tbar):
                traj, states, actions, rewards, next_states, sum_rewards, task_id, prev_states, prev_actions = sample
                self.key, key = jax.random.split(self.key, 2)
                sum_rewards = np.clip(sum_rewards.numpy(), a_min=0, a_max=1e+10)

                seq = np.zeros((8, 8))
                for j in range(8):
                    seq[j, j] = 1
                seq = np.tile(seq, (sum_rewards.shape[0], 1))

                self.key, new_task_encoder, new_policy_encoder, new_reward_decoder, new_transition_decoder, new_behavior_encoder, info = \
                    _update_pgte(key, self.task_encoder, self.policy_encoder, self.reward_decoder, self.transition_decoder,
                                 self.behavior_decoder, traj.numpy(), states.numpy(), actions.numpy(), rewards.numpy(),
                                 next_states.numpy(), sum_rewards, prev_states.numpy(), prev_actions.numpy(), seq)

                self.task_encoder = new_task_encoder
                self.policy_encoder = new_policy_encoder
                self.reward_decoder = new_reward_decoder
                self.transition_decoder = new_transition_decoder
                self.behavior_decoder = new_behavior_encoder

                transition_loss += info['transition_loss']
                reward_loss += info['reward_loss']
                kld_loss += info['policy_reg_loss']
                kld_loss_2 += info['task_reg_loss']
                policy_embedding_loss += info['policy_embedding_loss']
                behavior_loss += info['behavior_loss']
                tbar.set_description('Epochs %d: Trans. loss: %.4f Rew. loss: %.4f REG loss: %.4f TREG loss: %.4f Beh. loss: %.4f Pol.loss %.4f'
                                     %(i, transition_loss / (idx + 1), reward_loss / (idx + 1), kld_loss / (idx + 1), kld_loss_2 / (idx + 1),
                                       behavior_loss / (idx + 1),  policy_embedding_loss / (idx + 1)))

    def save(self, path, seed):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.task_encoder.save(os.path.join(path, 'policy_task_encoder_{}_seed_{}.jax'.format(self.mode, seed)))
        self.policy_encoder.save(os.path.join(path, 'policy_encoder_{}_seed_{}.jax'.format(self.mode, seed)))
        self.reward_decoder.save(os.path.join(path, 'reward_decoder_{}_seed_{}.jax'.format(self.mode, seed)))
        self.transition_decoder.save(os.path.join(path, 'transition_decoder_{}_seed_{}.jax'.format(self.mode, seed)))
        self.behavior_decoder.save(os.path.join(path, 'behavior_decoder_{}_seed_{}.jax'.format(self.mode, seed)))
        return os.path.join(path, 'policy_task_encoder_{}_seed_{}.jax'.format(self.mode, seed)), \
               os.path.join(path, 'reward_decoder_{}_seed_{}.jax'.format(self.mode, seed)),\
               os.path.join(path, 'transition_decoder_{}_seed_{}.jax'.format(self.mode, seed)),\
               os.path.join(path, 'behavior_decoder_{}_seed_{}.jax'.format(self.mode, seed))