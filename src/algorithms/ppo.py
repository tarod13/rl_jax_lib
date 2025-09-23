# rl_lib/algorithms/ppo.py
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from brax import envs

from .base import OnPolicyAlgorithm
from ..networks import ActorCriticNetwork
from ..utils import vectorized_rollouts, rollout_statistics


class PPO(OnPolicyAlgorithm):
    def __init__(
            self, 
            config_param: dict,
        ):
        # Initialize environment
        self.env = envs.get_environment(config_param['env_name'])
        self.action_dim = self.env.action_size
        self.obs_dim = self.env.observation_size
        self.limits = getattr(self.env.sys, 'actuator_ctrlrange', None)

        # Initialize network
        rngs = nnx.Rngs(config_param['seed'])
        self.network = ActorCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=config_param['hidden_dim'],
            limits=self.limits,
            rngs=rngs,
        )

        # Initialize optimizer
        self.optimizer = nnx.Optimizer(
            self.network, optax.adam(config_param['lr']))
        
        # Save config
        self.config = config_param

    @nnx.jit
    def get_log_prob_and_value(self, obs, action, key):
        mean_action, logstd_action, value = self.network(obs)
        log_probs_gaussian = -0.5 * (
            ((action - mean_action) / (jnp.exp(logstd_action) + 1e-8)) ** 2 
            + 2 * logstd_action 
            + jnp.log(2 * jnp.pi)
        )
        log_probs = log_probs_gaussian - jnp.log((1 - jnp.tanh(action) ** 2).clip(1e-8))
        log_prob = log_probs.sum(axis=-1)
        return log_prob, value
    
    @nnx.jit  
    def ppo_loss(self, obs, actions, old_log_probs, advantages, returns):
        new_log_probs, values = self.network(obs)

        # Policy loss
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(
            ratio, 
            min=self.config['min_ratio'], 
            max=self.config['max_ratio'],
        )
        
        policy_loss = -jnp.mean(jnp.minimum(
            ratio * advantages,
            clipped_ratio * advantages
        ))
        
        # Value loss  
        value_loss = jnp.mean((values - returns) ** 2)

        total_loss = policy_loss + self.config['value_loss_coef'] * value_loss
        return total_loss
    
    @nnx.jit
    def update(self, obs, actions, old_log_probs, advantages, returns):
        loss, grads = nnx.value_and_grad(self.ppo_loss)(
            obs, actions, old_log_probs, advantages, returns
        )
        self.optimizer.update(grads)
        return loss
    

    def collect_rollouts(self):
        # Implement rollout collection logic
        trajectories = vectorized_rollouts(
            env=self.env,
            model=self.network,
            num_rollouts=self.config['num_rollouts'],
            episode_length=self.config['episode_length'],
            deterministic=self.config['deterministic'],
        )

        rollout_stats = rollout_statistics(
            trajectories, gamma=1.0,
        )

        return trajectories, rollout_stats

    def train(self, env_state, key):
        # Implement training loop logic
        for training_step in range(self.config['num_training_steps']):
            env_state, key = self.collect_rollouts(env_state, key)
            # Further training logic goes here

        pass