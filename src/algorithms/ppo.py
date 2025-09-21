# rl_lib/algorithms/ppo.py
import jax
import jax.numpy as jnp
import optax
from .base import OnPolicyAlgorithm
from ..networks import ActorCriticNetwork


class PPO(OnPolicyAlgorithm):
    def __init__(
            self, 
            config_param: dict,
        ):
        # Initialize network
        rngs = nnx.Rngs(config_param['seed'])
        self.network = ActorCriticNetwork(
            config_param['action_dim'], config_param['hidden_dim'], rngs=rngs)

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
    
    def collect_rollouts(self, env_state, key):
        # Implement rollout collection logic
        pass