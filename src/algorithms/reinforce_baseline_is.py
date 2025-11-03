# rl_lib/algorithms/reinforce_baseline_is.py
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from .on_policy import OnPolicyAlgorithm
from ..networks import SeparateActorStateCriticNetwork


class REINFORCEwithBaselineIS(OnPolicyAlgorithm):
    def _init_network(self):
        """Initialize the policy network for REINFORCE with baseline and importance sampling."""

        # Initialize network
        rngs = nnx.Rngs(self.config.seed)
        limits = getattr(self.env.sys, 'actuator_ctrlrange', None)
        self.network = SeparateActorStateCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            limits=limits,
            rngs=rngs,
        )

        # Initialize optimizer
        self.optimizer = nnx.Optimizer(
            self.network, optax.adam(self.config.lr), wrt=nnx.Param)

    def get_log_prob(self, model, obs, action):
        mean_action, logstd_action = model.actor(obs)
        log_probs_gaussian = -0.5 * (
            ((action - mean_action) / (jnp.exp(logstd_action) + 1e-8)) ** 2 
            + 2 * logstd_action 
            + jnp.log(2 * jnp.pi)
        )
        log_probs = log_probs_gaussian - jnp.log((1 - jnp.tanh(action) ** 2).clip(1e-8))
        log_prob = log_probs.sum(axis=-1)
        return log_prob
    
    def loss(self, model, obs, actions, returns, old_log_probs):
        action_log_probs = self.get_log_prob(model, obs, actions).clip(-10.0, 2.0)
        likelihood_ratios = jnp.exp((action_log_probs - old_log_probs).clip(-10.0, 2.0))
        predicted_values = model.critic(obs)
        differences = returns - predicted_values
        
        policy_loss = -jnp.mean(likelihood_ratios * jax.lax.stop_gradient(differences))
        value_loss = jnp.mean(differences ** 2)
        return policy_loss + value_loss

    @nnx.jit
    def update(self, obs, actions, returns, info={}):
        # Get initial action (log-)likelihoods
        if not 'old_log_probs' in info:
            info['old_log_probs'] = self.get_log_prob(self.network, obs, actions).clip(-10.0, 2.0)
        old_log_probs = info['old_log_probs']

        loss_fn = lambda model: self.loss(model, obs, actions, returns, old_log_probs)
        loss, grads = nnx.value_and_grad(loss_fn)(self.network)
        self.optimizer.update(self.network, grads)
        return loss, grads, info