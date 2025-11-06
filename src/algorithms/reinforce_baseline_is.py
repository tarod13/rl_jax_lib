# rl_lib/algorithms/reinforce_baseline_is.py
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from .on_policy import OnPolicyAlgorithm
from ..networks import SSActorStateCriticNetwork


class REINFORCEwithBaselineIS(OnPolicyAlgorithm):
    def _init_network(self):
        """Initialize the policy network for REINFORCE with baseline and importance sampling."""

        # Initialize network
        rngs = nnx.Rngs(self.config.seed)
        limits = getattr(self.env.sys, 'actuator_ctrlrange', None)
        self.network = SSActorStateCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            limits=limits,
            rngs=rngs,
            nl=self.config.nl,
            use_layernorm=self.config.use_layernorm,
        )

        # Initialize optimizer
        self.optimizer = nnx.Optimizer(
            self.network, optax.adam(self.config.lr), wrt=nnx.Param)

    def get_log_prob(self, model, obs, action):
        pre_tanh_action = jnp.arctanh(jnp.clip(action, -0.999, 0.999))  # Invert tanh
        mean_action, logstd_action = model.actor(obs)
        log_probs_gaussian = -0.5 * (
            ((pre_tanh_action - mean_action) / (jnp.exp(logstd_action) + 1e-8)) ** 2
            + 2 * logstd_action
            + jnp.log(2 * jnp.pi)
        )
        log_probs = log_probs_gaussian - jnp.log((1 - action ** 2).clip(1e-8))
        log_prob = log_probs.sum(axis=-1)
        return log_prob
    
    def loss(self, model, obs, actions, returns, old_log_probs):
        action_log_probs = self.get_log_prob(model, obs, actions)
        likelihood_ratios = jnp.exp((action_log_probs - old_log_probs).clip(-10.0, 2.0))
        predicted_values = model.critic(obs)
        differences = returns - predicted_values
        
        policy_loss = -jnp.mean(likelihood_ratios * jax.lax.stop_gradient(differences))
        value_loss = jnp.mean(differences ** 2)
        return policy_loss + value_loss

    @nnx.jit
    def update(self, obs, actions, returns, advantages, old_log_probs):
        # Get initial action (log-)likelihoods
        loss_fn = lambda model: self.loss(model, obs, actions, returns, old_log_probs)
        loss, grads = nnx.value_and_grad(loss_fn)(self.network)
        self.optimizer.update(self.network, grads)
        return loss, grads