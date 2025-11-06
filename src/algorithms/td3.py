# rl_lib/algorithms/ppo.py
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from .on_policy import OnPolicyAlgorithm
from ..networks import SeparateActorStateCriticNetwork
from ..utils import clip_grads, polyak_update


class TD3(OnPolicyAlgorithm):
    def _init_network(self):
        """Initialize the policy network for on-policy TD3."""

        # Initialize network
        rngs = nnx.Rngs(self.config.seed)
        limits = getattr(self.env.sys, 'actuator_ctrlrange', None)
        self.online_network = SeparateActorStateCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            limits=limits,
            rngs=rngs,
            nl=self.config.nl,
            use_layernorm=self.config.use_layernorm,
        )
        self.target_network = SeparateActorStateCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            limits=limits,
            rngs=rngs,
            nl=self.config.nl,
            use_layernorm=self.config.use_layernorm,
        )
        self.target_network = polyak_update(self.target_network, self.online_network, tau=1.0)

        # Initialize optimizers
        self.optimizer_actor = nnx.Optimizer(
            self.online_network.actor, optax.adam(self.config.lr), wrt=nnx.Param)
        self.optimizer_critic_1 = nnx.Optimizer(
            self.online_network.critic_1, optax.adam(self.config.lr), wrt=nnx.Param)
        self.optimizer_critic_2 = nnx.Optimizer(
            self.online_network.critic_2, optax.adam(self.config.lr), wrt=nnx.Param)

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

    def critic_loss(self, model, obs, actions, targets):
        """Compute the critic loss for TD3."""
        # Compute target Q-values
        predicted_values = model(obs, actions)
        value_errors = targets - predicted_values
        loss = jnp.mean(value_errors ** 2)
        return loss
    
    def compute_targets(self, rewards, next_obs, dones, keys):
        next_actions = self.target_network.sample_action(next_obs, keys)
        next_values_1, next_values_2 = self.target_network.get_values(next_obs, next_actions)
        next_values = jnp.minimum(next_values_1, next_values_2)
        targets = rewards + self.config.gamma * (1.0 - dones) * next_values
        return targets

    @nnx.jit
    def update_critics(self, obs, actions, rewards, next_obs, dones, keys):
        """
        Perform a single TD3 update.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions [batch_size, action_dim]
            rewards: Rewards [batch_size]
            next_obs: Next observations [batch_size, obs_dim]
            dones: Done flags [batch_size]
            keys: RNG keys for action sampling
            
        Returns:
            loss: Scalar loss value
            grads: Gradients
        """
        targets = self.compute_targets(
            rewards,
            obs['next_obs'],
            obs['dones'],
            keys
        )

        critic_loss_fn = lambda model: self.critic_loss(model, obs, actions, targets)

        losses, grads = [], []
        for model, optimizer in [
            (self.online_network.critic_1, self.optimizer_critic_1),
            (self.online_network.critic_2, self.optimizer_critic_2),
        ]:
            loss, grad = nnx.value_and_grad(critic_loss_fn)(model)
            losses.append(loss)
            grads.append(grad)
            if self.config.max_grad_norm is not None:
                grads = clip_grads(grads, max_norm=self.config.max_grad_norm)
            optimizer.update(model, grads)
        return losses, grads