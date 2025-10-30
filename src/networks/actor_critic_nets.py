import jax.numpy as jnp
from flax import nnx


from .policy_nets import ActorNetwork
from .value_nets import StateValueNetwork


class ActorCriticNetwork(ActorNetwork):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            limits: jnp.ndarray = None,
            rngs: nnx.Rngs = None
        ):
        super().__init__(obs_dim, action_dim, hidden_dim, limits, rngs)
        self.action_embedding_layer = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(2*hidden_dim, 1, rngs=rngs)

    def __call__(self, x):
        # Forward pass through shared layers
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))

        # Get action distribution parameters
        mean_action = self.policy_head_mean(x)
        logstd_action = self.policy_head_logstd(x)
        
        # Calculate value
        action_embedding = nnx.relu(self.action_embedding_layer(mean_action))
        action_state_representation = jnp.concatenate([x, action_embedding], axis=-1)
        value = jnp.squeeze(self.value_head(action_state_representation), axis=-1)

        return mean_action, logstd_action, value
    

class SeparateActorStateCriticNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            limits: jnp.ndarray = None,
            rngs: nnx.Rngs = None
        ):
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            limits=limits,
            rngs=rngs,
        )
        self.critic = StateValueNetwork(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

    def __call__(self, x):
        mean_action, logstd_action = self.actor(x)
        value = jnp.squeeze(self.critic(x), axis=-1)
        return mean_action, logstd_action, value
    
    def sample_action(self, x, key):
        return self.actor.sample_action(x, key)
    
    def get_deterministic_action(self, x):
        return self.actor.get_deterministic_action(x)