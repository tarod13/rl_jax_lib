import jax.numpy as jnp
from flax import nnx


from .policy_nets import ActorNetwork


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