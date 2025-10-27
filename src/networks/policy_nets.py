import jax
import jax.numpy as jnp
from flax import nnx


class ActorNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            limits: jnp.ndarray = None, 
            rngs: nnx.Rngs = None
        ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.limits = nnx.Variable(jnp.abs(limits).max(axis=1)) if limits is not None else None
        self.rngs = rngs

        self.dense1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.policy_head_mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.policy_head_logstd = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, x):
        # Forward pass through shared layers
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))

        # Get action distribution parameters
        mean_action = self.policy_head_mean(x)
        logstd_action = self.policy_head_logstd(x)

        return mean_action, logstd_action
    
    def scale_action(self, action):
        if self.limits is not None:
            action = action * self.limits
        return action
    
    def sample_action(self, x, key):
        # Sample normal action
        mean_action, logstd_action = self(x)
        epsilon = jax.random.normal(key, mean_action.shape)
        
        # Apply reparameterization trick and squashing
        action = nnx.tanh(mean_action + jnp.exp(logstd_action) * epsilon)

        # Scale action to environment limits
        action = self.scale_action(action)

        return action, epsilon, mean_action, logstd_action

    def get_deterministic_action(self, x):
        mean_action, _, _ = self(x)
        action = nnx.tanh(mean_action)
        action = self.scale_action(action)
        return action