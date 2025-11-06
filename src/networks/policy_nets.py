import jax
import jax.numpy as jnp
from flax import nnx


class StochasticActorNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            limits: jnp.ndarray = None, 
            rngs: nnx.Rngs = None,
            nl: str = 'relu',
            use_layernorm: bool = False,
        ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.limits = nnx.Variable(jnp.abs(limits).max(axis=1)) if limits is not None else None
        self.rngs = rngs
        self.nl = nl
        self.use_layernorm = use_layernorm

        self.dense1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.policy_head_mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.policy_head_logstd = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        
        # Optional LayerNorm layers
        if self.use_layernorm:
            self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x):
        # Forward pass through shared layers
        if self.nl == 'relu':
            x = self.dense1(x)
            if self.use_layernorm:
                x = self.ln1(x)
            x = nnx.relu(x)
            
            x = self.dense2(x)
            if self.use_layernorm:
                x = self.ln2(x)
            x = nnx.relu(x)
        elif self.nl == 'tanh':
            x = self.dense1(x)
            if self.use_layernorm:
                x = self.ln1(x)
            x = nnx.tanh(x)
            
            x = self.dense2(x)
            if self.use_layernorm:
                x = self.ln2(x)
            x = nnx.tanh(x)
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nl}")

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
        mean_action, _ = self(x)
        action = nnx.tanh(mean_action)
        action = self.scale_action(action)
        return action
    

class ActorNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            limits: jnp.ndarray = None, 
            rngs: nnx.Rngs = None,
            nl: str = 'relu',
            use_layernorm: bool = False,
            sigma: float = 0.1,
        ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.limits = nnx.Variable(jnp.abs(limits).max(axis=1)) if limits is not None else None
        self.rngs = rngs
        self.nl = nl
        self.use_layernorm = use_layernorm
        self.sigma = sigma

        self.dense1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.action_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        
        # Optional LayerNorm layers
        if self.use_layernorm:
            self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x):
        # Forward pass through shared layers
        if self.nl == 'relu':
            x = self.dense1(x)
            if self.use_layernorm:
                x = self.ln1(x)
            x = nnx.relu(x)
            
            x = self.dense2(x)
            if self.use_layernorm:
                x = self.ln2(x)
            x = nnx.relu(x)
        elif self.nl == 'tanh':
            x = self.dense1(x)
            if self.use_layernorm:
                x = self.ln1(x)
            x = nnx.tanh(x)
            
            x = self.dense2(x)
            if self.use_layernorm:
                x = self.ln2(x)
            x = nnx.tanh(x)
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nl}")

        # Get action
        action = self.action_head(x)
        return action
    
    def scale_action(self, action):
        if self.limits is not None:
            action = action * self.limits
        return action
    
    def sample_action(self, x, key):
        # Sample normal action
        action = self(x)
        epsilon = jax.random.normal(key, action.shape)

        # Apply reparameterization trick and squashing
        action = nnx.tanh(action + epsilon)

        # Scale action to environment limits
        action = self.scale_action(action)

        return action, epsilon

    def get_deterministic_action(self, x):
        return self.scale_action(nnx.tanh(self(x)))