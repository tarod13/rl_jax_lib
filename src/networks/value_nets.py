from flax import nnx
import jax.numpy as jnp


class StateValueNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            hidden_dim: int = 256,
            rngs: nnx.Rngs = None,
            nl: str = 'relu',
            use_layernorm: bool = False,
        ):
        self.hidden_dim = hidden_dim
        self.rngs = rngs
        self.nl = nl
        self.use_layernorm = use_layernorm

        self.dense1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        
        # Optional LayerNorm layers
        if self.use_layernorm:
            self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x):
        # Forward pass through hidden layers
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

        # Get value
        value = self.value_head(x)
        return value.squeeze(axis=-1)
    

class ValueNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            rngs: nnx.Rngs = None,
            nl: str = 'relu',
            use_layernorm: bool = False,
        ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rngs = rngs
        self.nl = nl
        self.use_layernorm = use_layernorm

        self.dense1 = nnx.Linear(obs_dim + action_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        
        # Optional LayerNorm layers
        if self.use_layernorm:
            self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.ln2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, s, a):
        # Forward pass through hidden layers
        x = jnp.concatenate([s, a], axis=-1)
        x = self.dense1(x)
        if self.use_layernorm:
            x = self.ln1(x)
        if self.nl == 'relu':
            x = nnx.relu(x)
        elif self.nl == 'tanh':
            x = nnx.tanh(x)
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nl}")

        x = self.dense2(x)
        if self.use_layernorm:
            x = self.ln2(x)
        if self.nl == 'relu':
            x = nnx.relu(x)
        elif self.nl == 'tanh':
            x = nnx.tanh(x)
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nl}")

        # Get value
        value = self.value_head(x)
        return value