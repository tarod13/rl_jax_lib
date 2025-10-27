from flax import nnx


class ValueNetwork(nnx.Module):
    def __init__(
            self,
            obs_dim: int, 
            action_dim: int, 
            hidden_dim: int = 256,
            rngs: nnx.Rngs = None
        ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rngs = rngs

        self.dense1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x):
        # Forward pass through hidden layers
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))

        # Get value
        value = self.value_head(x)
        return value