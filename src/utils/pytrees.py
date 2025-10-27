import jax
import jax.numpy as jnp


def tree_norm(pytree):
    """Compute the global L2 norm of a pytree of arrays."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))