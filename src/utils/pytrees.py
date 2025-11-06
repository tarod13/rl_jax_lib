import jax
import jax.numpy as jnp


def tree_norm(pytree):
    """Compute the global L2 norm of a pytree of arrays."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))

def clip_grads(grads, max_norm):
    """Clip gradients by global norm."""
    # Compute global norm using existing function
    global_norm = tree_norm(grads)
    
    # Clip if necessary
    scale = jnp.minimum(max_norm / (global_norm + 1e-6), 1.0)
    clipped_grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    
    return clipped_grads