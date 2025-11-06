import jax
import jax.numpy as jnp
from flax import nnx


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

def polyak_update(target_model: nnx.Module, online_model: nnx.Module, tau: float):
    """
    Polyak averaging update of target network.
    
    Args:
        target_model: Target network to update
        online_model: Online network to copy from
        tau: Polyak coefficient (target = tau * online + (1-tau) * target)
    """
    # Get parameter state and rest (including Variables like eigenvectors)
    target_graph, target_state, target_rest = nnx.split(target_model, nnx.Param, ...)
    _, online_state, _ = nnx.split(online_model, nnx.Param, ...)
    
    # Perform Polyak averaging on the trainable parameters only
    updated_state = jax.tree.map(
        lambda t, o: tau * o + (1 - tau) * t,
        target_state,
        online_state
    )
    
    # Merge back into target model (parameters updated, rest unchanged)
    nnx.update(target_model, updated_state, target_rest)