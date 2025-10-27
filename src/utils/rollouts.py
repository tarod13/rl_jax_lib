import jax
import jax.numpy as jnp
from brax import envs
import functools
from flax import nnx
import numpy as np
import pickle
import os
from datetime import datetime

from ..networks import ActorCriticNetwork


def safe_action_clipping(action, limits):
    """Safely clip actions to prevent NaN propagation."""
    action = jnp.where(jnp.isnan(action), 0.0, action)
    
    if limits is not None:
        action = jnp.clip(action, limits[:, 0], limits[:, 1])
    
    return action


def single_rollout(env, model, key, episode_length=1000, deterministic=True):
    """Generate a single rollout with early termination using while_loop."""
    initial_state = env.reset(key)
    
    # Pre-allocate trajectory arrays
    obs_shape = initial_state.obs.shape
    action_shape = (env.action_size,)
    
    empty_trajectory = {
        'obs': jnp.empty((episode_length,) + obs_shape),
        'action': jnp.empty((episode_length,) + action_shape),
        'reward': jnp.empty(episode_length),
        'done': jnp.empty(episode_length, dtype=bool),
        'next_obs': jnp.empty((episode_length,) + obs_shape),
        'step_idx': jnp.arange(episode_length),
        'valid': jnp.empty(episode_length, dtype=bool),
    }
    
    def cond_fn(carry):
        step_count, state, done_flag, _, _ = carry
        return (step_count < episode_length) & (~done_flag)
    
    def body_fn(carry):
        step_count, state, done_flag, trajectory, key = carry
        obs = state.obs
        
        if deterministic:
            action = model.get_deterministic_action(obs)
        else:
            key, action_key = jax.random.split(key)
            action, _, _, _ = model.sample_action(obs, action_key)

        action = safe_action_clipping(action, getattr(env.sys, 'actuator_ctrlrange', None))
        
        next_state = env.step(state, action)
        new_done_flag = next_state.done.astype(bool)
        
        # Update trajectory at current step
        updated_trajectory = {
            'obs': trajectory['obs'].at[step_count].set(obs),
            'action': trajectory['action'].at[step_count].set(action),
            'reward': trajectory['reward'].at[step_count].set(next_state.reward),
            'done': trajectory['done'].at[step_count].set(new_done_flag),
            'next_obs': trajectory['next_obs'].at[step_count].set(next_state.obs),
            'step_idx': trajectory['step_idx'],
            'valid': trajectory['valid'].at[step_count].set(True),
        }
        
        return step_count + 1, next_state, new_done_flag, updated_trajectory, key
    
    initial_carry = (0, initial_state, jnp.array(False, dtype=bool), empty_trajectory, key)
    final_carry = jax.lax.while_loop(cond_fn, body_fn, initial_carry)

    return final_carry[3]  # Return final trajectory


def vectorized_rollouts(
        env, 
        model, 
        key,
        num_rollouts=100, 
        episode_length=1000, 
        deterministic=True, 
    ):
    """Generate multiple rollouts in parallel using JAX vectorization."""
    # Wrap the single rollout function with fixed parameters
    # The key will be the only varying argument to the wrapped function
    rollout_fn = functools.partial(
        single_rollout,
        env,
        model,
        episode_length=episode_length,
        deterministic=deterministic,
    )
    
    # Vectorize and JIT the wrapped function
    vectorized_rollout_fn = jax.jit(jax.vmap(rollout_fn))

    # Generate rollout keys
    keys = jax.random.split(key, num_rollouts + 1)
    key = keys[0]
    rollout_keys = keys[1:]

    # Call the vectorized rollout function with the generated keys
    trajectories = vectorized_rollout_fn(rollout_keys)

    return trajectories, key


def compute_returns(trajectories, gamma=1.0):
    """Compute episode running returns with discount factor and proper validity masking."""
    rewards = trajectories['reward']
    
    if 'valid' in trajectories:
        valid_mask = trajectories['valid']
        masked_rewards = rewards * valid_mask
    else:
        masked_rewards = rewards
    
    def discounted_sum(G, r):
        updated_G = G * gamma + r
        return updated_G, updated_G

    init_returns = jnp.zeros(rewards.shape[0])  # Shape: (num_rollouts,)
    final_returns, running_returns = jax.lax.scan(
        discounted_sum, 
        init = (init_returns),
        xs = jnp.transpose(masked_rewards, (1, 0)),
        reverse=True,
    )  # Transpose so that the scan is along episode steps and not rollouts

    return final_returns, jnp.transpose(running_returns, (1, 0))


def rollout_statistics(trajectories, gamma=1.0):
    """Compute comprehensive statistics from rollout data."""
    returns, _ = compute_returns(trajectories, gamma)
    rewards = trajectories['reward']

    returns_np = np.array(returns)
    rewards_np = np.array(rewards)
    
    episode_lengths = None
    if 'valid' in trajectories:
        valid_mask = np.array(trajectories['valid'])
        episode_lengths = np.sum(valid_mask, axis=1)
    
    stats = {
        'num_rollouts': len(returns_np),
        'episode_length': rewards_np.shape[-1],
        'mean_return': float(np.mean(returns_np)),
        'std_return': float(np.std(returns_np)),
        'min_return': float(np.min(returns_np)),
        'max_return': float(np.max(returns_np)),
        'median_return': float(np.median(returns_np)),
        'mean_reward_per_step': float(np.mean(rewards_np)),
        'std_reward_per_step': float(np.std(rewards_np)),
    }
    
    if episode_lengths is not None:
        stats.update({
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'min_episode_length': float(np.min(episode_lengths)),
            'max_episode_length': float(np.max(episode_lengths)),
            'episodes_terminated_early': int(np.sum(episode_lengths < rewards_np.shape[-1]))
        })
    
    return stats


def save_trajectories(trajectories, save_dir="rollout_data", filename=None):
    """Save trajectory data with metadata."""
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectories_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    save_data = {
        'trajectories': trajectories,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_rollouts': trajectories['obs'].shape[0],
            'episode_length': trajectories['obs'].shape[1],
            'obs_dim': trajectories['obs'].shape[-1],
            'action_dim': trajectories['action'].shape[-1],
        }
    }
    
    save_data = jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, save_data)
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filepath


def example_usage():
    """Example usage for trajectory generation."""
    env_name = 'ant'
    env = envs.get_environment(env_name)
    action_dim = env.action_size
    obs_dim = env.observation_size
    limits = getattr(env.sys, 'actuator_ctrlrange', None)

    model = ActorCriticNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        limits=limits,
        rngs=nnx.Rngs(42)
    )
    
    key = jax.random.PRNGKey(0)
    trajectories, key = vectorized_rollouts(
        env=env,
        model=model,
        key=key,
        num_rollouts=100,
        episode_length=1000,
        deterministic=True
    )
    
    stats = rollout_statistics(trajectories)
    save_path = save_trajectories(trajectories)
    
    return trajectories, save_path


if __name__ == "__main__":
    trajectories, save_path = example_usage()