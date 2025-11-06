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


def single_rollout(env, model, key, initial_state, episode_length=1000, deterministic=True):
    """Generate a single rollout with early termination using while_loop."""
    if initial_state is None:
        key, reset_key = jax.random.split(key)
        initial_state = env.reset(reset_key)
    
    # Pre-allocate trajectory arrays
    obs_shape = initial_state.obs.shape
    action_shape = (env.action_size,)
    
    empty_trajectory = {
        'obs': jnp.empty((episode_length,) + obs_shape),
        'action': jnp.empty((episode_length,) + action_shape),
        'reward': jnp.empty(episode_length),
        'done': jnp.empty(episode_length, dtype=bool),
        'truncated': jnp.empty(episode_length, dtype=bool),
        'next_obs': jnp.empty((episode_length,) + obs_shape),
        'step_idx': jnp.arange(episode_length),
    }
    
    def body_fn(_, carry):
        step_count, state, trajectory, key = carry
        obs = state.obs
        
        if deterministic:
            action = model.get_deterministic_action(obs)
        else:
            key, action_key = jax.random.split(key)
            action, _, _, _ = model.sample_action(obs, action_key)

        action = safe_action_clipping(action, getattr(env.sys, 'actuator_ctrlrange', None))
        
        next_state = env.step(state, action)

        done_flag = next_state.info['episode_done'].astype(bool)
        truncated_flag = next_state.info['truncation'].astype(bool)

        # Update trajectory at current step
        updated_trajectory = {
            'obs': trajectory['obs'].at[step_count].set(obs),
            'action': trajectory['action'].at[step_count].set(action),
            'reward': trajectory['reward'].at[step_count].set(next_state.reward),
            'done': trajectory['done'].at[step_count].set(done_flag),
            'truncated': trajectory['truncated'].at[step_count].set(truncated_flag),
            'next_obs': trajectory['next_obs'].at[step_count].set(next_state.obs),
            'step_idx': trajectory['step_idx'],
        }

        # If done, reset for next step
        key, reset_key = jax.random.split(key)
        next_state = jax.lax.cond(
            done_flag,
            lambda _: env.reset(reset_key),
            lambda _: next_state,
            operand=None,
        )
        
        return step_count + 1, next_state, updated_trajectory, key
    
    initial_carry = (0, initial_state, empty_trajectory, key)
    final_carry = jax.lax.fori_loop(0, episode_length, body_fn, initial_carry)

    last_state = final_carry[1]
    trajectory = final_carry[2]

    return last_state, trajectory


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
        initial_state=None,
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
    _, trajectories = vectorized_rollout_fn(rollout_keys)

    return trajectories, key


def vectorized_rollouts_multi_env(
        env,
        model, 
        key,
        num_rollouts,
        episode_length=1000, 
        deterministic=True,
        initial_states=None,
    ):
    """Generate multiple rollouts in parallel using JAX vectorization.
    
    Args:
        env: Single Brax environment (shared across all rollouts)
        model: Policy model
        key: Random key
        num_rollouts: Number of parallel rollouts
        episode_length: Length of each episode
        deterministic: Whether to use deterministic actions
        initial_states: Array of initial states (shape: [num_rollouts, ...])
    
    Returns:
        last_states: Final states for each rollout
        trajectories: Stacked trajectories from all rollouts
        key: Updated random key
    """
    # Generate rollout keys
    keys = jax.random.split(key, num_rollouts + 1)
    key = keys[0]
    rollout_keys = keys[1:]

    # Wrap single_rollout with the fixed environment and model
    rollout_fn = functools.partial(
        single_rollout,
        env,
        model,
        episode_length=episode_length,
        deterministic=deterministic,
    )
    
    # Vmap over keys and initial_states (both have leading dimension num_rollouts)
    vectorized_rollout_fn = jax.jit(jax.vmap(rollout_fn))

    # Call the vectorized function
    last_states, trajectories = vectorized_rollout_fn(rollout_keys, initial_states)

    return last_states, trajectories, key


def compute_returns(trajectories, gamma=1.0, init_returns=None, use_gae=False, 
                   gae_lambda=0.95, values=None, next_values=None):
    """Compute episode running returns or GAE advantages.
    
    Args:
        trajectories: Dictionary containing 'reward' and optionally 'done'
        gamma: Discount factor
        init_returns: Bootstrap values for final states (ignored if use_gae=True)
        use_gae: If True, compute GAE advantages instead of returns
        gae_lambda: Lambda parameter for GAE (used only if use_gae=True)
        values: State values V(s_t) for each timestep (required if use_gae=True)
        next_values: State values V(s_{t+1}) for each timestep (required if use_gae=True)
    
    Returns:
        final_returns: Returns/advantages at t=0 for each episode
        running_returns: Returns/advantages for all timesteps
    
    When a done signal is encountered, the return accumulation resets to 0, handling
    multiple episodes within a single trajectory.
    """
    rewards = trajectories['reward']
    dones = trajectories.get('done', jnp.zeros_like(rewards, dtype=bool))
    truncations = trajectories.get('truncated', jnp.zeros_like(rewards, dtype=bool))
    
    if use_gae:
        # Compute GAE advantages
        if values is None or next_values is None:
            raise ValueError("values and next_values must be provided when use_gae=True")
        
        # Compute TD errors: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        dones_not_truncated = dones * (1.0 - truncations)
        td_errors = rewards + gamma * next_values * (1.0 - dones_not_truncated) - values  # Dones_not_truncated since bootstrapping occurs on truncation

        # Initialize GAE accumulator (starting from the end of the trajectory)
        init_gae = jnp.zeros(rewards.shape[0])  # Shape: (num_rollouts,)
        
        def gae_step(gae, td_and_done):
            td, done = td_and_done
            # GAE_t = δ_t + γ * λ * (1 - done) * GAE_{t+1}
            updated_gae = td + gamma * gae_lambda * (1.0 - done) * gae
            return updated_gae, updated_gae
        
        final_advantages, running_advantages = jax.lax.scan(
            gae_step,
            init=init_gae,
            xs=(jnp.transpose(td_errors, (1, 0)), jnp.transpose(dones, (1, 0))),  # Done instead of dones_not_truncated to reset on episode end
            reverse=True,
        )
        
        return final_advantages, jnp.transpose(running_advantages, (1, 0))
    
    else:
        # Compute regular discounted returns
        if init_returns is None:
            init_returns = jnp.zeros(rewards.shape[0])  # Shape: (num_rollouts,)
        
        # If the final step has done=True, don't bootstrap with init_returns
        init_returns = jnp.where(dones[:, -1], 0.0, init_returns)
        
        def discounted_sum(G, r_and_done):
            r, done = r_and_done
            updated_G = G * gamma * (1.0 - done) + r
            return updated_G, updated_G
        
        final_returns, running_returns = jax.lax.scan(
            discounted_sum, 
            init=init_returns,
            xs=(jnp.transpose(rewards, (1, 0)), jnp.transpose(dones, (1, 0))),
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