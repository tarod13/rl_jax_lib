"""
Evaluation utilities for clean performance metrics.

This module provides evaluation functionality that:
1. Runs complete episodes until natural termination (done=True)
2. Computes undiscounted total returns
3. Uses deterministic policy for reproducibility
4. Provides a clean separation from training rollouts
5. Uses while_loop for efficiency (stops immediately when done)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any


def evaluate_agent(
    agent,
    key,
    num_eval_episodes: int = 10,
    max_episode_length: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluate an agent by running complete episodes until termination.
    
    Args:
        agent: RL agent to evaluate
        key: Random key for environment resets
        num_eval_episodes: Number of episodes to run for evaluation
        max_episode_length: Maximum steps per episode (safety limit)
        
    Returns:
        Dictionary containing:
            - 'returns': Array of undiscounted episode returns
            - 'episode_lengths': Array of episode lengths
            - 'mean_return': Mean of episode returns
            - 'std_return': Standard deviation of returns
            - 'min_return': Minimum return
            - 'max_return': Maximum return
            - 'mean_length': Mean episode length
    """
    # Generate fresh initial states for evaluation
    reset_keys = jax.random.split(key, num_eval_episodes)
    initial_states = [agent.env.reset(k) for k in reset_keys]
    initial_states_stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *initial_states
    )
    
    # Run rollouts until episodes complete (deterministic policy)
    episode_lengths, last_states, trajectories, _ = run_evaluation_rollouts(
        env=agent.env,
        model=agent.network,
        key=key,
        num_episodes=num_eval_episodes,
        max_episode_length=max_episode_length,
        initial_states=initial_states_stacked,
    )
    
    # Compute undiscounted returns for completed episodes
    returns, _ = compute_evaluation_metrics(trajectories)
    
    # Compute statistics
    eval_stats = {
        'returns': returns,
        'episode_lengths': episode_lengths,
        'mean_return': float(jnp.mean(returns)),
        'std_return': float(jnp.std(returns)),
        'min_return': float(jnp.min(returns)),
        'max_return': float(jnp.max(returns)),
        'mean_length': float(jnp.mean(episode_lengths)),
        'std_length': float(jnp.std(episode_lengths)),
    }
    
    return eval_stats


def run_evaluation_rollouts(
    env,
    model,
    key,
    num_episodes: int,
    max_episode_length: int,
    initial_states,
):
    """
    Run evaluation rollouts using deterministic policy until episode completion.
    
    This is similar to vectorized_rollouts_multi_env but specifically for evaluation:
    - Always uses deterministic actions
    - Tracks episode completion via done signals
    - Returns valid trajectory lengths
    """
    from functools import partial
    
    # Generate keys for each episode
    keys = jax.random.split(key, num_episodes + 1)
    key = keys[0]
    episode_keys = keys[1:]
    
    # Create the single episode evaluation function
    eval_episode_fn = partial(
        _single_evaluation_episode,
        env,
        model,
        max_episode_length=max_episode_length,
    )
    
    # Vectorize across episodes
    vectorized_eval_fn = jax.jit(jax.vmap(eval_episode_fn))
    
    # Run all episodes in parallel
    episode_lengths, last_states, trajectories = vectorized_eval_fn(episode_keys, initial_states)
    
    return episode_lengths, last_states, trajectories, key


def _single_evaluation_episode(env, model, key, initial_state, max_episode_length):
    """
    Run a single evaluation episode until done or max length.
    
    Uses deterministic actions and stops immediately when episode terminates.
    """
    # Pre-allocate trajectory arrays
    obs_shape = initial_state.obs.shape
    action_shape = (env.action_size,)
    
    empty_trajectory = {
        'obs': jnp.empty((max_episode_length,) + obs_shape),
        'action': jnp.empty((max_episode_length,) + action_shape),
        'reward': jnp.empty(max_episode_length),
        'done': jnp.empty(max_episode_length, dtype=bool),
        'next_obs': jnp.empty((max_episode_length,) + obs_shape),
        'valid': jnp.zeros(max_episode_length, dtype=bool),
    }
    
    def cond_fn(carry):
        """Continue while episode is not done and under max length."""
        step_count, state, trajectory, episode_done = carry
        return (~episode_done) & (step_count < max_episode_length)
    
    def body_fn(carry):
        """Take one step in the environment."""
        step_count, state, trajectory, episode_done = carry
        obs = state.obs
        
        # Get deterministic action
        action = model.get_deterministic_action(obs)
        
        # Clip actions safely
        limits = getattr(env.sys, 'actuator_ctrlrange', None)
        action = _safe_clip(action, limits)
        
        # Step environment
        next_state = env.step(state, action)
        new_done_flag = next_state.done.astype(bool)
        
        # Update trajectory at current step
        updated_trajectory = {
            'obs': trajectory['obs'].at[step_count].set(obs),
            'action': trajectory['action'].at[step_count].set(action),
            'reward': trajectory['reward'].at[step_count].set(next_state.reward),
            'done': trajectory['done'].at[step_count].set(new_done_flag),
            'next_obs': trajectory['next_obs'].at[step_count].set(next_state.obs),
            'valid': trajectory['valid'].at[step_count].set(jnp.asarray(True)),
        }
        
        # Update episode done flag
        episode_done = new_done_flag
        
        return step_count + 1, next_state, updated_trajectory, episode_done
    
    # Initial carry: not done yet
    initial_carry = (0, initial_state, empty_trajectory, jnp.asarray(False))
    final_carry = jax.lax.while_loop(cond_fn, body_fn, initial_carry)
    
    episode_lenght = final_carry[0]
    last_state = final_carry[1]
    trajectory = final_carry[2]
    
    return episode_lenght, last_state, trajectory


def _safe_clip(action, limits):
    """Safely clip actions to prevent NaN propagation."""
    action = jnp.where(jnp.isnan(action), 0.0, action)
    
    if limits is not None:
        action = jnp.clip(action, limits[:, 0], limits[:, 1])
    
    return action


def compute_evaluation_metrics(trajectories):
    """
    Compute undiscounted returns and episode lengths from evaluation trajectories.
    
    Args:
        trajectories: Dictionary with keys 'reward', 'done', 'valid'
        
    Returns:
        returns: Array of shape [num_episodes] with undiscounted total returns
        episode_lengths: Array of shape [num_episodes] with episode lengths
    """
    rewards = trajectories['reward']
    valid_mask = trajectories['valid']
    
    # Mask out invalid steps
    masked_rewards = jnp.where(valid_mask, rewards, 0.0)
    
    # Compute undiscounted returns (just sum of rewards)
    returns = jnp.sum(masked_rewards, axis=1)
    
    # Compute episode lengths
    episode_lengths = jnp.sum(valid_mask, axis=1)
    
    return returns, episode_lengths


def print_evaluation_summary(eval_stats: Dict[str, Any], step: int = None):
    """
    Pretty print evaluation statistics.
    
    Args:
        eval_stats: Dictionary from evaluate_agent
        step: Optional training step number
    """
    header = "EVALUATION SUMMARY"
    if step is not None:
        header += f" (Step {step})"
    
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"Number of episodes: {len(eval_stats['returns'])}")
    print(f"Mean return:        {eval_stats['mean_return']:8.2f} ± {eval_stats['std_return']:6.2f}")
    print(f"Min return:         {eval_stats['min_return']:8.2f}")
    print(f"Max return:         {eval_stats['max_return']:8.2f}")
    print(f"Mean length:        {eval_stats['mean_length']:8.2f} ± {eval_stats['std_length']:6.2f}")
    print("=" * 60 + "\n")