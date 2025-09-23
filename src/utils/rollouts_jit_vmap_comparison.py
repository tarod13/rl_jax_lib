import jax
import jax.numpy as jnp
from brax import envs
import functools
from flax import nnx
import numpy as np
import pickle
import os
import time
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
        'valid': jnp.empty(episode_length, dtype=bool)
    }
    
    def cond_fn(carry):
        step_count, state, done_flag, _ = carry
        return (step_count < episode_length) & (~done_flag)
    
    def body_fn(carry):
        step_count, state, done_flag, trajectory = carry
        obs = state.obs
        
        if deterministic:
            action = model.get_deterministic_action(obs)
        else:
            action, _, _, _ = model.sample_action(obs)
        
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
            'valid': trajectory['valid'].at[step_count].set(True)
        }
        
        return step_count + 1, next_state, new_done_flag, updated_trajectory
    
    initial_carry = (0, initial_state, jnp.array(False, dtype=bool), empty_trajectory)
    final_carry = jax.lax.while_loop(cond_fn, body_fn, initial_carry)

    return final_carry[3]  # Return final trajectory


def vectorized_rollouts_jit_vmap(env, model, num_rollouts=100, episode_length=1000, deterministic=True):
    """Generate multiple rollouts using jit(vmap(func)) strategy."""
    rollout_fn = functools.partial(
        single_rollout,
        env,
        model,
        episode_length=episode_length,
        deterministic=deterministic
    )
    
    # Strategy 1: jit(vmap(func)) - JIT the vectorized function
    vectorized_rollout_fn = jax.jit(jax.vmap(rollout_fn))
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_rollouts)
    
    trajectories = vectorized_rollout_fn(keys)
    
    return trajectories


def vectorized_rollouts_vmap_jit(env, model, num_rollouts=100, episode_length=1000, deterministic=True):
    """Generate multiple rollouts using vmap(jit(func)) strategy."""
    rollout_fn = functools.partial(
        single_rollout,
        env,
        model,
        episode_length=episode_length,
        deterministic=deterministic
    )
    
    # Strategy 2: vmap(jit(func)) - vectorize the JIT'd function
    jitted_rollout_fn = jax.jit(rollout_fn)
    vectorized_rollout_fn = jax.vmap(jitted_rollout_fn)
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_rollouts)
    
    trajectories = vectorized_rollout_fn(keys)
    
    return trajectories


def compare_jit_vmap_strategies(env, model, num_rollouts=100, episode_length=1000, deterministic=True, 
                               include_compilation_time=True, warmup_runs=2):
    """Compare performance of jit(vmap(func)) vs vmap(jit(func)) strategies.
    
    Args:
        env: Brax environment
        model: ActorCriticNetwork model
        num_rollouts: Number of rollouts for comparison
        episode_length: Maximum episode length
        deterministic: Whether to use deterministic actions
        include_compilation_time: If True, measure and show compilation times separately
        warmup_runs: Number of execution runs after compilation for averaging
        
    Returns:
        dict: Comparison results including timing and statistics
    """
    print(f"Comparing JIT/VMAP strategies with {num_rollouts} rollouts...")
    
    strategies = {
        'jit_vmap': {
            'name': 'jit(vmap(func))',
            'function': vectorized_rollouts_jit_vmap,
            'description': 'JIT compile the vectorized function'
        },
        'vmap_jit': {
            'name': 'vmap(jit(func))',
            'function': vectorized_rollouts_vmap_jit,
            'description': 'Vectorize the JIT compiled function'
        }
    }
    
    results = {}
    
    for strategy_key, strategy_info in strategies.items():
        print(f"\nTesting {strategy_info['name']} strategy...")
        print(f"  ({strategy_info['description']})")
        
        strategy_fn = strategy_info['function']
        
        if include_compilation_time:
            # Measure compilation time (first call triggers JIT)
            print(f"  Measuring compilation time...")
            compile_start = time.time()
            first_trajectories = strategy_fn(
                env=env, 
                model=model, 
                num_rollouts=10,  # Use smaller batch for compilation timing
                episode_length=episode_length, 
                deterministic=deterministic
            )
            # Block until compilation + execution is complete
            first_trajectories = jax.tree.map(lambda x: jax.device_get(x), first_trajectories)
            compile_end = time.time()
            compilation_time = compile_end - compile_start
            
            print(f"    Compilation + first execution: {compilation_time:.4f}s")
            
            # Now measure pure execution time with multiple runs for averaging
            execution_times = []
            for run in range(warmup_runs):
                exec_start = time.time()
                trajectories = strategy_fn(
                    env=env,
                    model=model,
                    num_rollouts=num_rollouts,
                    episode_length=episode_length,
                    deterministic=deterministic
                )
                trajectories = jax.tree.map(lambda x: jax.device_get(x), trajectories)
                exec_end = time.time()
                execution_times.append(exec_end - exec_start)
            
            avg_execution_time = np.mean(execution_times)
            std_execution_time = np.std(execution_times)
            
            # Estimate pure compilation time (rough approximation)
            # This assumes first run = compilation + execution, so we subtract average execution
            estimated_compilation_time = compilation_time - (avg_execution_time * 10 / num_rollouts)
            estimated_compilation_time = max(0, estimated_compilation_time)  # Ensure non-negative
            
            print(f"    Estimated compilation time: {estimated_compilation_time:.4f}s")
            print(f"    Average execution time: {avg_execution_time:.4f}s ± {std_execution_time:.4f}s")
            print(f"    Total time (compilation + execution): {estimated_compilation_time + avg_execution_time:.4f}s")
            
        else:
            # Original behavior: warmup first, then measure execution only
            print(f"  Warming up JIT compilation...")
            for _ in range(3):
                _ = strategy_fn(
                    env=env, 
                    model=model, 
                    num_rollouts=10, 
                    episode_length=episode_length, 
                    deterministic=deterministic
                )
            
            # Measure execution time
            start_time = time.time()
            trajectories = strategy_fn(
                env=env,
                model=model,
                num_rollouts=num_rollouts,
                episode_length=episode_length,
                deterministic=deterministic
            )
            trajectories = jax.tree.map(lambda x: jax.device_get(x), trajectories)
            end_time = time.time()
            
            avg_execution_time = end_time - start_time
            estimated_compilation_time = 0.0  # Not measured
            compilation_time = 0.0
        
        # Compute statistics
        stats = rollout_statistics(trajectories)
        
        results[strategy_key] = {
            'strategy_name': strategy_info['name'],
            'compilation_time': estimated_compilation_time if include_compilation_time else None,
            'execution_time': avg_execution_time,
            'total_time': estimated_compilation_time + avg_execution_time if include_compilation_time else avg_execution_time,
            'rollouts_per_second': num_rollouts / avg_execution_time,
            'statistics': stats,
            'trajectories': trajectories
        }
        
        print(f"    Rollouts/sec: {num_rollouts / avg_execution_time:.2f}")
        print(f"    Mean return: {stats['mean_return']:.4f}")
        print(f"    Mean episode length: {stats.get('mean_episode_length', 'N/A')}")
    
    # Compare results
    print(f"\n{'='*70}")
    print("JIT/VMAP STRATEGY COMPARISON")
    print(f"{'='*70}")
    
    jit_vmap_result = results['jit_vmap']
    vmap_jit_result = results['vmap_jit']
    
    if include_compilation_time:
        print(f"\nCOMPILATION TIME COMPARISON:")
        print(f"  jit(vmap(func)) compilation: {jit_vmap_result['compilation_time']:.4f}s")
        print(f"  vmap(jit(func)) compilation: {vmap_jit_result['compilation_time']:.4f}s")
        comp_diff = jit_vmap_result['compilation_time'] - vmap_jit_result['compilation_time']
        if abs(comp_diff) > 0.001:  # Only show if meaningful difference
            faster_compiler = 'vmap(jit(func))' if comp_diff > 0 else 'jit(vmap(func))'
            print(f"  Faster to compile: {faster_compiler} (by {abs(comp_diff):.4f}s)")
        else:
            print(f"  Compilation times are similar")
    
    print(f"\nEXECUTION TIME COMPARISON:")
    jit_vmap_exec = jit_vmap_result['execution_time']
    vmap_jit_exec = vmap_jit_result['execution_time']
    
    print(f"  jit(vmap(func)) execution: {jit_vmap_exec:.4f}s")
    print(f"  vmap(jit(func)) execution: {vmap_jit_exec:.4f}s")
    
    if jit_vmap_exec < vmap_jit_exec:
        faster_executor = 'jit(vmap(func))'
        exec_speedup = vmap_jit_exec / jit_vmap_exec
    else:
        faster_executor = 'vmap(jit(func))'
        exec_speedup = jit_vmap_exec / vmap_jit_exec
    
    print(f"  Faster executor: {faster_executor} (speedup: {exec_speedup:.2f}x)")
    
    if include_compilation_time:
        print(f"\nTOTAL TIME COMPARISON (compilation + execution):")
        jit_vmap_total = jit_vmap_result['total_time']
        vmap_jit_total = vmap_jit_result['total_time']
        
        print(f"  jit(vmap(func)) total: {jit_vmap_total:.4f}s")
        print(f"  vmap(jit(func)) total: {vmap_jit_total:.4f}s")
        
        if jit_vmap_total < vmap_jit_total:
            faster_overall = 'jit(vmap(func))'
            total_speedup = vmap_jit_total / jit_vmap_total
        else:
            faster_overall = 'vmap(jit(func))'
            total_speedup = jit_vmap_total / vmap_jit_total
        
        print(f"  Faster overall: {faster_overall} (speedup: {total_speedup:.2f}x)")
        
        print(f"\nSTRATEGY RECOMMENDATIONS:")
        print(f"  For one-off usage: Use {faster_overall}")
        print(f"  For repeated calls: Use {faster_executor}")
        
        print(f"\nWHY THE DIFFERENCE:")
        print(f"  jit(vmap(func)): Compiles entire vectorized computation as one function")
        print(f"    - Pros: Better memory layout, can optimize across batch dimension")
        print(f"    - Cons: Longer compilation time, less flexible batch sizes")
        print(f"  vmap(jit(func)): Vectorizes a pre-compiled single function")
        print(f"    - Pros: Faster compilation, works with dynamic batch sizes")
        print(f"    - Cons: May have suboptimal memory patterns, less cross-batch optimization")
    
    # Compare statistics
    jit_vmap_stats = jit_vmap_result['statistics']
    vmap_jit_stats = vmap_jit_result['statistics']
    
    print(f"\nRETURN COMPARISON:")
    print(f"  jit(vmap(func)) mean return: {jit_vmap_stats['mean_return']:.4f}")
    print(f"  vmap(jit(func)) mean return: {vmap_jit_stats['mean_return']:.4f}")
    print(f"  Difference: {abs(jit_vmap_stats['mean_return'] - vmap_jit_stats['mean_return']):.6f}")
    
    if 'mean_episode_length' in jit_vmap_stats and 'mean_episode_length' in vmap_jit_stats:
        print(f"\nEPISODE LENGTH COMPARISON:")
        print(f"  jit(vmap(func)) mean length: {jit_vmap_stats['mean_episode_length']:.2f}")
        print(f"  vmap(jit(func)) mean length: {vmap_jit_stats['mean_episode_length']:.2f}")
        print(f"  Difference: {abs(jit_vmap_stats['mean_episode_length'] - vmap_jit_stats['mean_episode_length']):.4f}")
    
    # Check if results are numerically close
    returns_close = np.allclose(
        compute_returns(jit_vmap_result['trajectories']),
        compute_returns(vmap_jit_result['trajectories']),
        rtol=1e-5, atol=1e-8
    )
    
    print(f"\nResults numerically equivalent: {returns_close}")
    
    return results


def compute_returns(trajectories, gamma=1.0):
    """Compute episode returns with discount factor and proper validity masking."""
    rewards = trajectories['reward']
    
    if 'valid' in trajectories:
        valid_mask = trajectories['valid']
        masked_rewards = rewards * valid_mask
    else:
        masked_rewards = rewards
    
    if gamma == 1.0:
        returns = jnp.sum(masked_rewards, axis=-1)
    else:
        episode_length = masked_rewards.shape[-1]
        discount_factors = jnp.power(gamma, jnp.arange(episode_length))
        discounted_rewards = masked_rewards * discount_factors[None, :]
        returns = jnp.sum(discounted_rewards, axis=-1)
    
    return returns


def rollout_statistics(trajectories, gamma=1.0):
    """Compute comprehensive statistics from rollout data."""
    returns = compute_returns(trajectories, gamma)
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


def example_strategy_comparison():
    """Example usage for comparing JIT/VMAP strategies."""
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
    
    # Compare jit(vmap()) vs vmap(jit()) strategies
    comparison_results = compare_jit_vmap_strategies(
        env=env,
        model=model,
        num_rollouts=100,
        episode_length=1000,
        deterministic=True,
        include_compilation_time=True,
        warmup_runs=2
    )
    
    return comparison_results


if __name__ == "__main__":
    # Run JIT/VMAP strategy comparison
    print("\n" + "="*80)
    print("RUNNING JIT/VMAP STRATEGY COMPARISON")
    print("="*80)
    comparison_results = example_strategy_comparison()