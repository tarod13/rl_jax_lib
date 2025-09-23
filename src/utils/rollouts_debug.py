import jax
import jax.numpy as jnp
from brax import envs
import functools
from flax import nnx
import numpy as np
import pickle
import os
from datetime import datetime

# Import your network - adjust path as needed
from ..networks import ActorCriticNetwork


def check_for_nans_numpy(data, name):
    """Helper function to check for NaN values in numpy data (non-JAX)."""
    if isinstance(data, dict):
        for key, value in data.items():
            if np.any(np.isnan(np.array(value))):
                print(f"WARNING: NaN found in {name}['{key}']")
                return True
    elif np.any(np.isnan(np.array(data))):
        print(f"WARNING: NaN found in {name}")
        return True
    return False

def safe_action_clipping(action, limits):
    """Safely clip actions to prevent NaN propagation."""
    # Replace any NaN with zeros
    action = jnp.where(jnp.isnan(action), 0.0, action)
    
    # Clip to limits if provided
    if limits is not None:
        action = jnp.clip(action, limits[:, 0], limits[:, 1])
    
    return action

def single_rollout(env, model, key, episode_length=1000, deterministic=True):
    """
    Generate a single rollout with proper early termination handling.
    """
    # Reset environment
    initial_state = env.reset(key)
    
    def step_fn(carry, step_idx):
        state, done_flag = carry
        obs = state.obs
        
        # Get action from model (only if not done)
        if deterministic:
            action = model.get_deterministic_action(obs)
        else:
            action, _, _, _ = model.sample_action(obs)
        
        # Safely clip actions (handles NaN)
        action = safe_action_clipping(action, getattr(env.sys, 'actuator_ctrlrange', None))
        
        # Only step environment if not already done
        # If done, keep the same state to avoid invalid states
        next_state = jax.lax.cond(
            done_flag,
            lambda s, a: s,  # If done, keep same state
            lambda s, a: env.step(s, a),  # If not done, step normally
            state, action
        )
        
        # Update done flag
        new_done_flag = done_flag | next_state.done.astype(bool)

        # Update reward
        reward = jnp.where(done_flag, 0.0, next_state.reward)  # Zero reward if already done
        
        # Store trajectory data with validity mask
        trajectory_step = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': new_done_flag,
            'next_obs': next_state.obs,
            'step_idx': step_idx,
            'valid': ~done_flag  # This step is valid if we weren't done before
        }
        
        return (next_state, new_done_flag), trajectory_step
    
    # Run rollout using scan for efficiency
    initial_done = jnp.array(False, dtype=bool)
    (final_state, final_done), trajectory = jax.lax.scan(
        step_fn,
        (initial_state, initial_done),  # Initial carry: (state, done_flag)
        jnp.arange(episode_length)
    )
    
    return trajectory

def debug_single_rollout(env, model, key, episode_length=10):
    """
    Debug version that runs a short rollout WITHOUT JIT for debugging.
    This version can have print statements and conditionals.
    """
    print("=== DEBUG ROLLOUT (NO JIT) ===")
    
    # Reset environment
    state = env.reset(key)
    print(f"Initial state obs shape: {state.obs.shape}")
    
    # Check initial observation
    initial_obs = np.array(state.obs)
    if check_for_nans_numpy(initial_obs, "initial_obs"):
        return None
    
    trajectory_data = []
    
    for step in range(episode_length):
        obs = state.obs
        print(f"Step {step}: obs range [{np.min(obs):.3f}, {np.max(obs):.3f}]")
        
        # Get action from model
        try:
            if hasattr(model, 'get_deterministic_action'):
                action = model.get_deterministic_action(obs)
            else:
                # Fallback for models without this method
                action, _, _, _ = model.sample_action(obs)
            
            action_np = np.array(action)
            print(f"Step {step}: action range [{np.min(action_np):.3f}, {np.max(action_np):.3f}]")
            
            if check_for_nans_numpy(action_np, f"action_step_{step}"):
                print(f"❌ NaN detected in action at step {step}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting action at step {step}: {e}")
            return None
        
        # Safely clip actions
        action = safe_action_clipping(action, getattr(env.sys, 'actuator_ctrlrange', None))
        
        # Step environment
        try:
            next_state = env.step(state, action)
            reward_np = np.array(next_state.reward)
            
            print(f"Step {step}: reward = {reward_np:.3f}")
            
            if check_for_nans_numpy(reward_np, f"reward_step_{step}"):
                print(f"❌ NaN detected in reward at step {step}")
                return None
                
        except Exception as e:
            print(f"❌ Error stepping environment at step {step}: {e}")
            return None
        
        # Store step data
        trajectory_data.append({
            'obs': np.array(obs),
            'action': np.array(action),
            'reward': np.array(next_state.reward),
            'done': np.array(next_state.done),
            'next_obs': np.array(next_state.obs)
        })
        
        # Update state
        state = next_state
        
        # Early termination if done
        if next_state.done:
            print(f"Episode terminated early at step {step}")
            break
    
    print("✅ Debug rollout completed successfully")
    return trajectory_data

def vectorized_rollouts(env, model, num_rollouts=100, episode_length=1000, deterministic=True):
    """
    Generate multiple rollouts in parallel using JAX vectorization.
    """
    print(f"Generating {num_rollouts} rollouts...")
    
    # Create rollout function
    rollout_fn = functools.partial(
        single_rollout,
        env,
        model,
        episode_length=episode_length,
        deterministic=deterministic
    )
    
    # JIT compile for speed
    rollout_fn = jax.jit(rollout_fn)
    
    # Generate random keys for each rollout
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_rollouts)
    
    # Vectorize across rollouts
    trajectories = jax.vmap(rollout_fn)(keys)
    
    return trajectories

def test_model_and_env(env, model, num_test_steps=10):
    """
    Test model and environment compatibility before running full rollouts.
    """
    print("=== TESTING MODEL AND ENVIRONMENT ===")
    
    # Test 1: Environment reset
    key = jax.random.PRNGKey(42)
    try:
        state = env.reset(key)
        print(f"✅ Environment reset successful")
        print(f"   Observation shape: {state.obs.shape}")
        print(f"   Observation range: [{np.min(state.obs):.3f}, {np.max(state.obs):.3f}]")
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return False
    
    # Test 2: Model forward pass
    try:
        if hasattr(model, 'get_deterministic_action'):
            test_action = model.get_deterministic_action(state.obs)
        else:
            test_action, _, _, _ = model.sample_action(state.obs)
        
        print(f"✅ Model forward pass successful")
        print(f"   Action shape: {test_action.shape}")
        print(f"   Action range: [{np.min(test_action):.3f}, {np.max(test_action):.3f}]")
        
        if check_for_nans_numpy(test_action, "test_action"):
            print("❌ Model produces NaN actions")
            return False
            
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False
    
    # Test 3: Environment step
    try:
        safe_action = safe_action_clipping(test_action, getattr(env.sys, 'actuator_ctrlrange', None))
        next_state = env.step(state, safe_action)
        
        print(f"✅ Environment step successful")
        print(f"   Reward: {next_state.reward:.3f}")
        print(f"   Done: {next_state.done}")
        
        if check_for_nans_numpy(next_state.reward, "test_reward"):
            print("❌ Environment produces NaN rewards")
            return False
            
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        return False
    
    # Test 4: Short debug rollout
    print(f"\n=== RUNNING {num_test_steps}-STEP DEBUG ROLLOUT ===")
    debug_traj = debug_single_rollout(env, model, key, episode_length=num_test_steps)
    
    if debug_traj is None:
        print("❌ Debug rollout failed")
        return False
    
    print("✅ All tests passed!")
    return True

def compute_returns(trajectories, gamma=1.0):
    """
    Compute episode returns with discount factor and proper validity masking.
    """
    rewards = trajectories['reward']
    
    # Use validity mask if available
    if 'valid' in trajectories:
        valid_mask = trajectories['valid']
        # Mask out invalid rewards
        masked_rewards = rewards * valid_mask
    else:
        masked_rewards = rewards
    
    if gamma == 1.0:
        # Undiscounted returns
        returns = jnp.sum(masked_rewards, axis=-1)
    else:
        # Discounted returns
        episode_length = masked_rewards.shape[-1]
        discount_factors = jnp.power(gamma, jnp.arange(episode_length))
        discounted_rewards = masked_rewards * discount_factors[None, :]
        returns = jnp.sum(discounted_rewards, axis=-1)
    
    return returns

def rollout_statistics(trajectories, gamma=1.0):
    """
    Compute comprehensive statistics from rollout data with validity handling.
    """
    returns = compute_returns(trajectories, gamma)
    rewards = trajectories['reward']
    
    # Convert to numpy for statistics computation
    returns_np = np.array(returns)
    rewards_np = np.array(rewards)
    
    # Calculate episode lengths if validity mask is available
    episode_lengths = None
    if 'valid' in trajectories:
        valid_mask = np.array(trajectories['valid'])
        episode_lengths = np.sum(valid_mask, axis=1)
    
    # Basic statistics
    stats = {
        'num_rollouts': len(returns_np),
        'episode_length': rewards_np.shape[-1],
        
        # Return statistics
        'mean_return': float(np.mean(returns_np)),
        'std_return': float(np.std(returns_np)),
        'min_return': float(np.min(returns_np)),
        'max_return': float(np.max(returns_np)),
        'median_return': float(np.median(returns_np)),
        
        # Reward statistics
        'mean_reward_per_step': float(np.mean(rewards_np)),
        'std_reward_per_step': float(np.std(rewards_np)),
        
        # Check for data quality issues
        'returns_with_nan': int(np.sum(np.isnan(returns_np))),
        'rewards_with_nan': int(np.sum(np.isnan(rewards_np))),
        'returns_finite': int(np.sum(np.isfinite(returns_np))),
    }
    
    # Add episode length statistics if available
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
    """
    Save trajectory data with metadata.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectories_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    # Prepare data to save
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
    
    # Convert JAX arrays to numpy for saving
    save_data = jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, save_data)
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Trajectories saved to: {filepath}")
    return filepath

def create_dummy_model(action_dim):
    """Create a simple JAX-compatible dummy model for testing."""
    class DummyModel:
        def __init__(self, action_dim):
            self.action_dim = action_dim
            # Fixed key for deterministic behavior
            self.key = jax.random.PRNGKey(42)
            
        def get_deterministic_action(self, obs):
            # Simple policy: small actions proportional to observation
            # Use a linear transformation of observations to actions
            # This is deterministic and JAX-compatible
            
            # Simple linear policy: action = tanh(W * obs) where W is derived from obs
            # Create a pseudo-weight matrix from the observation
            obs_sum = jnp.sum(obs)
            
            # Create action as a simple function of observations
            action = jnp.tanh(obs[:self.action_dim] * 0.1) if len(obs) >= self.action_dim else jnp.zeros(self.action_dim)
            
            # If obs is shorter than action_dim, pad with zeros
            if len(obs) < self.action_dim:
                action = jnp.zeros(self.action_dim)
            else:
                # Take first action_dim elements and scale them
                action = jnp.tanh(obs[:self.action_dim] * 0.1)
            
            return action
        
        def sample_action(self, obs):
            action = self.get_deterministic_action(obs)
            # For dummy model, return zeros for other values
            log_prob = jnp.zeros_like(action)
            value = jnp.array(0.0)
            entropy = jnp.array(0.0)
            return action, log_prob, value, entropy
    
    return DummyModel(action_dim)

def example_usage():
    """
    Example usage with proper testing and debugging flow.
    """
    print("=== TRAJECTORY GENERATION EXAMPLE ===")
    
    # Create environment
    env_name = 'ant'
    env = envs.get_environment(env_name)
    action_dim = env.action_size
    obs_dim = env.observation_size
    limits = getattr(env.sys, 'actuator_ctrlrange', None)
    
    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    if limits is not None:
        print(f"Action limits: [{limits[:, 0].min():.2f}, {limits[:, 1].max():.2f}]")

    # Create model - replace this with your actual model loading
    print("\n=== LOADING MODEL ===")
    # print("NOTE: Using dummy model for demonstration")
    # print("Replace this section with your actual ActorCriticNetwork loading")
    
    # """
    # Uncomment and modify this section for your actual model:
    model = ActorCriticNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        limits=limits,
        rngs=nnx.Rngs(42)
    )
    
    # # Initialize parameters with dummy forward pass
    # key = jax.random.PRNGKey(42)
    # dummy_obs = jnp.ones((1, obs_dim))
    # _ = model(dummy_obs)
    
    # # Load your trained weights here
    # # model.load_state_dict(your_trained_weights)
    # """
    
    # # For demonstration, use dummy model
    # model = create_dummy_model(action_dim)
    
    # Test model and environment before running full rollouts
    if not test_model_and_env(env, model, num_test_steps=5):
        print("❌ Model/environment test failed. Please fix the issues before proceeding.")
        return None, None
    
    print("\n=== GENERATING ROLLOUTS ===")
    
    # Start with a small number of rollouts but test longer episodes
    print("Testing with longer episodes (1000 steps)...")
    trajectories = vectorized_rollouts(
        env=env,
        model=model,
        num_rollouts=10,
        episode_length=1000,  # Now testing with longer episodes
        deterministic=True
    )
    
    print("✅ Rollouts generated successfully!")
    
    # Compute statistics
    print("\n=== COMPUTING STATISTICS ===")
    stats = rollout_statistics(trajectories)
    
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Additional analysis for early termination
    if 'valid' in trajectories:
        valid_steps = np.array(trajectories['valid'])
        episode_lengths = np.sum(valid_steps, axis=1)
        print(f"\nEpisode Length Analysis:")
        print(f"  Episodes that terminated early: {np.sum(episode_lengths < 1000)}/{len(episode_lengths)}")
        print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"  Shortest episode: {np.min(episode_lengths)}")
        print(f"  Longest episode: {np.max(episode_lengths)}")
    
    # Check for issues
    if stats['returns_with_nan'] > 0:
        print(f"⚠️  WARNING: {stats['returns_with_nan']} episodes have NaN returns!")
    else:
        print("✅ No NaN values detected in returns")
        
    if stats.get('episodes_terminated_early', 0) > 0:
        print(f"ℹ️  {stats['episodes_terminated_early']} episodes terminated before max length")
    
    # Save trajectories
    print("\n=== SAVING TRAJECTORIES ===")
    save_path = save_trajectories(trajectories)
    
    return trajectories, save_path

if __name__ == "__main__":
    trajectories, save_path = example_usage()