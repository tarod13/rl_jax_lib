import jax
import jax.numpy as jnp
from tqdm import tqdm
from flax import nnx


from .base import RLAlgorithm
from ..utils import (
    vectorized_rollouts_multi_env, compute_returns, tree_norm, 
    evaluate_agent, print_evaluation_summary,
)

ALGORITHMS_WITH_ADVANTAGE_ESTIMATION = [
    'ppo',
]
ALGORITHMS_WITH_LOG_PROBS = [
    'ppo',
    'reinforcewithbaseline',
    'reinforcewithbaselineis',
]


class OnPolicyAlgorithm(RLAlgorithm):
    """Base for on-policy algorithms (REINFORCE, PPO, TRPO)."""

    def collect_rollouts(self, key):
        # Check necessary config parameters
        required_params = ['num_rollouts', 'episode_length', 'deterministic']
        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing required config parameter: {param}")
            
        # Check network has been initialized
        if not hasattr(self, 'network'):
            raise ValueError("Network must be initialized before collecting rollouts.")

        # Implement rollout collection logic
        last_states, trajectories, key = vectorized_rollouts_multi_env(
            env=self.env,
            model=self.network,
            key=key,
            num_rollouts=self.config.num_rollouts,
            episode_length=self.config.episode_length,
            deterministic=self.config.deterministic,
            initial_states=self.initial_states,
        )

        # Update initial states for next call
        self.initial_states = nnx.data(last_states)

        # Compute final next values for return calculation initialization
        if hasattr(self.network, 'critic'):
            final_next_values = self.network.critic(trajectories['next_obs'][:,-1])
            final_next_values *= (1 - trajectories['done'][:,-1])
        else:
            final_next_values = jnp.zeros((self.config.num_rollouts,))

        # Compute returns if needed
        calculate_advantages = self.config.algorithm in ALGORITHMS_WITH_ADVANTAGE_ESTIMATION
        use_gae = getattr(self.config, 'use_gae', False)  # Get use_gae flag from config
        calculate_returns = not calculate_advantages or not use_gae
        if calculate_returns:
            _, returns = compute_returns(
                    trajectories, gamma=self.config.gamma, init_returns=final_next_values,
                )
            advantages = None
        else:
            returns = None

        if calculate_advantages:
            # Compute values for advantage estimation if needed
            if not hasattr(self.network, 'critic'):
                raise ValueError("Advantage estimation requires a critic network, but network has no critic.")

            # Compute values for all timesteps
            # Shape: [num_rollouts, episode_length]
            N, T = trajectories['obs'].shape[:2]
            values = self.network.critic(trajectories['obs'].reshape(N * T, -1))
            values = values.reshape(N, T)

            if use_gae:
                gae_lambda = getattr(self.config, 'gae_lambda', 0.95)        
                next_values = jnp.concatenate(
                    [values[:,1:], final_next_values[:,None]], axis=1
                )
                
                # Compute GAE advantages
                _, advantages = compute_returns(
                    trajectories, 
                    gamma=self.config.gamma,
                    use_gae=True,
                    gae_lambda=gae_lambda,
                    values=values,
                    next_values=next_values,
                )
                
                # For PPO and similar algorithms, we also need the value targets
                # Value targets = advantages + values
                returns = advantages + values
                
            else:
                # Standard return computation
                advantages = returns - values

        return trajectories, returns, advantages, key

    def train(self, key, num_steps=None, checkpoint_interval=0, keep_only_latest=True, 
              num_eval_episodes=10, max_eval_length=1000, run_eval_on_checkpoint=True):
        """
        Train the on-policy agent.
        
        Args:
            key: Random key for rollouts
            num_steps: Number of training steps (if None, uses config)
            checkpoint_interval: Save checkpoint every N steps (0 = no checkpoints)
            keep_only_latest: If True, only keep the most recent checkpoint (default: True)
            num_eval_episodes: Number of episodes for evaluation (default: 10)
            max_eval_length: Maximum episode length for evaluation (default: 1000)
            run_eval_on_checkpoint: If True, run evaluation before saving checkpoints (default: True)
            
        Returns:
            training_stats: Dictionary with 'loss_history', 'grad_norm_history', 'return_history', and 'eval_history'
        """
        if key is None:
            key = jax.random.PRNGKey(self.config.seed)
        
        if num_steps is None:
            num_steps = self.config.num_training_steps

        return_history = []
        loss_history = []
        grad_norm_history = []
        eval_history = []  # Track evaluation metrics

        num_epochs = getattr(self.config, 'num_epochs', 1)

        # Training loop
        for training_step in tqdm(range(num_steps), desc="Training Steps", leave=True):
            # Collect rollouts
            key, step_key = jax.random.split(key)
            trajectories, returns, advantages, _ = self.collect_rollouts(key=step_key)

            step_losses = []
            step_grad_norms = []
            
            # Flatten trajectories from [N, T, ...] to [NT, ...]
            N, T = trajectories['obs'].shape[:2]
            total_samples = N * T
            
            flat_obs = trajectories['obs'].reshape(total_samples, -1)
            flat_actions = trajectories['action'].reshape(total_samples, -1)
            flat_returns = returns.reshape(total_samples)
            if advantages is not None:
                flat_advantages = advantages.reshape(total_samples)
                normalize_advantages = getattr(self.config, 'normalize_advantages', False)
                if normalize_advantages:
                    flat_advantages = (flat_advantages - jnp.mean(flat_advantages)) / (jnp.std(flat_advantages) + 1e-8)
            else:
                flat_advantages = None
            
            # Set minibatch_size to full batch if None
            minibatch_size = getattr(self.config, 'minibatch_size', None)
            if minibatch_size is None:
                minibatch_size = total_samples
            
            # Compute number of minibatches
            num_minibatches = total_samples // minibatch_size
            
            # Compute old_log_probs in minibatches if needed
            old_log_probs_all = None
            need_old_log_probs = self.config.algorithm in ALGORITHMS_WITH_LOG_PROBS
            if need_old_log_probs:
                old_log_probs_list = []
                
                # Compute old_log_probs in minibatches
                for mb_idx in range(num_minibatches):
                    start_idx = mb_idx * minibatch_size
                    end_idx = start_idx + minibatch_size
                    
                    mb_obs = flat_obs[start_idx:end_idx]
                    mb_actions = flat_actions[start_idx:end_idx]
                    
                    # Compute log probs for this minibatch
                    mb_old_log_probs = self.get_log_prob(self.network, mb_obs, mb_actions)
                    old_log_probs_list.append(mb_old_log_probs)
                
                # Concatenate all minibatch log probs
                old_log_probs_all = jnp.concatenate(old_log_probs_list, axis=0)
            
            # Perform multiple epochs of training
            for epoch in range(num_epochs):
                # Shuffle indices for this epoch
                key, shuffle_key = jax.random.split(key)
                indices = jax.random.permutation(shuffle_key, total_samples)
                
                # Split into minibatches and update
                for mb_idx in range(num_minibatches):
                    start_idx = mb_idx * minibatch_size
                    end_idx = start_idx + minibatch_size
                    mb_indices = indices[start_idx:end_idx]
                    
                    # Extract minibatch
                    mb_obs = flat_obs[mb_indices]
                    mb_actions = flat_actions[mb_indices]
                    mb_returns = flat_returns[mb_indices]
                    if flat_advantages is not None:
                        mb_advantages = flat_advantages[mb_indices]
                    else:
                        mb_advantages = None
                    
                    # If old_log_probs exist, index into them for this minibatch
                    if old_log_probs_all is not None:
                        mb_old_log_probs = old_log_probs_all[mb_indices]
                    else:
                        mb_old_log_probs = None

                    # Update network
                    loss, grads = self.update(
                        mb_obs,
                        mb_actions,
                        mb_returns,
                        mb_advantages,
                        mb_old_log_probs,
                    )
                    step_losses.append(loss)
                    
                    # Calculate and log gradient norms
                    grad_norm = tree_norm(grads)
                    step_grad_norms.append(grad_norm)

            # Print progress statistics
            actual_step = training_step + 1
            episode_returns = returns[:, 0]  # Total episode returns
            mean_return = jnp.mean(episode_returns)
            std_return = jnp.std(episode_returns)
            mean_loss = jnp.mean(jnp.array(step_losses))
            mean_grad_norm = jnp.mean(jnp.array(step_grad_norms))
            
            tqdm.write(
                f"Step {actual_step:3d} | "
                f"Return: {mean_return:8.2f} ± {std_return:6.2f} | "
                f"Loss: {mean_loss:8.4f} | "
                f"Grad Norm: {mean_grad_norm:8.4e}"
            )

            # Log data
            return_history.append(episode_returns)
            loss_history.append(step_losses)
            grad_norm_history.append(step_grad_norms)

            # Save checkpoint if requested
            if checkpoint_interval > 0 and actual_step % checkpoint_interval == 0:
                # Run evaluation before saving checkpoint
                if run_eval_on_checkpoint:
                    key, eval_key = jax.random.split(key)
                    eval_stats = evaluate_agent(
                        agent=self,
                        key=eval_key,
                        num_eval_episodes=num_eval_episodes,
                        max_episode_length=max_eval_length,
                    )
                    eval_history.append({
                        'step': actual_step,
                        'stats': eval_stats,
                    })
                    print_evaluation_summary(eval_stats, step=actual_step)
                
                if self.experiment_manager is not None:
                    checkpoint_stats = {
                        'return_history': jnp.array(return_history),
                        'loss_history': jnp.array(loss_history),
                        'grad_norm_history': jnp.array(grad_norm_history),
                        'eval_history': eval_history,
                    }
                    self.save_checkpoint(
                        actual_step,
                        checkpoint_stats,
                        keep_only_latest=keep_only_latest
                    )

        training_stats = {
            'return_history': jnp.array(return_history),
            'loss_history': jnp.array(loss_history),
            'grad_norm_history': jnp.array(grad_norm_history),
            'eval_history': eval_history,
        }
        return training_stats