import jax
import jax.numpy as jnp
from tqdm import tqdm

from .base import RLAlgorithm
from ..utils import vectorized_rollouts, compute_returns, tree_norm


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
        trajectories, key = vectorized_rollouts(
            env=self.env,
            model=self.network,
            key=key,
            num_rollouts=self.config.num_rollouts,
            episode_length=self.config.episode_length,
            deterministic=self.config.deterministic,
        )

        _, returns = compute_returns(
            trajectories, gamma=self.config.gamma,
        )

        return trajectories, returns, key

    def train(self, key, num_steps=None, checkpoint_dir=None, checkpoint_interval=0, start_step=0, keep_only_latest=True):
        """
        Train the on-policy agent.
        
        Args:
            key: Random key for rollouts
            num_steps: Number of training steps (if None, uses config)
            checkpoint_dir: Directory to save checkpoints (if None, no checkpoints)
            checkpoint_interval: Save checkpoint every N steps (0 = no checkpoints)
            start_step: Starting step number (for resuming)
            keep_only_latest: If True, only keep the most recent checkpoint (default: True)
            
        Returns:
            training_stats: Dictionary with 'loss_history', 'grad_norm_history', and 'return_history'
        """
        if key is None:
            key = jax.random.PRNGKey(self.config.seed)
        
        if num_steps is None:
            num_steps = self.config.num_training_steps

        return_history = []
        loss_history = []
        grad_norm_history = []

        # Training loop
        for training_step in tqdm(range(num_steps), desc="Training Steps", leave=True):
            # Collect rollouts
            key, step_key = jax.random.split(key)
            trajectories, returns, _ = self.collect_rollouts(key=step_key)

            step_losses = []
            step_grad_norms = []
            
            # Perform updates for this step
            for update_ in range(self.config.num_updates_per_step):
                # Call the algorithm-specific update method
                loss, grads = self.update(
                    trajectories['obs'],
                    trajectories['action'],
                    returns,
                )
                step_losses.append(loss)

                # Calculate and log gradient norms
                grad_norm = tree_norm(grads)
                step_grad_norms.append(grad_norm)

            # Print progress statistics
            actual_step = start_step + training_step + 1
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
            if checkpoint_dir is not None and checkpoint_interval > 0:
                if actual_step % checkpoint_interval == 0:
                    checkpoint_stats = {
                        'return_history': jnp.array(return_history),
                        'loss_history': jnp.array(loss_history),
                        'grad_norm_history': jnp.array(grad_norm_history),
                    }
                    self.save_checkpoint(
                        checkpoint_dir, 
                        actual_step,
                        checkpoint_stats,
                        keep_only_latest=keep_only_latest
                    )   

        training_stats = {
            'return_history': jnp.array(return_history),
            'loss_history': jnp.array(loss_history),
            'grad_norm_history': jnp.array(grad_norm_history),
        }
        return training_stats