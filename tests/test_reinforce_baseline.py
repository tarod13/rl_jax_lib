try:
    from src.algorithms.reinforce_baseline import REINFORCEwithBaseline
    from src.utils import (
        print_training_summary,
        save_training_plots
    )
except:
    from pathlib import Path
    import sys

    # Define repository root for imports
    _THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = _THIS_FILE.parent.parent
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    from src.algorithms.reinforce_baseline import REINFORCEwithBaseline
    from src.utils import (
        print_training_summary,
        save_training_plots
    )

from dataclasses import dataclass
import tyro
from pathlib import Path
import jax.numpy as jnp


@dataclass
class Config:
    num_rollouts: int = 100
    episode_length: int = 200
    deterministic: bool = False
    seed: int = 42
    hidden_dim: int = 32
    env_name: str = 'ant'
    lr: float = 1e-5
    num_training_steps: int = 10
    num_updates_per_step: int = 5
    gamma: float = 0.99
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    checkpoint_interval: int = 5  # Save every N steps
    resume_from: str | None = None  # 'checkpoint_step_20.pkl'  # Path to checkpoint to resume from
    keep_only_latest: bool = True  # Only keep the most recent checkpoint
    
    # Plotting
    plot_path: str = 'analysis_plots/training_plots.png'


if __name__ == "__main__":
    config = tyro.cli(Config)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("REINFORCE with baseline training")
    print("="*60)
    
    # Resume from checkpoint if specified
    if config.resume_from is not None:
        checkpoint_path = checkpoint_dir / config.resume_from
        print(f"Resuming from checkpoint: {checkpoint_path}")
        agent, training_stats, start_step = REINFORCEwithBaseline.load_checkpoint(checkpoint_path)
        # Update config with potentially new values
        agent.config.num_training_steps = config.num_training_steps
        agent.config.num_updates_per_step = config.num_updates_per_step
        start_step += 1  # Start from next step
    else:
        print("Starting new training")
        agent = REINFORCEwithBaseline(config)
        training_stats = {
            'loss_history': [],
            'grad_norm_history': [],
            'return_history': [],
        }
        start_step = 0
    
    print(f"Environment: {agent.config.env_name}")
    print(f"Training steps: {start_step} → {config.num_training_steps}")
    print(f"Updates per step: {config.num_updates_per_step}")
    print(f"Rollouts per step: {config.num_rollouts}")
    print(f"Learning rate: {config.lr}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Seed: {config.seed}")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print("="*60 + "\n")
    
    # Train the agent with checkpointing
    training_stats_new = agent.train(
        key=None,
        num_steps=config.num_training_steps,
        checkpoint_dir=checkpoint_dir if config.checkpoint_interval > 0 else None,
        checkpoint_interval=config.checkpoint_interval,
        start_step=start_step,
        keep_only_latest=config.keep_only_latest,
    )
    
    # Merge with existing stats if resuming
    if start_step > 0:
        # Concatenate JAX arrays
        training_stats['loss_history'] = jnp.concatenate([
            training_stats['loss_history'],
            training_stats_new['loss_history']
        ], axis=0)
        training_stats['grad_norm_history'] = jnp.concatenate([
            training_stats['grad_norm_history'],
            training_stats_new['grad_norm_history']
        ], axis=0)
        # Concatenate returns if available
        if 'return_history' in training_stats_new and training_stats_new['return_history'] is not None:
            training_stats['return_history'] = jnp.concatenate([
                training_stats['return_history'],
                training_stats_new['return_history']
            ], axis=0)
    else:
        training_stats = training_stats_new
    
    # Save final checkpoint using base class method
    agent.save_checkpoint(checkpoint_dir, config.num_training_steps, training_stats, keep_only_latest=config.keep_only_latest)
    
    # Print summary using utility function
    print_training_summary(training_stats, algorithm_name="REINFORCEwithBaseline")
    
    # Create and save plot using utility function
    save_training_plots(training_stats, config.plot_path, algorithm_name="REINFORCEwithBaseline")