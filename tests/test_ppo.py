try:
    from src.algorithms.ppo import PPO
    from src.utils import (
        print_training_summary,
        save_training_plots,
        ExperimentManager,
    )
except:
    from pathlib import Path
    import sys

    # Define repository root for imports
    _THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = _THIS_FILE.parent.parent
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    from src.algorithms.ppo import PPO
    from src.utils import (
        print_training_summary,
        save_training_plots,
        ExperimentManager,
    )

from dataclasses import dataclass
import tyro
from pathlib import Path
import jax.numpy as jnp
from types import SimpleNamespace


@dataclass
class Config:
    num_rollouts: int = 100
    episode_length: int = 200
    deterministic: bool = False
    seed: int = 42
    hidden_dim: int = 32
    env_name: str = 'ant'
    lr: float = 1e-4
    num_training_steps: int = 10
    num_updates_per_step: int = 5
    gamma: float = 0.99
    epsilon: float = 0.2
    use_bootstrap_for_final_states: bool = True
    
    # Experiment tracking
    experiments_root: str = 'experiments'
    run_id: str | None = None  # Auto-generated if None
    checkpoint_interval: int = 5  # Save every N steps
    keep_only_latest: bool = True  # Only keep the most recent checkpoint
    
    # Evaluation settings
    num_eval_episodes: int = 100  # Number of episodes for evaluation
    max_eval_length: int = 10000  # Maximum episode length for evaluation
    run_eval_on_checkpoint: bool = True  # Run evaluation before saving checkpoints
    
    # Resume training
    resume_run_id: str | None = None  # Run ID to resume from
    resume_step: int | None = None  # Specific step to resume from


if __name__ == "__main__":
    config = tyro.cli(Config)
    
    print("="*60)
    print("PPO Training with Experiment Manager")
    print("="*60)
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(
        experiments_root=config.experiments_root,
        run_id=config.run_id
    )
    
    # Resume from checkpoint if specified
    if config.resume_run_id is not None:
        print(f"Resuming from run: {config.resume_run_id}")
        
        # Load the run's experiment manager and config
        resume_exp_manager = ExperimentManager(
            experiments_root=config.experiments_root,
            run_id=config.resume_run_id
        )
        
        # Load config from the run
        run_config_dict = ExperimentManager.load_config(
            config.resume_run_id,
            experiments_root=config.experiments_root
        )
        run_config = SimpleNamespace(**run_config_dict)
        
        # Determine which step to load from
        if config.resume_step is None:
            # Load from latest checkpoint
            available_steps = resume_exp_manager.list_checkpoints()
            if available_steps:
                config.resume_step = available_steps[-1]
            else:
                print("⚠️  No checkpoints found in resume run. Starting fresh.")
                config.resume_step = None
        
        if config.resume_step is not None:
            # Load agent from checkpoint
            agent, training_stats, loaded_step = PPO.load_checkpoint(
                run_id=config.resume_run_id,
                step=config.resume_step,
                experiment_manager_class=ExperimentManager,
                experiments_root=config.experiments_root
            )
            
            # Update agent's experiment manager to the new run (if creating new run)
            # or keep the old one (if continuing in same run)
            if config.run_id is None:
                # If no run_id specified, we're in a new run
                agent.experiment_manager = exp_manager
            
            print(f"✅ Resumed from step {loaded_step}")
            start_step = loaded_step
        else:
            # Start fresh but use the config from the resumed run
            agent = PPO(run_config, experiment_manager=exp_manager)
            training_stats = {
                'loss_history': [],
                'grad_norm_history': [],
                'return_history': [],
                'eval_history': [],
            }
            start_step = 0
    else:
        print("Starting new training run")
        agent = PPO(config, experiment_manager=exp_manager)
        training_stats = {
            'loss_history': [],
            'grad_norm_history': [],
            'return_history': [],
            'eval_history': [],
        }
        start_step = 0
        
        # Save config for this run
        exp_manager.save_config(config)
    
    # Print run information
    exp_manager.print_run_info()
    
    print(f"Environment: {agent.config.env_name}")
    print(f"Training steps: {start_step} → {config.num_training_steps}")
    print(f"Updates per step: {config.num_updates_per_step}")
    print(f"Rollouts per step: {config.num_rollouts}")
    print(f"Learning rate: {agent.config.lr}")
    print(f"Hidden dim: {agent.config.hidden_dim}")
    print(f"Seed: {agent.config.seed}")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print(f"Evaluation episodes: {config.num_eval_episodes}")
    print(f"Run eval on checkpoint: {config.run_eval_on_checkpoint}")
    print("="*60 + "\n")
    
    # Train the agent with checkpointing and evaluation
    training_stats_new = agent.train(
        key=None,
        num_steps=config.num_training_steps - start_step,
        checkpoint_interval=config.checkpoint_interval,
        keep_only_latest=config.keep_only_latest,
        num_eval_episodes=config.num_eval_episodes,
        max_eval_length=config.max_eval_length,
        run_eval_on_checkpoint=config.run_eval_on_checkpoint,
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
        # Merge eval_history (it's a list of dicts, not a JAX array)
        if 'eval_history' in training_stats_new:
            training_stats['eval_history'] = training_stats.get('eval_history', []) + training_stats_new['eval_history']
    else:
        training_stats = training_stats_new
    
    # Save final checkpoint
    agent.save_checkpoint(
        config.num_training_steps,
        training_stats,
        keep_only_latest=config.keep_only_latest
    )
    
    # Print summary using utility function
    print_training_summary(training_stats, algorithm_name="PPO")
    
    # Create and save plot using utility function
    plot_path = exp_manager.get_plot_path("PPO")
    save_training_plots(training_stats, plot_path, algorithm_name="PPO")
    
    print(f"\n✅ Training complete!")
    print(f"Run ID: {exp_manager.run_id}")
    print(f"Results saved to: {exp_manager.run_dir}")
    
    # Print evaluation summary if we have eval data
    if training_stats.get('eval_history'):
        print("\n" + "="*60)
        print("EVALUATION HISTORY SUMMARY")
        print("="*60)
        for eval_data in training_stats['eval_history']:
            step = eval_data['step']
            stats = eval_data['stats']
            print(f"Step {step:3d}: Mean Return = {stats['mean_return']:8.2f} ± {stats['std_return']:6.2f}")
        print("="*60)