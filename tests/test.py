try:
    from src.algorithms import (
        PPO, REINFORCE, REINFORCEwithBaseline, REINFORCEwithBaselineIS
    )
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
    
    from src.algorithms import (
        PPO, REINFORCE, REINFORCEwithBaseline, REINFORCEwithBaselineIS
    )
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


# Map algorithm names to their classes
ALGORITHM_MAP = {
    'PPO': PPO,
    'REINFORCE': REINFORCE,
    'REINFORCEwithBaseline': REINFORCEwithBaseline,
    'REINFORCEwithBaselineIS': REINFORCEwithBaselineIS,
}


@dataclass
class Config:
    # Algorithm selection
    algorithm: str = 'PPO'  # 'PPO', 'REINFORCE', 'REINFORCEwithBaseline', 'REINFORCEwithBaselineIS'
    
    # Environment and rollout settings
    num_rollouts: int = 1
    episode_length: int = 2048
    deterministic: bool = False
    seed: int = 42
    hidden_dim: int = 64
    env_name: str = 'hopper'
    lr: float = 3e-4
    num_training_steps: int = 500
    gamma: float = 0.99
    num_epochs: int = 10  # Number of epochs/updates-per-step over the collected data
    nl: str = 'tanh'  # Non-linearity for networks: 'relu' or 'tanh'
    use_layernorm: bool = True  # Use LayerNorm after each hidden layer
    
    # Bootstrap for final states (used by PPO and REINFORCEwithBaselineIS)
    use_bootstrap_for_final_states: bool = True
    
    # PPO-specific parameters
    epsilon: float = 0.1  # clipping parameter
    minibatch_size: int | None = 32  # Minibatch size K (None = use full batch)
    max_grad_norm: float | None = 0.5  # Max norm for gradient clipping
    
    # Advantage handling parameters
    use_gae: bool = True  # Use GAE instead of standard returns
    gae_lambda: float = 0.95  # Lambda parameter for GAE (bias-variance tradeoff)
    normalize_advantages: bool = True  # Whether to normalize advantages by substracting mean and dividing by stddev
    
    # Experiment tracking
    experiments_root: str = 'experiments'
    run_id: str | None = None  # Auto-generated if None
    checkpoint_interval: int = 25  # Save every N steps
    keep_only_latest: bool = True  # Only keep the most recent checkpoint
    experiment_ID: str = ""  # Optional experiment ID
    experiment_description: str = ""  # Optional experiment description
    
    # Evaluation settings
    num_eval_episodes: int = 100  # Number of episodes for evaluation
    max_eval_length: int = 10000  # Maximum episode length for evaluation
    run_eval_on_checkpoint: bool = True  # Run evaluation before saving checkpoints
    
    # Resume training
    resume_run_id: str | None = None  # Run ID to resume from
    resume_step: int | None = None  # Specific step to resume from


def validate_config(config: Config) -> None:
    """Validate that the specified algorithm exists."""
    if config.algorithm not in ALGORITHM_MAP:
        available = ", ".join(ALGORITHM_MAP.keys())
        raise ValueError(
            f"Unknown algorithm: {config.algorithm}. "
            f"Available algorithms: {available}"
        )


def get_algorithm_class(algorithm_name: str):
    """Get the algorithm class by name."""
    return ALGORITHM_MAP[algorithm_name]


def print_config_info(config: Config, agent) -> None:
    """Print training configuration and parameters."""
    print("="*60)
    print(f"{config.algorithm} Training with Experiment Manager")
    print("="*60)
    print(f"Environment: {agent.config.env_name}")
    print(f"Algorithm: {config.algorithm}")
    
    # Algorithm-specific parameter info
    if config.algorithm == 'PPO':
        print(f"Training mode: {'Minibatch (PPO-style)' if config.minibatch_size is not None else 'Full batch'}")
        if config.minibatch_size is not None:
            print(f"  Minibatch size: {config.minibatch_size}")
            print(f"  Num epochs: {config.num_epochs}")
            print(f"  Total samples per step: {config.num_rollouts * config.episode_length}")
            print(f"  Updates per step: {config.num_epochs * (config.num_rollouts * config.episode_length // config.minibatch_size)}")
        else:
            print(f"  Num epochs: {config.num_epochs}")
            print(f"  Updates per step: {config.num_epochs}")
    else:
        print(f"Updates per step: {config.num_updates_per_step}")
    
    # GAE information
    if hasattr(config, 'use_gae') and config.use_gae:
        print(f"Advantage estimation: GAE (λ={config.gae_lambda})")
    else:
        print(f"Advantage estimation: Standard returns")
    
    print(f"Rollouts per step: {config.num_rollouts}")
    print(f"Episode length: {config.episode_length}")
    print(f"Learning rate: {agent.config.lr}")
    print(f"Hidden dim: {agent.config.hidden_dim}")
    print(f"Gamma (discount): {agent.config.gamma}")
    print(f"Non-linearity: {agent.config.nl}")
    print(f"Seed: {agent.config.seed}")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print(f"Evaluation episodes: {config.num_eval_episodes}")
    print(f"Run eval on checkpoint: {config.run_eval_on_checkpoint}")
    print("="*60 + "\n")


if __name__ == "__main__":
    config = tyro.cli(Config)
    
    # Validate algorithm choice
    validate_config(config)
    
    AlgorithmClass = get_algorithm_class(config.algorithm)
    
    print("="*60)
    print(f"{config.algorithm} Training")
    print("="*60)
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(
        experiments_root=config.experiments_root,
        run_id=config.run_id
    )
    
    # Resume from checkpoint if specified
    if config.resume_run_id is not None:
        print(f"Resuming from run: {config.resume_run_id}")
        
        # Load config from the run
        run_config_dict = ExperimentManager.load_config(
            config.resume_run_id,
            experiments_root=config.experiments_root
        )
        run_config = SimpleNamespace(**run_config_dict)
        
        # Create experiment manager for the resume run
        resume_exp_manager = ExperimentManager(
            experiments_root=config.experiments_root,
            run_id=config.resume_run_id
        )
        
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
            agent, training_stats, loaded_step = AlgorithmClass.load_checkpoint(
                run_id=config.resume_run_id,
                step=config.resume_step,
                experiment_manager_class=ExperimentManager,
                experiments_root=config.experiments_root
            )
            
            # Update agent's experiment manager to the new run (if creating new run)
            if config.run_id is None:
                agent.experiment_manager = exp_manager
            
            print(f"✅ Resumed from step {loaded_step}")
            start_step = loaded_step
        else:
            # Start fresh but use the config from the resumed run
            agent = AlgorithmClass(run_config, experiment_manager=exp_manager)
            training_stats = {
                'loss_history': [],
                'grad_norm_history': [],
                'return_history': [],
                'eval_history': [],
            }
            start_step = 0
    else:
        print("Starting new training run")
        agent = AlgorithmClass(config, experiment_manager=exp_manager)
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
    print_config_info(config, agent)
    
    print(f"Training steps: {start_step} → {config.num_training_steps}\n")
    
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
    print_training_summary(training_stats, algorithm_name=config.algorithm)
    
    # Create and save plot using utility function
    plot_path = exp_manager.get_plot_path(config.algorithm)
    save_training_plots(training_stats, plot_path, algorithm_name=config.algorithm)
    
    print(f"\n✅ Training complete!")
    print(f"Algorithm: {config.algorithm}")
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