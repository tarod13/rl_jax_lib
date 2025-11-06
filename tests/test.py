from pathlib import Path
import jax.numpy as jnp
from types import SimpleNamespace

try:
    from src.config import ExperimentConfig
    
except:
    import sys

    # Define repository root for imports
    _THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = _THIS_FILE.parent.parent
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.config import ExperimentConfig
    
from src.algorithms import (
    PPO, REINFORCE, REINFORCEwithBaseline, REINFORCEwithBaselineIS
)
from src.utils import (
    print_training_summary,
    save_training_plots,
    ExperimentManager,
    parse_args,
    create_config_from_args,
)

from src.config import (
    OnPolicyConfig,
    PPOConfig,
)


# Map algorithm names to their classes
ALGORITHM_MAP = {
    'ppo': PPO,
    'reinforce': REINFORCE,
    'reinforcewithbaseline': REINFORCEwithBaseline,
    'reinforcewithbaselineis': REINFORCEwithBaselineIS,
}

# Map algorithm names to their config classes
ALGORITHM_CONFIG_MAP = {
    'ppo': PPOConfig,
    'reinforce': OnPolicyConfig,
    'reinforcewithbaseline': OnPolicyConfig,
    'reinforcewithbaselineis': OnPolicyConfig,
}


def get_algorithm_from_args_or_resume(args: dict, experiments_root: str = 'experiments') -> str:
    """Determine algorithm from args or resume."""
    # Check if resuming
    if 'resume_run_id' in args:
        try:
            run_config_dict = ExperimentManager.load_config(
                args['resume_run_id'],
                experiments_root=experiments_root
            )
            algorithm = run_config_dict.get('algorithm', 'PPO')
            print(f"Resuming with algorithm: {algorithm}")
            return algorithm
        except:
            pass
    
    # Check for algorithm in args
    return args.get('algorithm', 'PPO')


def validate_config(config) -> None:
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


def print_config_info(exp_config, algo_config, agent) -> None:
    """Print training configuration and parameters."""
    print("="*60)
    print(f"{algo_config.algorithm} Training with Experiment Manager")
    print("="*60)
    print(f"Environment: {agent.config.env_name}")
    print(f"Algorithm: {algo_config.algorithm}")
    
    # Algorithm-specific parameter info
    if algo_config.algorithm == 'PPO':
        minibatch_size = getattr(algo_config, 'minibatch_size', None)
        print(f"Training mode: {'Minibatch (PPO-style)' if minibatch_size is not None else 'Full batch'}")
        if minibatch_size is not None:
            print(f"  Minibatch size: {minibatch_size}")
            print(f"  Num epochs: {algo_config.num_epochs}")
            print(f"  Total samples per step: {algo_config.num_rollouts * algo_config.episode_length}")
            print(f"  Updates per step: {algo_config.num_epochs * (algo_config.num_rollouts * algo_config.episode_length // minibatch_size)}")
        else:
            print(f"  Num epochs: {algo_config.num_epochs}")
            print(f"  Updates per step: {algo_config.num_epochs}")
    else:
        if hasattr(algo_config, 'num_updates_per_step'):
            print(f"Updates per step: {algo_config.num_updates_per_step}")
    
    # GAE information
    if hasattr(algo_config, 'use_gae') and algo_config.use_gae:
        print(f"Advantage estimation: GAE (λ={algo_config.gae_lambda})")
    else:
        print(f"Advantage estimation: Standard returns")
    
    print(f"Rollouts per step: {algo_config.num_rollouts}")
    print(f"Episode length: {algo_config.episode_length}")
    print(f"Learning rate: {agent.config.lr}")
    print(f"Hidden dim: {agent.config.hidden_dim}")
    print(f"Gamma (discount): {agent.config.gamma}")
    print(f"Non-linearity: {agent.config.nl}")
    print(f"Seed: {agent.config.seed}")
    print(f"Checkpoint interval: {exp_config.checkpoint_interval}")
    print(f"Evaluation episodes: {exp_config.num_eval_episodes}")
    print(f"Run eval on checkpoint: {exp_config.run_eval_on_checkpoint}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Parse arbitrary keyword arguments
    args = parse_args()
    
    # Create experiment config with overrides
    exp_config = create_config_from_args(ExperimentConfig, args)
    
    # Determine algorithm
    algorithm = get_algorithm_from_args_or_resume(
        args, exp_config.experiments_root)
    algorithm = algorithm.lower()
    
    # Get algorithm config class
    AlgoConfigClass = ALGORITHM_CONFIG_MAP[algorithm]
    
    # Create algorithm config with overrides
    algo_config = create_config_from_args(AlgoConfigClass, args)
    algo_config.algorithm = algorithm  # Ensure algorithm name is set
    
    # Validate algorithm
    validate_config(algo_config)
    
    # Get algorithm class
    AlgorithmClass = get_algorithm_class(algo_config.algorithm)
    
    print("="*60)
    print(f"{algo_config.algorithm} Training")
    print("="*60)
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(
        experiments_root=exp_config.experiments_root,
        run_id=exp_config.run_id
    )
    
    # Resume from checkpoint if specified
    if exp_config.resume_run_id is not None:
        print(f"Resuming from run: {exp_config.resume_run_id}")
        
        # Load config from the run
        run_config_dict = ExperimentManager.load_config(
            exp_config.resume_run_id,
            experiments_root=exp_config.experiments_root
        )
        run_config = SimpleNamespace(**run_config_dict)
        
        # Create experiment manager for the resume run
        resume_exp_manager = ExperimentManager(
            experiments_root=exp_config.experiments_root,
            run_id=exp_config.resume_run_id
        )
        
        # Determine which step to load from
        if exp_config.resume_step is None:
            # Load from latest checkpoint
            available_steps = resume_exp_manager.list_checkpoints()
            if available_steps:
                exp_config.resume_step = available_steps[-1]
            else:
                print("⚠️  No checkpoints found in resume run. Starting fresh.")
                exp_config.resume_step = None
        
        if exp_config.resume_step is not None:
            # Load agent from checkpoint
            agent, training_stats, loaded_step = AlgorithmClass.load_checkpoint(
                run_id=exp_config.resume_run_id,
                step=exp_config.resume_step,
                experiment_manager_class=ExperimentManager,
                experiments_root=exp_config.experiments_root
            )
            
            # Update agent's experiment manager to the new run (if creating new run)
            if exp_config.run_id is None:
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
        # Pass the algorithm config to the algorithm
        agent = AlgorithmClass(algo_config, experiment_manager=exp_manager)
        training_stats = {
            'loss_history': [],
            'grad_norm_history': [],
            'return_history': [],
            'eval_history': [],
        }
        start_step = 0
        
        # Save config for this run (merge both configs for compatibility)
        from dataclasses import asdict
        merged_config = {**asdict(exp_config), **asdict(algo_config)}
        exp_manager.save_config(SimpleNamespace(**merged_config))
    
    # Print run information
    exp_manager.print_run_info()
    print_config_info(exp_config, algo_config, agent)
    
    print(f"Training steps: {start_step} → {algo_config.num_training_steps}\n")
    
    # Train the agent with checkpointing and evaluation
    training_stats_new = agent.train(
        key=None,
        num_steps=algo_config.num_training_steps - start_step,
        checkpoint_interval=exp_config.checkpoint_interval,
        keep_only_latest=exp_config.keep_only_latest,
        num_eval_episodes=exp_config.num_eval_episodes,
        max_eval_length=exp_config.max_eval_length,
        run_eval_on_checkpoint=exp_config.run_eval_on_checkpoint,
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
        algo_config.num_training_steps,
        training_stats,
        keep_only_latest=exp_config.keep_only_latest
    )
    
    # Print summary using utility function
    print_training_summary(training_stats, algorithm_name=algo_config.algorithm)
    
    # Create and save plot using utility function
    plot_path = exp_manager.get_plot_path(algo_config.algorithm)
    save_training_plots(training_stats, plot_path, algorithm_name=algo_config.algorithm)
    
    print(f"\n✅ Training complete!")
    print(f"Algorithm: {algo_config.algorithm}")
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