"""
Checkpoint Analysis Script

This script allows you to analyze training checkpoints and generate plots
without stopping the training process. Works with the experiment manager
to organize runs and their outputs.

Usage:
    # Analyze latest checkpoint from a run
    python analyze_checkpoint.py --run-id run_20241103_143045_a1b2c3d4
    
    # Analyze specific step from a run
    python analyze_checkpoint.py --run-id run_20241103_143045_a1b2c3d4 --step 50
    
    # List all available runs
    python analyze_checkpoint.py --list-runs
    
    # List checkpoints in a run
    python analyze_checkpoint.py --run-id run_20241103_143045_a1b2c3d4 --list-checkpoints
"""

try:
    from src.utils import (
        print_training_summary,
        save_training_plots,
        ExperimentManager,
        plot_eval_progress,
    )
except:
    from pathlib import Path
    import sys

    # Define repository root for imports
    _THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = _THIS_FILE.parent.parent
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    from src.utils import (
        print_training_summary,
        save_training_plots,
        ExperimentManager,
        plot_eval_progress,
    )

from dataclasses import dataclass
import tyro
from pathlib import Path
import pickle
from typing import Optional


def load_checkpoint_stats(checkpoint_path: Path):
    """
    Load training statistics from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        training_stats: Dictionary of training statistics
        step: Current training step
        config: Training configuration (as dict)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    training_stats = checkpoint['training_stats']
    step = checkpoint['step']
    config = checkpoint['config']
    
    # Config is now stored as a dict, so it will load without issues
    return training_stats, step, config


def print_config_summary(config):
    """
    Pretty print the configuration.
    
    Args:
        config: Configuration dict or object
    """
    # Convert to dict if it's an object
    if isinstance(config, dict):
        config_dict = config
    elif hasattr(config, '__dict__'):
        config_dict = vars(config)
    elif hasattr(config, '__dataclass_fields__'):
        import dataclasses
        config_dict = dataclasses.asdict(config)
    else:
        print(f"Config: {config}")
        return
    
    print("\n📋 Configuration:")
    print("-" * 40)
    for key, value in sorted(config_dict.items()):
        print(f"  {key:25s}: {value}")
    print("-" * 40)


def find_latest_checkpoint(checkpoint_dir: Path):
    """
    Find the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint file
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pkl"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
    
    # Sort by step number (extract from filename)
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    return checkpoint_files[-1]


def list_checkpoints_in_dir(checkpoint_dir: Path):
    """
    List all available checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_step_*.pkl"))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in: {checkpoint_dir}")
        return
    
    print(f"\n📂 Available checkpoints in {checkpoint_dir}:")
    print("=" * 60)
    for ckpt in checkpoint_files:
        step = int(ckpt.stem.split('_')[-1])
        print(f"  Step {step:5d}: {ckpt.name}")
    print("=" * 60 + "\n")


@dataclass
class AnalysisConfig:
    experiments_root: str = 'experiments'  # Root directory for experiments
    run_id: Optional[str] = None  # Run ID to analyze
    step: Optional[int] = None  # Specific step to analyze (latest if None)
    
    list_runs: bool = False  # List all available runs
    list_checkpoints: bool = False  # List checkpoints in the run
    algorithm_name: str = 'ppo'  # Name of the algorithm for plot titles


def main():
    config = tyro.cli(AnalysisConfig)
    
    # List all runs if requested
    if config.list_runs:
        print("\n" + "=" * 60)
        print("AVAILABLE RUNS")
        print("=" * 60)
        runs = ExperimentManager.list_runs(config.experiments_root)
        if not runs:
            print("❌ No runs found in experiments directory")
            return
        
        for run_id in runs:
            exp_manager = ExperimentManager(config.experiments_root, run_id)
            checkpoints = exp_manager.list_checkpoints()
            print(f"  {run_id}")
            print(f"    Latest step: {checkpoints[-1] if checkpoints else 'None'}")
            print(f"    Total checkpoints: {len(checkpoints)}")
        print("=" * 60 + "\n")
        return
    
    # Check that run_id is provided
    if config.run_id is None:
        print("❌ Error: Must specify --run-id")
        print("   Or use --list-runs to see available runs")
        return
    
    # Initialize experiment manager for this run
    try:
        exp_manager = ExperimentManager(config.experiments_root, config.run_id)
    except Exception as e:
        print(f"❌ Error initializing experiment manager: {e}")
        return
    
    # List checkpoints in this run if requested
    if config.list_checkpoints:
        print(f"\n📂 Checkpoints in run '{config.run_id}':")
        print("=" * 60)
        checkpoints = exp_manager.list_checkpoints()
        if not checkpoints:
            print("❌ No checkpoints found in this run")
        else:
            for step in checkpoints:
                print(f"  Step {step:5d}")
        print("=" * 60 + "\n")
        return
    
    # Determine which checkpoint to analyze
    available_steps = exp_manager.list_checkpoints()
    
    if not available_steps:
        print(f"❌ No checkpoints found in run: {config.run_id}")
        return
    
    if config.step is not None:
        if config.step not in available_steps:
            print(f"❌ Step {config.step} not found in run")
            print(f"   Available steps: {available_steps}")
            return
        checkpoint_step = config.step
    else:
        checkpoint_step = available_steps[-1]
    
    checkpoint_path = exp_manager.get_checkpoint_path(checkpoint_step)
    
    # Load checkpoint data
    print("\n" + "=" * 60)
    print("CHECKPOINT ANALYSIS")
    print("=" * 60)
    print(f"Run ID: {config.run_id}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    try:
        training_stats, step, checkpoint_config = load_checkpoint_stats(checkpoint_path)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    print(f"Training step: {step}")
    
    # Print configuration
    if isinstance(checkpoint_config, dict):
        print(f"Environment: {checkpoint_config.get('env_name', 'unknown')}")
        print(f"Learning rate: {checkpoint_config.get('lr', 'unknown')}")
        print(f"Hidden dim: {checkpoint_config.get('hidden_dim', 'unknown')}")
    else:
        # Fallback for old checkpoints
        print(f"Environment: {getattr(checkpoint_config, 'env_name', 'unknown')}")
        print(f"Learning rate: {getattr(checkpoint_config, 'lr', 'unknown')}")
        print(f"Hidden dim: {getattr(checkpoint_config, 'hidden_dim', 'unknown')}")
    
    # Print full config summary
    print_config_summary(checkpoint_config)
    
    print("=" * 60 + "\n")
    
    # Print training summary
    print_training_summary(training_stats, algorithm_name=config.algorithm_name)
    
    # Create and save plots in the run's plots directory
    plot_path = exp_manager.get_plot_path(config.algorithm_name)
    
    save_training_plots(
        training_stats, 
        str(plot_path), 
        algorithm_name=f"{config.algorithm_name} (Step {step})"
    )
    
    # If we have evaluation data, also create a focused evaluation plot
    if 'eval_history' in training_stats and training_stats['eval_history']:
        eval_plot_path = exp_manager.plots_dir / f"eval_progress_{config.algorithm_name.lower()}.png"
        
        plot_eval_progress(
            training_stats,
            str(eval_plot_path),
            algorithm_name=f"{config.algorithm_name} (Step {step})"
        )
    
    print(f"\n✅ Analysis complete! Plots saved to: {plot_path}")


if __name__ == "__main__":
    main()