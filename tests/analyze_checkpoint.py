"""
Checkpoint Analysis Script

This script allows you to analyze training checkpoints and generate plots
without stopping the training process. Works with the experiment manager
to organize runs and their outputs. Supports both single-run and multi-run
comparison analysis.

Usage:
    # Analyze latest checkpoint from a run
    python analyze_checkpoint.py --run-ids run_20241103_143045_a1b2c3d4
    
    # Analyze specific step from a run
    python analyze_checkpoint.py --run-ids run_20241103_143045_a1b2c3d4 --step 50
    
    # Compare multiple runs (generates comparison plot)
    python analyze_checkpoint.py --run-ids run_20241103_143045_a1b2c3d4 run_20241103_150000_b2c3d4e5
    
    # List all available runs
    python analyze_checkpoint.py --list-runs
    
    # List checkpoints in a run
    python analyze_checkpoint.py --run-ids run_20241103_143045_a1b2c3d4 --list-checkpoints
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
from typing import Optional, List
import hashlib


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
    
    print("\n🔋 Configuration:")
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


def extract_experiment_id(run_id: str) -> Optional[str]:
    """
    Extract experiment_id from run_id if present.
    Run IDs typically follow format: run_YYYYMMDD_HHMMSS_experiment_id
    
    Args:
        run_id: Run identifier string
        
    Returns:
        experiment_id if found, None otherwise
    """
    parts = run_id.split('_')
    if len(parts) >= 4:
        # Assume format: run_date_time_experiment_id
        return '_'.join(parts[3:])
    return None


def get_run_label(run_id: str, experiment_id: Optional[str] = None) -> str:
    """
    Get a human-readable label for a run.
    
    Args:
        run_id: Run identifier
        experiment_id: Experiment identifier (optional)
        
    Returns:
        Label string preferring experiment_id if available
    """
    if experiment_id:
        return experiment_id
    return run_id


def generate_comparison_suffix(run_ids: List[str]) -> str:
    """
    Generate a concise suffix for comparison plot filenames.
    Uses experiment_ids if available, otherwise creates a hash of run_ids.
    
    Args:
        run_ids: List of run identifiers
        
    Returns:
        String suffix for filename
    """
    # Try to extract experiment IDs
    exp_ids = [extract_experiment_id(rid) for rid in run_ids]
    
    if all(exp_ids):
        # All have experiment IDs - concatenate them
        return '_vs_'.join(exp_ids)
    else:
        # Fall back to hashing the run_ids for a concise suffix
        combined = '_'.join(run_ids)
        hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"comparison_{hash_suffix}"


@dataclass
class AnalysisConfig:
    experiments_root: str = 'experiments'  # Root directory for experiments
    run_ids: Optional[List[str]] = None  # List of run IDs to analyze
    step: Optional[int] = None  # Specific step to analyze (latest if None)
    
    list_runs: bool = False  # List all available runs
    list_checkpoints: bool = False  # List checkpoints in the run
    algorithm_name: str = 'ppo'  # Name of the algorithm for plot titles


def analyze_single_run(config: AnalysisConfig, run_id: str):
    """
    Analyze a single run and generate plots.
    
    Args:
        config: Analysis configuration
        run_id: Run ID to analyze
    """
    # Initialize experiment manager for this run
    try:
        exp_manager = ExperimentManager(config.experiments_root, run_id)
    except Exception as e:
        print(f"❌ Error initializing experiment manager: {e}")
        return
    
    # List checkpoints in this run if requested
    if config.list_checkpoints:
        print(f"\n📂 Checkpoints in run '{run_id}':")
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
        print(f"❌ No checkpoints found in run: {run_id}")
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
    print(f"Run ID: {run_id}")
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


def analyze_multiple_runs(config: AnalysisConfig, run_ids: List[str]):
    """
    Analyze multiple runs and generate a comparison plot.
    
    Args:
        config: Analysis configuration
        run_ids: List of run IDs to compare
    """
    print("\n" + "=" * 60)
    print("MULTI-RUN COMPARISON ANALYSIS")
    print("=" * 60)
    print(f"Comparing {len(run_ids)} runs:")
    for rid in run_ids:
        print(f"  - {rid}")
    
    # Load data from all runs
    all_stats = {}
    all_steps = {}
    run_labels = {}
    
    for run_id in run_ids:
        try:
            exp_manager = ExperimentManager(config.experiments_root, run_id)
        except Exception as e:
            print(f"❌ Error initializing experiment manager for {run_id}: {e}")
            continue
        
        available_steps = exp_manager.list_checkpoints()
        
        if not available_steps:
            print(f"❌ No checkpoints found in run: {run_id}")
            continue
        
        if config.step is not None:
            if config.step not in available_steps:
                print(f"⚠️  Step {config.step} not found in run {run_id}")
                print(f"   Available steps: {available_steps}")
                continue
            checkpoint_step = config.step
        else:
            checkpoint_step = available_steps[-1]
        
        checkpoint_path = exp_manager.get_checkpoint_path(checkpoint_step)
        
        try:
            training_stats, step, checkpoint_config = load_checkpoint_stats(checkpoint_path)
            all_stats[run_id] = training_stats
            all_steps[run_id] = step
            
            # Extract and use experiment_id for labeling
            exp_id = extract_experiment_id(run_id)
            run_labels[run_id] = get_run_label(run_id, exp_id)
            
            print(f"✓ Loaded {run_id} (Step {step})")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint for {run_id}: {e}")
            continue
    
    if not all_stats:
        print("\n❌ No valid runs loaded for comparison")
        return
    
    print(f"\n✓ Successfully loaded {len(all_stats)} run(s)")
    
    # Determine output path for comparison plot
    # Create a comparisons directory alongside the run directories
    exp_manager = ExperimentManager(config.experiments_root, run_ids[0])
    
    comparisons_dir = exp_manager.experiments_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison suffix
    suffix = generate_comparison_suffix(run_ids)
    comparison_plot_name = f"comparison_{suffix}_{config.algorithm_name.lower()}.png"
    comparison_plot_path = comparisons_dir / comparison_plot_name
    
    # Save comparison plots using existing plotting utilities
    # Note: The save_training_plots function would need to be adapted to handle multiple datasets
    # For now, we create a simple comparison by calling it with labeled data
    
    print(f"\n📊 Generating comparison plot...")
    
    # Create a merged stats dictionary with labels for plotting
    # This assumes save_training_plots or a variant can handle multiple series
    try:
        # Try to use matplotlib directly if needed for custom comparison plotting
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{config.algorithm_name} Comparison: {suffix}", fontsize=14, fontweight='bold')
        
        # Plot rewards if available
        if 'rewards' in next(iter(all_stats.values())):
            ax = axes[0, 0]
            for run_id, stats in all_stats.items():
                if 'rewards' in stats:
                    ax.plot(stats['rewards'], label=run_labels[run_id], alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Rewards')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot losses if available
        if 'losses' in next(iter(all_stats.values())):
            ax = axes[0, 1]
            for run_id, stats in all_stats.items():
                if 'losses' in stats:
                    ax.plot(stats['losses'], label=run_labels[run_id], alpha=0.7)
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot policy loss if available
        if 'policy_losses' in next(iter(all_stats.values())):
            ax = axes[1, 0]
            for run_id, stats in all_stats.items():
                if 'policy_losses' in stats:
                    ax.plot(stats['policy_losses'], label=run_labels[run_id], alpha=0.7)
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Policy Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot value loss if available
        if 'value_losses' in next(iter(all_stats.values())):
            ax = axes[1, 1]
            for run_id, stats in all_stats.items():
                if 'value_losses' in stats:
                    ax.plot(stats['value_losses'], label=run_labels[run_id], alpha=0.7)
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Value Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(comparison_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison plot saved to: {comparison_plot_path}")
        
    except ImportError:
        print("⚠️  matplotlib not available, skipping comparison plot generation")
    except Exception as e:
        print(f"❌ Error generating comparison plot: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for run_id, stats in all_stats.items():
        print(f"\n{run_labels[run_id]} (Step {all_steps[run_id]}):")
        if 'rewards' in stats and stats['rewards']:
            print(f"  Final reward: {stats['rewards'][-1]:.4f}")
        if 'losses' in stats and stats['losses']:
            print(f"  Final loss: {stats['losses'][-1]:.4f}")
    print("=" * 60 + "\n")


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
    
    # Check that run_ids are provided
    if config.run_ids is None or len(config.run_ids) == 0:
        print("❌ Error: Must specify --run-ids (one or more run IDs)")
        print("   Or use --list-runs to see available runs")
        return
    
    # Handle single run vs. multiple runs
    if len(config.run_ids) == 1:
        # Single run: use original behavior
        analyze_single_run(config, config.run_ids[0])
    else:
        # Multiple runs: generate comparison plots
        analyze_multiple_runs(config, config.run_ids)


if __name__ == "__main__":
    main()