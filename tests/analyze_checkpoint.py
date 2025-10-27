"""
Checkpoint Analysis Script

This script allows you to analyze training checkpoints and generate plots
without stopping the training process. You can check progress intermittently
by pointing this script at your checkpoint directory.

Usage:
    python analyze_checkpoints.py --checkpoint-path checkpoints/checkpoint_step_10.pkl
    python analyze_checkpoints.py --checkpoint-dir checkpoints --latest
"""

try:
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
    
    from src.utils import (
        print_training_summary,
        save_training_plots
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


def list_checkpoints(checkpoint_dir: Path):
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
    
    print(f"\n📁 Available checkpoints in {checkpoint_dir}:")
    print("=" * 60)
    for ckpt in checkpoint_files:
        step = int(ckpt.stem.split('_')[-1])
        print(f"  Step {step:5d}: {ckpt.name}")
    print("=" * 60 + "\n")


@dataclass
class AnalysisConfig:
    checkpoint_path: Optional[str] = None  # Specific checkpoint file to analyze
    checkpoint_dir: str = 'checkpoints'  # Directory containing checkpoints
    latest: bool = True  # Use the latest checkpoint in the directory
    list_checkpoints: bool = False  # List all available checkpoints
    plot_path: str = 'analysis_plots/checkpoint_analysis.png'  # Where to save plots
    algorithm_name: str = 'REINFORCE'  # Name of the algorithm for plot titles


def main():
    config = tyro.cli(AnalysisConfig)
    
    checkpoint_dir = Path(config.checkpoint_dir)
    
    # List checkpoints if requested
    if config.list_checkpoints:
        list_checkpoints(checkpoint_dir)
        return
    
    # Determine which checkpoint to analyze
    if config.checkpoint_path is not None:
        checkpoint_path = Path(config.checkpoint_path)
    elif config.latest:
        print(f"🔍 Finding latest checkpoint in {checkpoint_dir}...")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        print(f"📂 Using checkpoint: {checkpoint_path}")
    else:
        print("❌ Error: Must specify either --checkpoint-path or --latest")
        print("   Or use --list-checkpoints to see available checkpoints")
        return
    
    # Load checkpoint data
    print("\n" + "=" * 60)
    print("CHECKPOINT ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    
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
    
    # Create and save plots
    plot_path = Path(config.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_training_plots(
        training_stats, 
        str(plot_path), 
        algorithm_name=f"{config.algorithm_name} (Step {step})"
    )
    
    print(f"\n✅ Analysis complete! Plots saved to: {plot_path}")


if __name__ == "__main__":
    main()