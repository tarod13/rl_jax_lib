"""
Enhanced plotting utilities for training and evaluation metrics.

This module extends the standard plotting to include evaluation returns
alongside training metrics for comprehensive analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def save_training_plots_with_eval(
    training_stats: Dict[str, Any],
    save_path: str,
    algorithm_name: str = "Algorithm",
    figsize: tuple = (15, 10),
    dpi: int = 150,
):
    """
    Create and save comprehensive training plots including evaluation metrics.
    
    Args:
        training_stats: Dictionary containing:
            - 'return_history': Training returns (array of shape [steps, num_rollouts])
            - 'loss_history': Training losses
            - 'grad_norm_history': Gradient norms
            - 'eval_history': List of evaluation results (optional)
        save_path: Path to save the plot
        algorithm_name: Name for plot title
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
    """
    # Check if we have evaluation data
    has_eval = 'eval_history' in training_stats and training_stats['eval_history']
    
    # Determine number of subplots
    num_plots = 4 if has_eval else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize)
    
    # Extract training data
    returns = np.array(training_stats['return_history'])
    losses = np.array(training_stats['loss_history'])
    grad_norms = np.array(training_stats['grad_norm_history'])
    
    # Training steps
    train_steps = np.arange(1, len(returns) + 1)
    
    # ============================================================
    # Plot 1: Training Returns (from rollouts)
    # ============================================================
    ax = axes[0]
    
    # Compute mean and std of returns at each step
    mean_returns = np.mean(returns, axis=1)
    std_returns = np.std(returns, axis=1)
    
    ax.plot(train_steps, mean_returns, label='Mean Return', color='blue', linewidth=2)
    ax.fill_between(
        train_steps,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.3,
        color='blue',
        label='±1 Std Dev'
    )
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Return (Discounted)')
    ax.set_title(f'{algorithm_name} - Training Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 2: Evaluation Returns (if available)
    # ============================================================
    if has_eval:
        ax = axes[1]
        
        # Extract evaluation data
        eval_steps = [e['step'] for e in training_stats['eval_history']]
        eval_mean_returns = [e['stats']['mean_return'] for e in training_stats['eval_history']]
        eval_std_returns = [e['stats']['std_return'] for e in training_stats['eval_history']]
        
        eval_mean_returns = np.array(eval_mean_returns)
        eval_std_returns = np.array(eval_std_returns)
        
        # Plot evaluation returns with error bars
        ax.errorbar(
            eval_steps,
            eval_mean_returns,
            yerr=eval_std_returns,
            marker='o',
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
            color='red',
            label='Evaluation Return (Undiscounted)'
        )
        
        # Also plot training returns in background for comparison
        ax.plot(
            train_steps,
            mean_returns,
            alpha=0.3,
            color='blue',
            linewidth=1,
            linestyle='--',
            label='Training Return (Discounted)'
        )
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Return')
        ax.set_title(f'{algorithm_name} - Evaluation Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotate with final evaluation return
        if eval_steps:
            final_return = eval_mean_returns[-1]
            final_std = eval_std_returns[-1]
            ax.text(
                0.98, 0.98,
                f'Final: {final_return:.1f} ± {final_std:.1f}',
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10
            )
    
    # ============================================================
    # Plot 3: Training Loss
    # ============================================================
    ax = axes[2] if has_eval else axes[1]
    
    # Compute mean loss per step (across updates)
    mean_losses = np.mean(losses, axis=1)
    
    ax.plot(train_steps, mean_losses, color='orange', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{algorithm_name} - Training Loss')
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 4: Gradient Norms
    # ============================================================
    ax = axes[3] if has_eval else axes[2]
    
    # Compute mean gradient norm per step
    mean_grad_norms = np.mean(grad_norms, axis=1)
    
    ax.plot(train_steps, mean_grad_norms, color='green', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(f'{algorithm_name} - Gradient Norms')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # ============================================================
    # Save figure
    # ============================================================
    plt.tight_layout()
    
    # Ensure directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to: {save_path}")


def print_training_summary_with_eval(
    training_stats: Dict[str, Any],
    algorithm_name: str = "Algorithm"
):
    """
    Print comprehensive training summary including evaluation metrics.
    
    Args:
        training_stats: Dictionary with training and evaluation data
        algorithm_name: Name of the algorithm
    """
    print("\n" + "=" * 60)
    print(f"{algorithm_name.upper()} TRAINING SUMMARY")
    print("=" * 60)
    
    # Training statistics
    returns = np.array(training_stats['return_history'])
    losses = np.array(training_stats['loss_history'])
    grad_norms = np.array(training_stats['grad_norm_history'])
    
    num_steps = len(returns)
    
    print(f"\nTraining Steps: {num_steps}")
    print(f"\nTraining Returns (Discounted):")
    print(f"  Final mean:  {np.mean(returns[-1]):8.2f} ± {np.std(returns[-1]):6.2f}")
    print(f"  Overall mean: {np.mean(returns):8.2f}")
    print(f"  Min:         {np.min(returns):8.2f}")
    print(f"  Max:         {np.max(returns):8.2f}")
    
    print(f"\nTraining Loss:")
    print(f"  Final:       {np.mean(losses[-1]):8.4f}")
    print(f"  Mean:        {np.mean(losses):8.4f}")
    print(f"  Min:         {np.min(losses):8.4f}")
    print(f"  Max:         {np.max(losses):8.4f}")
    
    print(f"\nGradient Norms:")
    print(f"  Final:       {np.mean(grad_norms[-1]):8.4e}")
    print(f"  Mean:        {np.mean(grad_norms):8.4e}")
    print(f"  Min:         {np.min(grad_norms):8.4e}")
    print(f"  Max:         {np.max(grad_norms):8.4e}")
    
    # Evaluation statistics (if available)
    if 'eval_history' in training_stats and training_stats['eval_history']:
        print("\n" + "-" * 60)
        print("EVALUATION SUMMARY")
        print("-" * 60)
        
        eval_history = training_stats['eval_history']
        print(f"\nNumber of evaluations: {len(eval_history)}")
        
        # Show all evaluation results
        print(f"\nEvaluation Returns (Undiscounted):")
        for eval_data in eval_history:
            step = eval_data['step']
            stats = eval_data['stats']
            mean_ret = stats['mean_return']
            std_ret = stats['std_return']
            mean_len = stats['mean_length']
            print(f"  Step {step:4d}: {mean_ret:8.2f} ± {std_ret:6.2f}  "
                  f"(length: {mean_len:6.1f})")
        
        # Summary statistics
        all_eval_returns = [e['stats']['mean_return'] for e in eval_history]
        print(f"\nEvaluation Statistics:")
        print(f"  Best:        {np.max(all_eval_returns):8.2f}")
        print(f"  Final:       {all_eval_returns[-1]:8.2f}")
        print(f"  Mean:        {np.mean(all_eval_returns):8.2f}")
        print(f"  Improvement: {all_eval_returns[-1] - all_eval_returns[0]:8.2f}")
    
    print("\n" + "=" * 60 + "\n")


def plot_eval_progress(
    training_stats: Dict[str, Any],
    save_path: Optional[str] = None,
    algorithm_name: str = "Algorithm",
    figsize: tuple = (10, 6),
    dpi: int = 150,
):
    """
    Create a focused plot showing evaluation progress over training.
    
    This creates a single, publication-quality plot of evaluation returns.
    
    Args:
        training_stats: Dictionary containing eval_history
        save_path: Path to save plot (if None, just display)
        algorithm_name: Name for title
        figsize: Figure size
        dpi: Resolution
    """
    if 'eval_history' not in training_stats or not training_stats['eval_history']:
        print("⚠️  No evaluation data available for plotting")
        return
    
    eval_history = training_stats['eval_history']
    
    # Extract data
    eval_steps = [e['step'] for e in eval_history]
    eval_returns = [e['stats']['mean_return'] for e in eval_history]
    eval_stds = [e['stats']['std_return'] for e in eval_history]
    
    eval_returns = np.array(eval_returns)
    eval_stds = np.array(eval_stds)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with error bars
    ax.errorbar(
        eval_steps,
        eval_returns,
        yerr=eval_stds,
        marker='o',
        markersize=10,
        capsize=5,
        capthick=2,
        linewidth=2.5,
        color='darkblue',
        ecolor='skyblue',
        label='Evaluation Return'
    )
    
    # Add trend line
    if len(eval_steps) > 2:
        z = np.polyfit(eval_steps, eval_returns, 1)
        p = np.poly1d(z)
        ax.plot(
            eval_steps,
            p(eval_steps),
            linestyle='--',
            color='red',
            alpha=0.5,
            linewidth=2,
            label=f'Trend (slope: {z[0]:.2f})'
        )
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Evaluation Return (Undiscounted)', fontsize=12)
    ax.set_title(f'{algorithm_name} - Evaluation Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(eval_returns) > 1:
        improvement = eval_returns[-1] - eval_returns[0]
        pct_improvement = (improvement / abs(eval_returns[0])) * 100
        ax.text(
            0.02, 0.98,
            f'Total Improvement: {improvement:+.1f} ({pct_improvement:+.1f}%)',
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"📊 Evaluation plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


# Backward compatibility: keep old function names
def save_training_plots(training_stats, save_path, algorithm_name="Algorithm"):
    """
    Backward compatible wrapper that uses the enhanced plotting.
    
    Automatically uses the new plotting if eval_history is present,
    otherwise falls back to standard plotting.
    """
    save_training_plots_with_eval(training_stats, save_path, algorithm_name)


def print_training_summary(training_stats, algorithm_name="Algorithm"):
    """
    Backward compatible wrapper for training summary.
    """
    print_training_summary_with_eval(training_stats, algorithm_name)