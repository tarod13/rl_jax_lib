# rl_lib/utils/training_utils.py
"""
Reusable utilities for training, plotting, and analyzing RL algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_training_stats(training_stats, algorithm_name="RL"):
    """
    Create training statistics plots.
    
    Args:
        training_stats: Dictionary containing 'loss_history', 'grad_norm_history', 
                       and optionally 'return_history'
        algorithm_name: Name of the algorithm for plot titles
        
    Returns:
        fig: matplotlib figure object
    """
    loss_history = np.array(training_stats['loss_history'])  # Shape: (num_steps, num_updates)
    grad_norm_history = np.array(training_stats['grad_norm_history'])
    
    num_steps, num_updates = loss_history.shape
    
    # Check if returns are available
    has_returns = 'return_history' in training_stats and training_stats['return_history'] is not None
    if has_returns:
        return_history = np.array(training_stats['return_history'])  # Shape: (num_steps, num_rollouts)
        n_rows = 3
    else:
        n_rows = 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    fig.suptitle(f'{algorithm_name} Training Progress', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    # ========== Plot 1: Loss over all updates ==========
    ax1 = axes[0]
    
    # Flatten to show all updates
    all_losses = loss_history.flatten()
    update_indices = np.arange(len(all_losses))
    
    ax1.plot(update_indices, all_losses, alpha=0.6, linewidth=0.8, color='steelblue')
    
    # Add moving average for smoothing
    window_size = min(50, len(all_losses) // 10)
    if window_size > 1:
        moving_avg = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(np.arange(window_size-1, len(all_losses)), moving_avg, 
                color='darkblue', linewidth=2, label=f'Moving Avg (window={window_size})')
        ax1.legend()
    
    ax1.set_xlabel('Update Number', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Over All Updates', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ========== Plot 2: Average loss per training step ==========
    ax2 = axes[1]
    
    # Average loss across updates for each step
    avg_loss_per_step = loss_history.mean(axis=1)
    step_indices = np.arange(num_steps)
    
    ax2.plot(step_indices, avg_loss_per_step, marker='o', markersize=4, 
            linewidth=2, color='coral', label='Average Loss')
    
    # Add error bars showing variance
    std_loss_per_step = loss_history.std(axis=1)
    ax2.fill_between(step_indices, 
                     avg_loss_per_step - std_loss_per_step,
                     avg_loss_per_step + std_loss_per_step,
                     alpha=0.3, color='coral', label='±1 Std Dev')
    
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('Average Loss', fontsize=11)
    ax2.set_title('Average Loss Per Training Step', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== Plot 3: Gradient norm over all updates ==========
    ax3 = axes[2]
    
    # Flatten gradient norms
    all_grad_norms = grad_norm_history.flatten()
    
    ax3.plot(update_indices, all_grad_norms, alpha=0.6, linewidth=0.8, color='seagreen')
    
    # Add moving average
    if window_size > 1:
        moving_avg_grad = np.convolve(all_grad_norms, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(np.arange(window_size-1, len(all_grad_norms)), moving_avg_grad,
                color='darkgreen', linewidth=2, label=f'Moving Avg (window={window_size})')
        ax3.legend()
    
    ax3.set_xlabel('Update Number', fontsize=11)
    ax3.set_ylabel('Gradient Norm', fontsize=11)
    ax3.set_title('Gradient Norm Over All Updates', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale often better for gradient norms
    
    # ========== Plot 4: Average gradient norm per training step ==========
    ax4 = axes[3]
    
    # Average gradient norm across updates for each step
    avg_grad_per_step = grad_norm_history.mean(axis=1)
    
    ax4.plot(step_indices, avg_grad_per_step, marker='s', markersize=4,
            linewidth=2, color='mediumpurple', label='Average Grad Norm')
    
    # Add error bars
    std_grad_per_step = grad_norm_history.std(axis=1)
    ax4.fill_between(step_indices,
                     avg_grad_per_step - std_grad_per_step,
                     avg_grad_per_step + std_grad_per_step,
                     alpha=0.3, color='mediumpurple', label='±1 Std Dev')
    
    ax4.set_xlabel('Training Step', fontsize=11)
    ax4.set_ylabel('Average Gradient Norm', fontsize=11)
    ax4.set_title('Average Gradient Norm Per Training Step', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # ========== Plot 5 & 6: Returns (if available) ==========
    if has_returns:
        num_rollouts = return_history.shape[1] if len(return_history.shape) > 1 else 1
        
        # Plot 5: Returns over training steps (individual rollouts)
        ax5 = axes[4]
        
        # Plot returns for each rollout with transparency
        num_plot_rollouts = min(num_rollouts, 10)  # Limit to 10 rollouts for readability
        for rollout_idx in range(num_plot_rollouts):
            ax5.plot(step_indices, return_history[:, rollout_idx], 
                    alpha=0.3, linewidth=0.8)
        
        # Plot mean returns with bold line
        mean_returns = return_history.mean(axis=1)
        ax5.plot(step_indices, mean_returns, 
                color='darkred', linewidth=2.5, label='Mean Returns', zorder=10)
        
        ax5.set_xlabel('Training Step', fontsize=11)
        ax5.set_ylabel('Returns', fontsize=11)
        ax5.set_title('Episode Returns Over Training', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Returns statistics per step
        ax6 = axes[5]
        
        # Mean returns with error bars
        std_returns = return_history.std(axis=1)
        ax6.plot(step_indices, mean_returns, marker='o', markersize=4,
                linewidth=2, color='darkred', label='Mean Returns')
        ax6.fill_between(step_indices,
                        mean_returns - std_returns,
                        mean_returns + std_returns,
                        alpha=0.3, color='darkred', label='±1 Std Dev')
        
        # Also plot min and max
        min_returns = return_history.min(axis=1)
        max_returns = return_history.max(axis=1)
        ax6.plot(step_indices, min_returns, '--', linewidth=1, 
                color='darkred', alpha=0.5, label='Min/Max')
        ax6.plot(step_indices, max_returns, '--', linewidth=1, 
                color='darkred', alpha=0.5)
        
        ax6.set_xlabel('Training Step', fontsize=11)
        ax6.set_ylabel('Returns', fontsize=11)
        ax6.set_title('Returns Statistics Per Training Step', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def print_training_summary(training_stats, algorithm_name="RL"):
    """
    Print summary statistics of training.
    
    Args:
        training_stats: Dictionary containing 'loss_history', 'grad_norm_history',
                       and optionally 'return_history'
        algorithm_name: Name of the algorithm for the summary
    """
    loss_history = np.array(training_stats['loss_history'])
    grad_norm_history = np.array(training_stats['grad_norm_history'])
    
    num_steps, num_updates = loss_history.shape
    
    print("\n" + "="*60)
    print(f"{algorithm_name} TRAINING SUMMARY")
    print("="*60)
    print(f"Total training steps: {num_steps}")
    print(f"Updates per step: {num_updates}")
    print(f"Total updates: {num_steps * num_updates}")
    print(f"\nLoss Statistics:")
    print(f"  Initial loss: {loss_history[0].mean():.4f} (±{loss_history[0].std():.4f})")
    print(f"  Final loss: {loss_history[-1].mean():.4f} (±{loss_history[-1].std():.4f})")
    print(f"  Overall mean: {loss_history.mean():.4f}")
    print(f"  Overall std: {loss_history.std():.4f}")
    print(f"\nGradient Norm Statistics:")
    print(f"  Initial grad norm: {grad_norm_history[0].mean():.4e} (±{grad_norm_history[0].std():.4e})")
    print(f"  Final grad norm: {grad_norm_history[-1].mean():.4e} (±{grad_norm_history[-1].std():.4e})")
    print(f"  Overall mean: {grad_norm_history.mean():.4e}")
    print(f"  Overall std: {grad_norm_history.std():.4e}")
    
    # Print returns statistics if available
    if 'return_history' in training_stats and training_stats['return_history'] is not None:
        return_history = np.array(training_stats['return_history'])
        print(f"\nReturns Statistics:")
        print(f"  Initial returns: {return_history[0].mean():.4f} (±{return_history[0].std():.4f})")
        print(f"  Final returns: {return_history[-1].mean():.4f} (±{return_history[-1].std():.4f})")
        print(f"  Best returns: {return_history.max():.4f}")
        print(f"  Worst returns: {return_history.min():.4f}")
        print(f"  Overall mean: {return_history.mean():.4f}")
        print(f"  Overall std: {return_history.std():.4f}")
    
    print("="*60 + "\n")


def save_training_plots(training_stats, save_path, algorithm_name="RL", dpi=150):
    """
    Create and save training plots to a file.
    
    Args:
        training_stats: Dictionary containing training statistics
        save_path: Path to save the plot
        algorithm_name: Name of the algorithm for plot titles
        dpi: Resolution of saved plot
    """
    fig = plot_training_stats(training_stats, algorithm_name)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 Training plots saved to: {save_path}")
    plt.close(fig)