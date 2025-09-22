import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrajectoryAnalyzer:
    """
    Comprehensive trajectory analysis and visualization class.
    """
    
    def __init__(self, trajectory_file):
        """Load trajectory data from file."""
        self.load_trajectories(trajectory_file)
        self.prepare_data()
    
    def load_trajectories(self, filepath):
        """Load trajectory data from pickle file."""
        print(f"Loading trajectories from: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'trajectories' in data:
            self.trajectories = data['trajectories']
            self.metadata = data.get('metadata', {})
        else:
            # Assume data is the trajectories directly
            self.trajectories = data
            self.metadata = {}
        
        print("Trajectory data loaded successfully!")
        print(f"Metadata: {self.metadata}")
    
    def prepare_data(self):
        """Prepare data for analysis."""
        # Extract basic information
        self.num_rollouts = self.trajectories['obs'].shape[0]
        self.episode_length = self.trajectories['obs'].shape[1]
        self.obs_dim = self.trajectories['obs'].shape[2]
        self.action_dim = self.trajectories['action'].shape[2]
        
        print(f"\nData shape:")
        print(f"  Number of rollouts: {self.num_rollouts}")
        print(f"  Episode length: {self.episode_length}")
        print(f"  Observation dimension: {self.obs_dim}")
        print(f"  Action dimension: {self.action_dim}")
        
        # Flatten data for analysis
        self.obs_flat = self.trajectories['obs'].reshape(-1, self.obs_dim)
        self.action_flat = self.trajectories['action'].reshape(-1, self.action_dim)
        self.reward_flat = self.trajectories['reward'].reshape(-1)
        
        # Check for invalid data
        self.check_data_quality()
    
    def check_data_quality(self):
        """Check for NaN, inf, and other data quality issues."""
        print("\n=== DATA QUALITY CHECK ===")
        
        # Check observations
        obs_nan = np.isnan(self.obs_flat).sum()
        obs_inf = np.isinf(self.obs_flat).sum()
        print(f"Observations - NaN: {obs_nan}, Inf: {obs_inf}")
        
        # Check actions
        action_nan = np.isnan(self.action_flat).sum()
        action_inf = np.isinf(self.action_flat).sum()
        print(f"Actions - NaN: {action_nan}, Inf: {action_inf}")
        
        # Check rewards
        reward_nan = np.isnan(self.reward_flat).sum()
        reward_inf = np.isinf(self.reward_flat).sum()
        print(f"Rewards - NaN: {reward_nan}, Inf: {reward_inf}")
        
        # Check ranges
        print(f"\nData ranges:")
        print(f"  Observations: [{np.nanmin(self.obs_flat):.3f}, {np.nanmax(self.obs_flat):.3f}]")
        print(f"  Actions: [{np.nanmin(self.action_flat):.3f}, {np.nanmax(self.action_flat):.3f}]")
        print(f"  Rewards: [{np.nanmin(self.reward_flat):.3f}, {np.nanmax(self.reward_flat):.3f}]")
        
        # Clean data if necessary
        if obs_nan > 0 or action_nan > 0 or reward_nan > 0:
            print("\nWARNING: NaN values detected! Cleaning data...")
            self.clean_data()
    
    def clean_data(self):
        """Clean invalid data points."""
        # Find valid indices (no NaN or inf)
        obs_valid = np.isfinite(self.obs_flat).all(axis=1)
        action_valid = np.isfinite(self.action_flat).all(axis=1)
        reward_valid = np.isfinite(self.reward_flat)
        
        valid_mask = obs_valid & action_valid & reward_valid
        valid_ratio = valid_mask.sum() / len(valid_mask)
        
        print(f"Valid data ratio: {valid_ratio:.1%}")
        
        if valid_ratio < 0.8:
            print("WARNING: Less than 80% of data is valid!")
        
        # Keep only valid data
        self.obs_flat = self.obs_flat[valid_mask]
        self.action_flat = self.action_flat[valid_mask]
        self.reward_flat = self.reward_flat[valid_mask]
    
    def compute_statistics(self):
        """Compute comprehensive statistics."""
        print("\n=== STATISTICS ===")
        
        # Compute returns for each episode
        rewards_per_episode = self.trajectories['reward'].sum(axis=1)
        
        stats = {
            'episode_returns': {
                'mean': np.mean(rewards_per_episode),
                'std': np.std(rewards_per_episode),
                'min': np.min(rewards_per_episode),
                'max': np.max(rewards_per_episode),
                'median': np.median(rewards_per_episode),
                'q25': np.percentile(rewards_per_episode, 25),
                'q75': np.percentile(rewards_per_episode, 75)
            },
            'step_rewards': {
                'mean': np.mean(self.reward_flat),
                'std': np.std(self.reward_flat),
                'min': np.min(self.reward_flat),
                'max': np.max(self.reward_flat)
            },
            'observations': {
                'mean': np.mean(self.obs_flat, axis=0),
                'std': np.std(self.obs_flat, axis=0),
                'min': np.min(self.obs_flat, axis=0),
                'max': np.max(self.obs_flat, axis=0)
            },
            'actions': {
                'mean': np.mean(self.action_flat, axis=0),
                'std': np.std(self.action_flat, axis=0),
                'min': np.min(self.action_flat, axis=0),
                'max': np.max(self.action_flat, axis=0)
            }
        }
        
        # Print key statistics
        print(f"Episode Returns:")
        for key, value in stats['episode_returns'].items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nStep Rewards:")
        for key, value in stats['step_rewards'].items():
            print(f"  {key}: {value:.3f}")
        
        self.stats = stats
        return stats
    
    def perform_pca_analysis(self):
        """Perform PCA analysis on observations and actions."""
        print("\n=== PCA ANALYSIS ===")
        
        # PCA on observations
        obs_scaler = StandardScaler()
        obs_scaled = obs_scaler.fit_transform(self.obs_flat)
        
        obs_pca = PCA()
        obs_pca_data = obs_pca.fit_transform(obs_scaled)
        
        print(f"Observation PCA:")
        print(f"  Explained variance ratio (first 5 components): {obs_pca.explained_variance_ratio_[:5]}")
        print(f"  Cumulative variance (first 10 components): {np.cumsum(obs_pca.explained_variance_ratio_[:10])}")
        
        # PCA on actions
        action_scaler = StandardScaler()
        action_scaled = action_scaler.fit_transform(self.action_flat)
        
        action_pca = PCA()
        action_pca_data = action_pca.fit_transform(action_scaled)
        
        print(f"\nAction PCA:")
        print(f"  Explained variance ratio (first 5 components): {action_pca.explained_variance_ratio_[:5]}")
        print(f"  Cumulative variance (first {min(10, self.action_dim)} components): {np.cumsum(action_pca.explained_variance_ratio_[:min(10, self.action_dim)])}")
        
        # Store results
        self.obs_pca = obs_pca
        self.obs_pca_data = obs_pca_data
        self.action_pca = action_pca
        self.action_pca_data = action_pca_data
        
        return obs_pca, action_pca
    
    def create_visualizations(self, save_dir="analysis_plots"):
        """Create comprehensive visualizations."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Set up the plotting
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Episode returns distribution
        self.plot_returns_distribution(save_dir)
        
        # 2. PCA visualizations
        self.plot_pca_analysis(save_dir)
        
        # 3. Action and observation distributions
        self.plot_distributions(save_dir)
        
        # 4. Trajectory visualization
        self.plot_trajectory_analysis(save_dir)
        
        # 5. Correlation analysis
        self.plot_correlation_analysis(save_dir)
        
        print(f"\nPlots saved to: {save_dir}/")
    
    def plot_returns_distribution(self, save_dir):
        """Plot episode returns distribution."""
        returns = self.trajectories['reward'].sum(axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Episode Returns Analysis', fontsize=16)
        
        # Histogram
        axes[0, 0].hist(returns, bins=30, alpha=0.7, density=True)
        axes[0, 0].axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}')
        axes[0, 0].set_xlabel('Episode Return')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(returns)
        axes[0, 1].set_ylabel('Episode Return')
        axes[0, 1].set_title('Returns Box Plot')
        
        # Returns over episodes
        axes[1, 0].plot(returns, alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].set_title('Returns Over Episodes')
        
        # Cumulative average
        cumulative_avg = np.cumsum(returns) / np.arange(1, len(returns) + 1)
        axes[1, 1].plot(cumulative_avg)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Average Return')
        axes[1, 1].set_title('Learning Curve (if episodes are sequential)')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/returns_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, save_dir):
        """Plot PCA analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PCA Analysis', fontsize=16)
        
        # Observation PCA - explained variance
        axes[0, 0].plot(np.cumsum(self.obs_pca.explained_variance_ratio_))
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Cumulative Explained Variance')
        axes[0, 0].set_title('Observation PCA - Explained Variance')
        axes[0, 0].grid(True)
        
        # Observation PCA - 2D projection
        scatter = axes[0, 1].scatter(self.obs_pca_data[:, 0], self.obs_pca_data[:, 1], 
                                   c=self.reward_flat, alpha=0.6, cmap='viridis')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        axes[0, 1].set_title('Observation PCA - PC1 vs PC2')
        plt.colorbar(scatter, ax=axes[0, 1], label='Reward')
        
        # Observation PCA - 3D-like view (PC1 vs PC3)
        if self.obs_pca_data.shape[1] > 2:
            axes[0, 2].scatter(self.obs_pca_data[:, 0], self.obs_pca_data[:, 2], 
                             c=self.reward_flat, alpha=0.6, cmap='viridis')
            axes[0, 2].set_xlabel('PC1')
            axes[0, 2].set_ylabel('PC3')
            axes[0, 2].set_title('Observation PCA - PC1 vs PC3')
        
        # Action PCA - explained variance
        axes[1, 0].plot(np.cumsum(self.action_pca.explained_variance_ratio_))
        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Cumulative Explained Variance')
        axes[1, 0].set_title('Action PCA - Explained Variance')
        axes[1, 0].grid(True)
        
        # Action PCA - 2D projection
        axes[1, 1].scatter(self.action_pca_data[:, 0], self.action_pca_data[:, 1], 
                          c=self.reward_flat, alpha=0.6, cmap='viridis')
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        axes[1, 1].set_title('Action PCA - PC1 vs PC2')
        
        # Action distribution in PC space
        if self.action_pca_data.shape[1] > 2:
            axes[1, 2].scatter(self.action_pca_data[:, 0], self.action_pca_data[:, 2], 
                             c=self.reward_flat, alpha=0.6, cmap='viridis')
            axes[1, 2].set_xlabel('PC1')
            axes[1, 2].set_ylabel('PC3')
            axes[1, 2].set_title('Action PCA - PC1 vs PC3')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pca_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distributions(self, save_dir):
        """Plot action and observation distributions."""
        # Sample a subset for plotting if data is large
        n_samples = min(10000, len(self.obs_flat))
        idx = np.random.choice(len(self.obs_flat), n_samples, replace=False)
        
        obs_sample = self.obs_flat[idx]
        action_sample = self.action_flat[idx]
        
        # Plot observation distributions (first few dimensions)
        n_obs_to_plot = min(6, self.obs_dim)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Observation Distributions (First 6 Dimensions)', fontsize=16)
        
        for i in range(n_obs_to_plot):
            row, col = i // 3, i % 3
            axes[row, col].hist(obs_sample[:, i], bins=50, alpha=0.7)
            axes[row, col].set_title(f'Observation Dim {i}')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(n_obs_to_plot, 6):
            row, col = i // 3, i % 3
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/observation_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot action distributions
        n_action_to_plot = min(6, self.action_dim)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Action Distributions', fontsize=16)
        
        for i in range(n_action_to_plot):
            row, col = i // 3, i % 3
            axes[row, col].hist(action_sample[:, i], bins=50, alpha=0.7)
            axes[row, col].set_title(f'Action Dim {i}')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(n_action_to_plot, 6):
            row, col = i // 3, i % 3
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trajectory_analysis(self, save_dir):
        """Plot trajectory-specific analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trajectory Analysis', fontsize=16)
        
        # Reward over time (average across episodes)
        avg_reward_over_time = np.mean(self.trajectories['reward'], axis=0)
        std_reward_over_time = np.std(self.trajectories['reward'], axis=0)
        
        time_steps = np.arange(len(avg_reward_over_time))
        axes[0, 0].plot(time_steps, avg_reward_over_time, label='Mean')
        axes[0, 0].fill_between(time_steps, 
                               avg_reward_over_time - std_reward_over_time,
                               avg_reward_over_time + std_reward_over_time,
                               alpha=0.3, label='±1 std')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Average Reward Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # State space exploration (using first 2 PCA components)
        axes[0, 1].scatter(self.obs_pca_data[:, 0], self.obs_pca_data[:, 1], 
                          alpha=0.1, s=1)
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        axes[0, 1].set_title('State Space Exploration (PCA)')
        
        # Action magnitude over time
        action_magnitudes = np.linalg.norm(self.trajectories['action'], axis=2)
        avg_action_mag = np.mean(action_magnitudes, axis=0)
        std_action_mag = np.std(action_magnitudes, axis=0)
        
        axes[1, 0].plot(time_steps, avg_action_mag, label='Mean')
        axes[1, 0].fill_between(time_steps,
                               avg_action_mag - std_action_mag,
                               avg_action_mag + std_action_mag,
                               alpha=0.3, label='±1 std')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Action Magnitude')
        axes[1, 0].set_title('Action Magnitude Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Episode length distribution (if done flags are available)
        if 'done' in self.trajectories:
            episode_lengths = []
            for episode in range(self.num_rollouts):
                done_flags = self.trajectories['done'][episode]
                if np.any(done_flags):
                    length = np.argmax(done_flags) + 1
                else:
                    length = self.episode_length
                episode_lengths.append(length)
            
            axes[1, 1].hist(episode_lengths, bins=20, alpha=0.7)
            axes[1, 1].set_xlabel('Episode Length')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Episode Length Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No episode termination data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Episode Length Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/trajectory_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self, save_dir):
        """Plot correlation analysis between different components."""
        # Sample data for correlation analysis
        n_samples = min(5000, len(self.obs_flat))
        idx = np.random.choice(len(self.obs_flat), n_samples, replace=False)
        
        # Create correlation matrix for first few PCA components
        pca_components = np.column_stack([
            self.obs_pca_data[idx, :min(5, self.obs_pca_data.shape[1])],
            self.action_pca_data[idx, :min(3, self.action_pca_data.shape[1])],
            self.reward_flat[idx].reshape(-1, 1)
        ])
        
        columns = [f'Obs_PC{i+1}' for i in range(min(5, self.obs_pca_data.shape[1]))]
        columns += [f'Action_PC{i+1}' for i in range(min(3, self.action_pca_data.shape[1]))]
        columns += ['Reward']
        
        corr_matrix = np.corrcoef(pca_components.T)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.set_yticklabels(columns)
        
        # Add correlation values
        for i in range(len(columns)):
            for j in range(len(columns)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Correlation Matrix (PCA Components + Rewards)', fontsize=16)
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()


def load_and_analyze(trajectory_file):
    """
    Main function to load and analyze trajectory data.
    
    Args:
        trajectory_file: Path to the saved trajectory pickle file
    """
    # Create analyzer
    analyzer = TrajectoryAnalyzer(trajectory_file)
    
    # Compute statistics
    stats = analyzer.compute_statistics()
    
    # Perform PCA analysis
    obs_pca, action_pca = analyzer.perform_pca_analysis()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    returns = analyzer.trajectories['reward'].sum(axis=1)
    print(f"Data contains {analyzer.num_rollouts} episodes of length {analyzer.episode_length}")
    print(f"Mean episode return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"Observation space: {analyzer.obs_dim}D -> {np.sum(obs_pca.explained_variance_ratio_ > 0.01)} effective dimensions")
    print(f"Action space: {analyzer.action_dim}D -> {np.sum(action_pca.explained_variance_ratio_ > 0.01)} effective dimensions")
    
    if np.isnan(returns).any():
        print("⚠️  WARNING: NaN values detected in returns!")
    else:
        print("✅ Data quality check passed!")
    
    return analyzer

if __name__ == "__main__":
    # Example usage
    trajectory_file = "rollout_data/trajectories_20250921_230304.pkl"  # Adjust path as needed
    
    if Path(trajectory_file).exists():
        analyzer = load_and_analyze(trajectory_file)
    else:
        print(f"Trajectory file {trajectory_file} not found!")
        print("Please run the rollout generation script first.")