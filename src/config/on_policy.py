from dataclasses import dataclass


@dataclass
class OnPolicyConfig:
    """Base configuration for on-policy (rollout-based) algorithms."""
    
    # Algorithm selection
    algorithm: str = 'PPO'  # 'PPO', 'REINFORCE', 'REINFORCEwithBaseline', 'REINFORCEwithBaselineIS'
    
    # Environment settings
    env_name: str = 'hopper'
    seed: int = 42
    
    # Rollout settings
    num_rollouts: int = 1
    episode_length: int = 2048
    deterministic: bool = False
    
    # Network architecture
    hidden_dim: int = 64
    nl: str = 'tanh'  # Non-linearity for networks: 'relu' or 'tanh'
    use_layernorm: bool = True  # Use LayerNorm after each hidden layer
    
    # Training settings
    lr: float = 3e-4
    num_training_steps: int = 500
    gamma: float = 0.99  # Discount factor
    num_epochs: int = 10  # Number of epochs/updates-per-step over the collected data