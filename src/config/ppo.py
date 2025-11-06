from dataclasses import dataclass
from .on_policy import OnPolicyConfig


@dataclass
class PPOConfig(OnPolicyConfig):
    """Configuration for Proximal Policy Optimization (PPO) algorithm."""
    
    # Override algorithm name
    algorithm: str = 'PPO'
    
    # PPO-specific parameters
    epsilon: float = 0.1  # Clipping parameter for policy updates
    minibatch_size: int | None = 32  # Minibatch size K (None = use full batch)
    max_grad_norm: float | None = 0.5  # Max norm for gradient clipping (None = no clipping)
    
    # Advantage estimation
    use_gae: bool = True  # Use Generalized Advantage Estimation (GAE)
    gae_lambda: float = 0.95  # Lambda parameter for GAE (bias-variance tradeoff)
    normalize_advantages: bool = True  # Normalize advantages (subtract mean, divide by stddev)
    
    # Bootstrap for final states
    use_bootstrap_for_final_states: bool = True  # Use value function bootstrap for terminal states