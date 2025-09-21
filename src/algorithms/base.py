from abc import ABC, abstractmethod
import jax
from typing import Any, Dict

class RLAlgorithm(ABC):
    """Base class for RL algorithms."""
    
    def __init__(self, config):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
    
    @abstractmethod
    def update(self, batch: Any) -> Dict:
        """Single training update."""
        pass
    
    @abstractmethod
    def collect_rollouts(self, env_state, key) -> Any:
        """Collect training data."""
        pass

class OnPolicyAlgorithm(RLAlgorithm):
    """Base for on-policy algorithms (PPO, TRPO)."""
    pass

class OffPolicyAlgorithm(RLAlgorithm):
    """Base for off-policy algorithms (SAC)."""
    pass