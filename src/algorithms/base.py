from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path
import pickle
from brax import envs
from flax import nnx


class RLAlgorithm(nnx.Module, ABC):
    """Base class for RL algorithms."""
    
    def __init__(self, config):
        # Save config
        self.config = config
        
        # Initialize environment
        self.env = envs.get_environment(config.env_name)
        self.action_dim = self.env.action_size
        self.obs_dim = self.env.observation_size
        
        # Initialize network (algorithm-specific)
        self._init_network()
    
    @abstractmethod
    def _init_network(self):
        """
        Initialize the network(s) for this algorithm.
        Must set self.network at minimum.
        
        This is algorithm-specific - different algorithms may need:
        - Different network architectures
        - Multiple networks (actor-critic, Q-networks, etc.)
        - Different parameterizations
        """
        pass
    
    @abstractmethod
    def update(self, batch: Any) -> Dict:
        """Single training update."""
        pass
    
    @abstractmethod
    def collect_rollouts(self, key) -> Any:
        """Collect training data."""
        pass
    
    def save_state(self, path):
        """
        Save agent state (network and optimizer).
        
        Args:
            path: Path to save state file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the state from NNX
        _, state = nnx.split(self)
        
        # Save using pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Agent state saved to: {path}")
    
    def load_state(self, path):
        """
        Load agent state (network and optimizer).
        
        Args:
            path: Path to state file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        
        # Load state
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Update self with loaded state
        nnx.update(self, state)
        
        print(f"📂 Agent state loaded from: {path}")
    
    def save_checkpoint(self, checkpoint_dir, step, training_stats, keep_only_latest=True):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            step: Current training step
            training_stats: Dictionary of training statistics
            keep_only_latest: If True, delete previous checkpoints (default: True)
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Delete old checkpoints if keep_only_latest is True
        if keep_only_latest:
            old_checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pkl"))
            old_agent_states = list(checkpoint_dir.glob("agent_state_step_*.nnx"))
            
            for old_file in old_checkpoints + old_agent_states:
                try:
                    old_file.unlink()
                    print(f"🗑️  Removed old checkpoint: {old_file.name}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not remove {old_file.name}: {e}")
        
        # Save agent state
        self.save_state(checkpoint_dir / f'agent_state_step_{step}.nnx')
        
        # Convert config to dict to avoid pickle issues with dataclasses
        if hasattr(self.config, '__dataclass_fields__'):
            # It's a dataclass, use dataclasses.asdict
            import dataclasses
            config_dict = dataclasses.asdict(self.config)
        elif hasattr(self.config, '__dict__'):
            # It has __dict__, use that
            config_dict = vars(self.config)
        else:
            # Fallback: store as-is and hope for the best
            config_dict = self.config
        
        # Save training stats and metadata
        checkpoint = {
            'step': step,
            'training_stats': training_stats,
            'config': config_dict,  # Store as dict instead of object
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Checkpoint saved at step {step}: {checkpoint_path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path, agent=None):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            agent: Optional existing agent to load state into
            
        Returns:
            agent: RL agent with loaded state
            training_stats: Training statistics dictionary
            step: Step number from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint metadata
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        step = checkpoint['step']
        training_stats = checkpoint['training_stats']
        config = checkpoint['config']
        
        # If config is a dict (new format), convert to SimpleNamespace for attribute access
        if isinstance(config, dict):
            from types import SimpleNamespace
            config = SimpleNamespace(**config)
        
        # Load agent state
        agent_state_path = checkpoint_path.parent / f'agent_state_step_{step}.nnx'
        
        if agent is None:
            # Create new agent with config from checkpoint
            agent = cls(config)
        
        agent.load_state(agent_state_path)
        
        print(f"📂 Checkpoint loaded from step {step}: {checkpoint_path}")
        
        return agent, training_stats, step


class OffPolicyAlgorithm(RLAlgorithm):
    """Base for off-policy algorithms (SAC)."""
    pass