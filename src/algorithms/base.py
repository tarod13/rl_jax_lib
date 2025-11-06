from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import pickle
from brax import envs
from flax import nnx
import jax
import jax.numpy as jnp


class RLAlgorithm(nnx.Module, ABC):
    """Base class for RL algorithms."""
    
    def __init__(self, config, experiment_manager=None):
        # Save config
        self.config = config
        self.experiment_manager = experiment_manager
        
        # Initialize environment
        self.env = envs.create(config.env_name, auto_reset=False)  # Disable auto-reset here; handle manually in rollouts
        self.action_dim = self.env.action_size
        self.obs_dim = self.env.observation_size

        # Generate num_rollouts initial states from the single environment
        key = jax.random.PRNGKey(self.config.seed)
        reset_keys = jax.random.split(key, self.config.num_rollouts)

        initial_states = [self.env.reset(k) for k in reset_keys]
        initial_states_stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *initial_states
        )  # Merge initial states into a single PyTree with leading batch dim
        self.initial_states = nnx.data(initial_states_stacked)  # Store as NNX data to avoid being treated as parameters

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
    
    def save_state(self, path: Optional[Path] = None):
        """
        Save agent state (network and optimizer).
        
        Args:
            path: Path to save state file. If None, uses experiment manager's path.
        """
        if path is None:
            if self.experiment_manager is None:
                raise ValueError("Must provide path or experiment_manager")
            # This will be called from save_checkpoint with explicit step
            path = Path(path) if path else None
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the state from NNX
        _, state = nnx.split(self)
        
        # Save using pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Agent state saved to: {path}")
    
    def load_state(self, path: Path):
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
        
        print(f"🔓 Agent state loaded from: {path}")
    
    def save_checkpoint(self, step: int, training_stats: Dict, keep_only_latest: bool = True):
        """
        Save training checkpoint using experiment manager.
        
        Args:
            step: Current training step
            training_stats: Dictionary of training statistics
            keep_only_latest: If True, delete previous checkpoints (default: True)
        """
        if self.experiment_manager is None:
            raise ValueError("experiment_manager must be set to save checkpoints")
        
        checkpoint_dir = self.experiment_manager.checkpoints_dir
        
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
        agent_state_path = self.experiment_manager.get_agent_state_path(step)
        self.save_state(agent_state_path)
        
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
        
        checkpoint_path = self.experiment_manager.get_checkpoint_path(step)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Checkpoint saved at step {step}: {checkpoint_path}")
    
    @classmethod
    def load_checkpoint(cls, run_id: str, step: int, experiment_manager_class=None, experiments_root: str = "experiments", agent=None):
        """
        Load training checkpoint using run ID and step.
        
        Args:
            run_id: Unique run identifier
            step: Checkpoint step to load
            experiment_manager_class: ExperimentManager class (auto-imported if None)
            experiments_root: Root directory for experiments
            agent: Optional existing agent to load state into
            
        Returns:
            agent: RL agent with loaded state
            training_stats: Training statistics dictionary
            step: Step number from checkpoint
        """
        if experiment_manager_class is None:
            from ..utils import ExperimentManager
            experiment_manager_class = ExperimentManager
        
        # Create experiment manager for this run
        exp_manager = experiment_manager_class(experiments_root=experiments_root, run_id=run_id)
        
        # Load checkpoint metadata
        checkpoint_path = exp_manager.get_checkpoint_path(step)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
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
        agent_state_path = exp_manager.get_agent_state_path(step)
        
        if agent is None:
            # Create new agent with config from checkpoint
            agent = cls(config, experiment_manager=exp_manager)
        else:
            agent.experiment_manager = exp_manager
        
        agent.load_state(agent_state_path)
        
        print(f"🔓 Checkpoint loaded from step {step}: {checkpoint_path}")
        
        return agent, training_stats, step


class OffPolicyAlgorithm(RLAlgorithm):
    """Base for off-policy algorithms (SAC)."""
    pass