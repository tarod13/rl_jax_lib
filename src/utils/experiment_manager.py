"""
Experiment management utilities for organizing runs with unique IDs.
Handles directory structure, config persistence, and checkpoint organization.
"""

import json
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional, Dict, Any
import dataclasses


class ExperimentManager:
    """Manages experiment directories and metadata for a single run."""
    
    def __init__(self, experiments_root: str = "experiments", run_id: Optional[str] = None):
        """
        Initialize the experiment manager.
        
        Args:
            experiments_root: Root directory for all experiments (default: "experiments")
            run_id: Unique run identifier. If None, generates one automatically.
        """
        self.experiments_root = Path(experiments_root)
        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.experiments_root / self.run_id
        
        # Create directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_path = self.run_dir / "config.json"
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID using timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{unique_suffix}"
    
    def save_config(self, config: Any) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration object (dataclass, dict, or object with __dict__)
        """
        # Convert config to dictionary
        if dataclasses.is_dataclass(config):
            config_dict = dataclasses.asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        elif hasattr(config, '__dict__'):
            config_dict = vars(config)
        else:
            raise ValueError(f"Cannot serialize config of type {type(config)}")
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"📋 Config saved to: {self.config_path}")
    
    @staticmethod
    def load_config(run_id: str, experiments_root: str = "experiments") -> Dict[str, Any]:
        """
        Load configuration from a saved run.
        
        Args:
            run_id: Unique run identifier
            experiments_root: Root directory for experiments
            
        Returns:
            Dictionary containing the configuration
        """
        config_path = Path(experiments_root) / run_id / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"📋 Config loaded from: {config_path}")
        return config
    
    def get_checkpoint_path(self, step: int) -> Path:
        """Get path for checkpoint at specific step."""
        return self.checkpoints_dir / f"checkpoint_step_{step}.pkl"
    
    def get_agent_state_path(self, step: int) -> Path:
        """Get path for agent state at specific step."""
        return self.checkpoints_dir / f"agent_state_step_{step}.nnx"
    
    def get_plot_path(self, algorithm_name: str = "algorithm") -> Path:
        """Get path for training plots."""
        return self.plots_dir / f"training_plots_{algorithm_name.lower()}.png"
    
    def list_checkpoints(self) -> list[int]:
        """List available checkpoint steps in this run."""
        checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_step_*.pkl"))
        steps = [int(f.stem.split("_")[-1]) for f in checkpoint_files]
        return sorted(steps)
    
    def print_run_info(self) -> None:
        """Print information about this run."""
        print(f"\n{'='*60}")
        print(f"Run ID: {self.run_id}")
        print(f"Run Directory: {self.run_dir}")
        print(f"Checkpoints: {self.checkpoints_dir}")
        print(f"Plots: {self.plots_dir}")
        print(f"Config: {self.config_path}")
        available_steps = self.list_checkpoints()
        if available_steps:
            print(f"Available checkpoint steps: {available_steps}")
        else:
            print("No checkpoints saved yet")
        print(f"{'='*60}\n")
    
    @staticmethod
    def list_runs(experiments_root: str = "experiments") -> list[str]:
        """List all available run IDs."""
        experiments_path = Path(experiments_root)
        if not experiments_path.exists():
            return []
        
        run_dirs = [d.name for d in experiments_path.iterdir() if d.is_dir()]
        return sorted(run_dirs)