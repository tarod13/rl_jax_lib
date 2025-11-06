from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for experiment management, checkpointing, and evaluation."""
    
    # Experiment tracking
    experiments_root: str = 'experiments'
    run_id: str | None = None  # Auto-generated if None
    experiment_ID: str = ""  # Optional experiment ID
    experiment_description: str = ""  # Optional experiment description
    
    # Checkpointing
    checkpoint_interval: int = 25  # Save every N steps
    keep_only_latest: bool = True  # Only keep the most recent checkpoint
    
    # Evaluation settings
    num_eval_episodes: int = 100  # Number of episodes for evaluation
    max_eval_length: int = 10000  # Maximum episode length for evaluation
    run_eval_on_checkpoint: bool = True  # Run evaluation before saving checkpoints
    
    # Resume training
    resume_run_id: str | None = None  # Run ID to resume from
    resume_step: int | None = None  # Specific step to resume from