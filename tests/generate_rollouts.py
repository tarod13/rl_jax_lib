try:
    from src.utils.rollouts import example_usage as generate_rollouts_example
except:

    from pathlib import Path
    import sys

    # Define repository root for imports
    _THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = _THIS_FILE.parent.parent
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    from src.utils.rollouts import example_usage as generate_rollouts_example


if __name__ == "__main__":
    trajectories, save_path = generate_rollouts_example()
    print(f"Generated trajectories saved at: {save_path}")