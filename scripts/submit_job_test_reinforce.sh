#!/bin/bash
#SBATCH --job-name=reinforce_training
#SBATCH --account=aip-machado
#SBATCH --time=02:00:00                  # Maximum 2 hours per job
#SBATCH --cpus-per-task=4                # Number of CPU cores
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem=16G                        # Memory per job
#SBATCH --array=0-9                      # Run 10 jobs (seeds 42-51)
#SBATCH --output=logs/reinforce_seed_%a.out
#SBATCH --error=logs/reinforce_seed_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate seed based on array task ID (42, 43, 44, ..., 51)
SEED=$((42 + SLURM_ARRAY_TASK_ID))

echo "========================================"
echo "Starting REINFORCE training"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load modules and activate environment
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate

# Prevent JAX from grabbing all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Create output directories for this seed
CHECKPOINT_DIR="checkpoints/seed_${SEED}"
PLOT_DIR="analysis_plots"
PLOT_PATH="${PLOT_DIR}/training_plots_seed_${SEED}.png"

mkdir -p $CHECKPOINT_DIR
mkdir -p $PLOT_DIR

# Run training with seed-specific directories
python test_reinforce.py \
    --seed $SEED \
    --checkpoint_dir $CHECKPOINT_DIR \
    --plot_path $PLOT_PATH \
    --num_training_steps 10000 \
    --checkpoint_interval 100 \
    --num_updates_per_step 1 \
    --num_rollouts 100 \
    --episode_length 1000 \
    --lr 1e-5 \
    --hidden_dim 64 \
    --env_name hopper

echo "========================================"
echo "Job completed for seed $SEED"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Plots saved to: $PLOT_PATH"
echo "========================================"