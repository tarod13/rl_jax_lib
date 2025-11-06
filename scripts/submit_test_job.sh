#!/bin/bash
#SBATCH --job-name=reinforce_training
#SBATCH --account=aip-machado
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --output=logs/reinforce_seed_%a.out
#SBATCH --error=logs/reinforce_seed_%a.err

mkdir -p logs

SEED=$((42 + SLURM_ARRAY_TASK_ID))

echo "========================================"
echo "Starting REINFORCE training"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Node: $SLURM_NODELIST"
echo "========================================"

module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate

# Dynamic allocation of GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run training
python tests/test.py \
    --seed $SEED \
    --algorithm ppo \
    --num_training_steps 1000 \
    --checkpoint_interval 50 \
    --num_epochs 10 \
    --num_rollouts 1 \
    --episode_length 2048 \
    --lr 3e-4 \
    --hidden_dim 64 \
    --env_name hopper \
    --num_eval_episodes 100 \
    --run_eval_on_checkpoint true

echo "========================================"
echo "Job completed for seed $SEED"
echo "Results saved to: experiments/run_* (see logs above)"
echo "========================================"