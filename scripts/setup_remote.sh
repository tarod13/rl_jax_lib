#!/bin/bash
# scripts/setup_simple_mujoco.sh

module load python/3.11 cuda/12.9 apptainer/1.2

if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

# Install packages in order
pip install jax flax optax tensorboard

# Install mujoco with pre-built binaries
pip install mujoco-py  # Alternative MuJoCo binding
# OR
pip install mujoco>=2.3.7  # Official MuJoCo with binaries

pip install brax

echo "✅ MuJoCo setup complete!"