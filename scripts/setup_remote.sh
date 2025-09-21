#!/bin/bash
# Setup script for RL library on Compute Canada

# Load required system modules
module load python/3.11 cuda/12.9

# Create virtual environment (only if it doesn't exist)
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Update pip and install core ML packages
pip install --upgrade pip
pip install jax flax optax tensorboard

# Install Brax dependencies one by one to avoid conflicts
pip install etils flask jaxopt trimesh

# Install Brax without auto-installing problematic dependencies
pip install --no-deps brax

# Prevent JAX from grabbing all GPU memory at once
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Setup complete!"