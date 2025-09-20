#!/bin/bash
# scripts/setup_remote.sh

# Load system modules (Compute Canada specific)
module load python/3.11 cuda/12.9 apptainer/1.2

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install JAX first (without CUDA for now, we'll add it later)
echo "Installing JAX..."
pip install jax

# Install other packages (excluding mujoco for now)
echo "Installing other packages..."
pip install flax optax tensorboard

# Install Brax (this might pull in some mujoco dependencies but should work)
echo "Installing Brax..."
pip install brax

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "✅ Setup complete!"
echo "Note: CUDA support may need to be configured separately"
echo "Test with: python -c 'import jax; print(jax.devices())'"