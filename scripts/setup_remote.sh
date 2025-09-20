#!/bin/bash
# scripts/setup_remote.sh

# 1. Load system modules (Compute Canada specific)
module load python/3.11 cuda/12.1 apptainer/1.2

# 2. Create isolated Python environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax brax optax nnx wandb
pip install -e .  # Install your package in development mode