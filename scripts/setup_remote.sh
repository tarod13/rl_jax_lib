#!/bin/bash
# scripts/setup_remote.sh

module load python/3.11 cuda/12.9 apptainer/1.2

if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Core packages are already installed, but let's ensure we have them
pip install jax flax optax tensorboard

# Set up MuJoCo environment BEFORE installing
mkdir -p $HOME/.mujoco
export MUJOCO_PATH=$HOME/.mujoco

# Download and install MuJoCo binaries
cd $HOME/.mujoco
if [ ! -f "mujoco-2.3.7-linux-x86_64.tar.gz" ]; then
    wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
    tar -xzf mujoco-2.3.7-linux-x86_64.tar.gz
fi

export MUJOCO_PATH=$HOME/.mujoco/mujoco-2.3.7
export LD_LIBRARY_PATH=$MUJOCO_PATH/lib:$LD_LIBRARY_PATH

# Go back to project directory
cd $OLDPWD

# Now install MuJoCo Python package
pip install mujoco

# Then install Brax
pip install brax

echo "Setup complete!"
echo "MuJoCo path: $MUJOCO_PATH"