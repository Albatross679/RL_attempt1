#!/bin/bash
# Deployment script for RL_attempt1 project
# Run this on your cloud server to set up the environment

set -e  # Exit on any error

echo "=== Cloning RL_attempt1 repository ==="
git clone git@github.com:Albatross679/RL_attempt1.git
cd RL_attempt1

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Installing CUDA/PyTorch packages ==="
pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu118

echo "=== Setup complete! ==="
echo "Virtual environment is activated and ready to use."
