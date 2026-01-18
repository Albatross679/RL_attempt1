#!/usr/bin/env python
"""
Kaggle training script for Snake Coil PPO task.
This script runs in a Kaggle notebook environment with GPU acceleration.
"""

import os
import subprocess
import sys

def run_cmd(cmd, check=True):
    """Run shell command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode

def main():
    # Clone the repository with all submodules
    print("=" * 60)
    print("STEP 1: Cloning repository")
    print("=" * 60)
    run_cmd("git clone --recursive https://github.com/Albatross679/RL_attempt1.git")
    os.chdir("RL_attempt1")

    # Install CUDA packages from PyTorch index
    print("\n" + "=" * 60)
    print("STEP 2: Installing CUDA packages from PyTorch index")
    print("=" * 60)
    run_cmd("pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu118")

    # Install llvmlite and triton from PyPI (not available on PyTorch index)
    print("\n" + "=" * 60)
    print("STEP 2b: Installing llvmlite and triton from PyPI")
    print("=" * 60)
    run_cmd("pip install llvmlite==0.46.0 triton==3.3.1")

    # Install editable submodules
    print("\n" + "=" * 60)
    print("STEP 3: Installing dismech-python")
    print("=" * 60)
    run_cmd("pip install -e dismech-python/")

    print("\n" + "=" * 60)
    print("STEP 4: Installing alf")
    print("=" * 60)
    run_cmd("pip install -e alf/")

    # Install remaining dependencies (skip already installed)
    print("\n" + "=" * 60)
    print("STEP 5: Installing remaining dependencies")
    print("=" * 60)
    # Filter out editable installs and large packages already installed
    run_cmd("pip install absl-py gin-config tensorboard cloudpickle gym pybullet matplotlib numpy scipy", check=False)

    # Verify installations
    print("\n" + "=" * 60)
    print("STEP 6: Verifying installations")
    print("=" * 60)
    run_cmd("python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\"")
    run_cmd("python -c \"import alf; print(f'ALF imported successfully')\"")
    run_cmd("python -c \"import dismech; print(f'DisMech imported successfully')\"")

    # Run training
    print("\n" + "=" * 60)
    print("STEP 7: Starting training")
    print("=" * 60)
    run_cmd("python -m alf.bin.train --conf dismech-rl/confs/snake_coil_ppo_conf.py --root_dir /kaggle/working/snake_coil_results")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
