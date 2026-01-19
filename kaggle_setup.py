#!/usr/bin/env python3
"""
Kaggle Notebook Setup Script for DisMech RL Training
=====================================================

This script sets up the DisMech RL training environment in a Kaggle notebook.
It clones the GitHub repository and installs all required dependencies.

Usage in Kaggle Notebook:
-------------------------
1. Create a new Kaggle notebook with GPU enabled
2. Copy this entire script into a code cell and run it
3. After setup completes, run training in subsequent cells

Note: Kaggle provides ~30 hours of GPU time per week.
"""

import subprocess
import sys
import os

def run_cmd(cmd, check=True):
    """Run a shell command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=False)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("DisMech RL Training - Kaggle Environment Setup")
    print("=" * 60)
    
    # -------------------- Clone Repository --------------------
    print("\n[1/5] Cloning GitHub repository...")
    
    REPO_URL = "https://github.com/Albatross679/RL_attempt1.git"
    REPO_DIR = "/kaggle/working/RL_attempt1"
    
    if os.path.exists(REPO_DIR):
        print(f"Repository already exists at {REPO_DIR}")
        os.chdir(REPO_DIR)
        run_cmd("git pull", check=False)
    else:
        run_cmd(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
        os.chdir(REPO_DIR)
    
    print(f"Working directory: {os.getcwd()}")
    
    # -------------------- Upgrade pip --------------------
    print("\n[2/5] Upgrading pip and setuptools...")
    run_cmd(f"{sys.executable} -m pip install --upgrade pip setuptools wheel -q")
    
    # -------------------- Install Core Dependencies --------------------
    print("\n[3/5] Installing core dependencies...")
    
    # Kaggle has PyTorch pre-installed, but we may need specific versions
    # Check if torch is available first
    try:
        import torch
        print(f"PyTorch already installed: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("Installing PyTorch...")
        run_cmd(f"{sys.executable} -m pip install torch torchvision torchaudio -q")
    
    # Install other dependencies (from setup.sh)
    deps = [
        "numpy",
        "scipy", 
        "matplotlib",
        "pybind11",
        "pybullet",
        "opencv-python",
        "gym==0.15.4",
        "cloudpickle==1.2.2",
        "pyglet==1.3.2",
        "absl-py",
        "tensorboard",
        "protobuf",
        "pyyaml",
        "h5py",
        "numba",
        "llvmlite",
        "psutil",
        "dill",
        "tqdm",
        "ipython",
        "pyelastica",
        "pyvista",
        "huggingface-hub",
    ]
    
    run_cmd(f"{sys.executable} -m pip install {' '.join(deps)} -q")
    
    # Install HorizonRobotics packages
    print("\n[4/5] Installing HorizonRobotics packages...")
    run_cmd(f"{sys.executable} -m pip install git+https://github.com/HorizonRobotics/gin-config.git@6757358b3faec531741cf138889031efa08fed1e -q")
    run_cmd(f"{sys.executable} -m pip install git+https://github.com/HorizonRobotics/cnest.git@b7a62849ac4531225229cff3a5d5f8fc654bda3f -q")
    
    # -------------------- Install Local Packages --------------------
    print("\n[5/5] Installing local packages (ALF and dismech-python)...")
    
    if os.path.exists("alf"):
        run_cmd(f"{sys.executable} -m pip install -e ./alf -q")
        print("✅ ALF installed")
    else:
        print("⚠️ ALF directory not found")
    
    if os.path.exists("dismech-python"):
        run_cmd(f"{sys.executable} -m pip install -e ./dismech-python -q")
        print("✅ dismech-python installed")
    else:
        print("⚠️ dismech-python directory not found")
    
    # -------------------- Verify Installation --------------------
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)
    
    verification_code = """
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import numpy
print(f'NumPy: {numpy.__version__}')
import gym
print(f'Gym: {gym.__version__}')
try:
    import alf
    print('ALF: ✅')
except ImportError as e:
    print(f'ALF: ❌ ({e})')
try:
    import dismech
    print('dismech: ✅')
except ImportError as e:
    print(f'dismech: ❌ ({e})')
"""
    exec(verification_code)
    
    # -------------------- Print Next Steps --------------------
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print(f"""
Working Directory: {REPO_DIR}

To run training, use the following in a new cell:

    import os
    os.chdir("{REPO_DIR}/dismech-rl")
    
    # For SAC training:
    !python -m alf.bin.train --conf confs/follow_conf.py --root_dir ./results/kaggle_sac
    
    # For PPO training:
    !python -m alf.bin.train --conf confs/follow_ppo_conf.py --root_dir ./results/kaggle_ppo

Note: Reduce NUM_PARALLEL_ENVS for Kaggle's memory limits:
    --conf_param "create_environment.num_parallel_environments=32"
""")

if __name__ == "__main__":
    main()
