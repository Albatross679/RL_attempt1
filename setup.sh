#!/bin/bash

# ============================================================
# Environment Setup Script for RL Training Project
# ============================================================
# This script sets up a complete Python 3.12 environment with all
# dependencies for the DisMech RL training project.
#
# Prerequisites:
#   - Ubuntu 22.04 or compatible Linux
#   - NVIDIA GPU with CUDA support (for GPU training)
#   - sudo access (for Python 3.12 installation if not present)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv312"
PYTHON_VERSION="3.12"

echo "============================================================"
echo "RL Training Environment Setup"
echo "============================================================"
echo ""

# -------------------- Check/Install Python 3.12 --------------------
check_python312() {
    if command -v python${PYTHON_VERSION} &> /dev/null; then
        echo "✅ Python ${PYTHON_VERSION} found: $(python${PYTHON_VERSION} --version)"
        return 0
    else
        echo "❌ Python ${PYTHON_VERSION} not found"
        return 1
    fi
}

install_python312() {
    echo "Installing Python ${PYTHON_VERSION}..."
    echo "This requires sudo access."
    
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
    
    echo "✅ Python ${PYTHON_VERSION} installed"
}

if ! check_python312; then
    read -p "Install Python ${PYTHON_VERSION}? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        install_python312
    else
        echo "Python ${PYTHON_VERSION} is required. Exiting."
        exit 1
    fi
fi

# -------------------- Create Virtual Environment --------------------
echo ""
echo "Creating Python ${PYTHON_VERSION} virtual environment..."

if [ -d "${VENV_PATH}" ]; then
    read -p "Virtual environment already exists. Recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_PATH}"
        python${PYTHON_VERSION} -m venv "${VENV_PATH}"
        echo "✅ Virtual environment recreated at ${VENV_PATH}"
    else
        echo "Using existing virtual environment"
    fi
else
    python${PYTHON_VERSION} -m venv "${VENV_PATH}"
    echo "✅ Virtual environment created at ${VENV_PATH}"
fi

# -------------------- Activate Environment --------------------
source "${VENV_PATH}/bin/activate"
echo "✅ Activated virtual environment"

# -------------------- Upgrade pip --------------------
echo ""
echo "Upgrading pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools
echo "✅ pip upgraded"

# -------------------- Install PyTorch with CUDA --------------------
echo ""
echo "Installing PyTorch with CUDA 11.8 support..."
echo "(This may take several minutes due to large downloads)"

pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo "✅ PyTorch installed"

# -------------------- Install Other Dependencies --------------------
echo ""
echo "Installing remaining dependencies..."

# Core dependencies
pip install numpy==1.26.0 scipy matplotlib pybind11 pybullet opencv-python \
    gym==0.15.4 cloudpickle==1.2.2 pyglet==1.3.2

# ML and training dependencies
pip install absl-py tensorboard protobuf pyyaml h5py numba llvmlite psutil dill tqdm ipython

# Pyelastica for simulation
pip install pyelastica

# Install gin-config from HorizonRobotics
pip install git+https://github.com/HorizonRobotics/gin-config.git@6757358b3faec531741cf138889031efa08fed1e

# Install cnest from HorizonRobotics
pip install git+https://github.com/HorizonRobotics/cnest.git@b7a62849ac4531225229cff3a5d5f8fc654bda3f

echo "✅ Dependencies installed"

# -------------------- Install Local Packages --------------------
echo ""
echo "Installing local packages (editable mode)..."

if [ -d "${SCRIPT_DIR}/alf" ]; then
    pip install -e "${SCRIPT_DIR}/alf"
    echo "✅ ALF installed"
else
    echo "⚠️  ALF directory not found, skipping"
fi

if [ -d "${SCRIPT_DIR}/dismech-python" ]; then
    pip install -e "${SCRIPT_DIR}/dismech-python"
    echo "✅ dismech-python installed"
else
    echo "⚠️  dismech-python directory not found, skipping"
fi

# -------------------- Verify Installation --------------------
echo ""
echo "Verifying installation..."

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
import numpy
print(f'  NumPy: {numpy.__version__}')
import gym
print(f'  Gym: {gym.__version__}')
try:
    import alf
    print('  ALF: ✅')
except ImportError:
    print('  ALF: ❌ (not installed)')
try:
    import dismech
    print('  dismech: ✅')
except ImportError:
    print('  dismech: ❌ (not installed)')
"

# -------------------- Complete --------------------
echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
echo "To run training:"
echo "  cd dismech-rl/scripts"
echo "  bash train_follow_sac.sh"
echo ""
echo "The training scripts will automatically activate the environment."
echo "============================================================"
