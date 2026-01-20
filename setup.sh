#!/bin/bash

# ============================================================
# Environment Setup Script for RL Training Project
# ============================================================
# Multi-platform setup for: OSC HPC, Oracle Cloud, generic Linux
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv"

echo "============================================================"
echo "RL Training Environment Setup"
echo "============================================================"
echo ""

# -------------------- Detect Python 3.12 --------------------
detect_python312() {
    # Priority 1: HPC module system
    if command -v module &>/dev/null; then
        echo "HPC environment detected, loading python/3.12 module..."
        module load python/3.12 2>/dev/null && {
            PYTHON_CMD="python3"
            echo "Using HPC module: python/3.12"
            return 0
        }
    fi

    # Priority 2: System python3.12
    if command -v python3.12 &>/dev/null; then
        PYTHON_CMD="python3.12"
        echo "Using system Python: $(python3.12 --version)"
        return 0
    fi

    # Priority 3: Check if python3 is 3.12
    if command -v python3 &>/dev/null; then
        PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$PY_VERSION" == "3.12" ]]; then
            PYTHON_CMD="python3"
            echo "Using python3: $(python3 --version)"
            return 0
        fi
    fi

    return 1
}

if ! detect_python312; then
    echo "Python 3.12 not found. Install options:"
    echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo "  Oracle Cloud:  sudo dnf install python3.12"
    echo "  RHEL/CentOS:   sudo dnf install python3.12"
    exit 1
fi

# -------------------- Create Virtual Environment --------------------
echo ""
echo "Creating Python 3.12 virtual environment..."

if [ -d "${VENV_PATH}" ]; then
    read -p "Virtual environment already exists. Recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_PATH}"
        $PYTHON_CMD -m venv "${VENV_PATH}"
        echo "Virtual environment recreated at ${VENV_PATH}"
    else
        echo "Using existing virtual environment"
    fi
else
    $PYTHON_CMD -m venv "${VENV_PATH}"
    echo "Virtual environment created at ${VENV_PATH}"
fi

# -------------------- Activate Environment --------------------
source "${VENV_PATH}/bin/activate"
echo "Activated virtual environment"

# -------------------- Install Packages --------------------
echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo ""
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing dependencies..."
pip install numpy==1.26.0 scipy matplotlib pybind11 pybullet opencv-python \
    gym==0.15.4 cloudpickle==1.2.2 pyglet==1.3.2

pip install absl-py tensorboard protobuf pyyaml h5py numba llvmlite psutil dill tqdm ipython ninja
pip install pyelastica
pip install git+https://github.com/HorizonRobotics/gin-config.git@6757358b3faec531241cf138889031efa08fed1e
pip install git+https://github.com/HorizonRobotics/cnest.git@b7a62849ac4531225229cff3a5d5f8fc654bda3f

echo ""
echo "Installing local packages (editable mode)..."
pip install -e "${SCRIPT_DIR}/alf"
pip install -e "${SCRIPT_DIR}/dismech-python"

# -------------------- Optional: Claude Code CLI --------------------
echo ""
read -p "Install Claude Code CLI? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v npm &>/dev/null; then
        npm install -g @anthropic-ai/claude-code || echo "npm install failed, trying curl..."
        curl -fsSL https://claude.ai/install.sh | sh
    else
        curl -fsSL https://claude.ai/install.sh | sh
    fi
fi

# -------------------- Verify Installation --------------------
echo ""
echo "Verifying installation..."

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
import numpy
print(f'  NumPy: {numpy.__version__}')
import alf
print('  ALF: OK')
import dismech
print('  dismech: OK')
"

# -------------------- Verify Ninja (Critical for SLURM) --------------------
echo ""
echo "Verifying ninja (required for PyTorch JIT)..."

# Check ninja is in venv
if [ -f "${VENV_PATH}/bin/ninja" ]; then
    echo "  Ninja location: ${VENV_PATH}/bin/ninja"
    echo "  Ninja version: $(${VENV_PATH}/bin/ninja --version)"
else
    echo "  WARNING: Ninja not found in venv!"
    echo "  Run: pip install ninja"
fi

# Test ninja subprocess call (matches PyTorch's check)
echo "  Testing subprocess call (PyTorch JIT requirement)..."
python3 -c "
import subprocess
try:
    subprocess.check_output('ninja --version'.split())
    print('  Ninja subprocess check: PASSED')
except Exception as e:
    print(f'  Ninja subprocess check: FAILED - {e}')
    print('  This may cause issues with PyTorch JIT compilation.')
"

# -------------------- SLURM Instructions --------------------
echo ""
echo "============================================================"
echo "IMPORTANT: For SLURM Jobs (OSC Cluster)"
echo "============================================================"
echo ""
echo "When running on SLURM, add these environment settings:"
echo ""
echo "# 1. Load modules"
echo "module load python/3.12"
echo "module load gcc/13.2.0"
echo ""
echo "# 2. Set compilers to GCC (avoid deprecated ICC)"
echo "export CC=gcc"
echo "export CXX=g++"
echo ""
echo "# 3. Intel runtime libraries (for cnest - compiled against Intel)"
echo "export LD_LIBRARY_PATH=\"/apps/spack/0.21/pitzer/linux-rhel9-skylake/intel-oneapi-compilers/gcc/11.4.1/2023.2.3-xq4aqvz/compiler/2023.2.3/linux/compiler/lib/intel64_lin:\${LD_LIBRARY_PATH}\""
echo ""
echo "# 4. GCC 13.2.0 libstdc++ (for PyTorch JIT extensions)"
echo "export LD_LIBRARY_PATH=\"/apps/spack/0.21/pitzer/linux-rhel9-skylake/gcc/gcc/11.4.1/13.2.0-dveccoq/lib64:\${LD_LIBRARY_PATH}\""
echo "export LD_PRELOAD=\"/apps/spack/0.21/pitzer/linux-rhel9-skylake/gcc/gcc/11.4.1/13.2.0-dveccoq/lib64/libstdc++.so.6\""
echo ""
echo "# 5. Clear cached torch extensions"
echo "rm -rf ~/.cache/torch_extensions/py312_cu118/penv 2>/dev/null || true"
echo ""
echo "# 6. Add venv bin to PATH BEFORE activation"
echo "export PATH=\"${VENV_PATH}/bin:\${PATH}\""
echo "source ${VENV_PATH}/bin/activate"
echo ""
echo "# 7. CUBLAS deterministic mode (required for PyTorch)"
echo "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo ""
echo "See submit_snake_ppo.slurm for a complete working example."
echo ""

# -------------------- Complete --------------------
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
echo "To run training:"
echo "  cd dismech-rl/scripts"
echo "  bash train_follow_sac.sh"
echo ""
