#!/bin/bash
#=============================================================================
# INTEGRATION TEST: Quick verification job (15 min max)
#
# PURPOSE:
#   Test that the environment setup works correctly before running full training.
#   Verifies: ninja, PyTorch JIT compilation, CUDA, environment imports.
#
# USAGE:
#   sbatch test_integration.slurm
#   # Wait for completion, then check:
#   cat slurm_logs/test_integration_<job_id>.out | grep "PASSED\|FAILED"
#=============================================================================

#SBATCH --job-name=test_integration
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/test_integration_%j.out
#SBATCH --error=slurm_logs/test_integration_%j.err

# ============================================================
# Environment Setup (same as submit_snake_ppo.slurm)
# ============================================================
PROJECT_ROOT="/users/PAS3272/qifanwen/RL_attempt1"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Load required modules
module load python/3.12
module load gcc/13.2.0

# Set compilers to GCC (avoid deprecated ICC)
export CC=gcc
export CXX=g++

# Intel runtime libraries needed for cnest (compiled against Intel)
# This provides libimf.so without loading the full Intel compiler module
export LD_LIBRARY_PATH="/apps/spack/0.21/pitzer/linux-rhel9-skylake/intel-oneapi-compilers/gcc/11.4.1/2023.2.3-xq4aqvz/compiler/2023.2.3/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}"

# GCC 13.2.0 runtime libraries - provides GLIBCXX_3.4.32 for JIT-compiled extensions
# Python bundles an older libstdc++ with RPATH, so we must use LD_PRELOAD to override it
export LD_LIBRARY_PATH="/apps/spack/0.21/pitzer/linux-rhel9-skylake/gcc/gcc/11.4.1/13.2.0-dveccoq/lib64:${LD_LIBRARY_PATH}"
export LD_PRELOAD="/apps/spack/0.21/pitzer/linux-rhel9-skylake/gcc/gcc/11.4.1/13.2.0-dveccoq/lib64/libstdc++.so.6"

# Clear cached torch extensions to ensure they use correct compiler/libs
rm -rf /users/PAS3272/qifanwen/.cache/torch_extensions/py312_cu118/penv 2>/dev/null || true

# CRITICAL: Export venv bin to PATH BEFORE activation
export PATH="${VENV_PATH}/bin:${PATH}"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Performance settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Required for PyTorch deterministic algorithms with cuBLAS on CUDA >= 10.2
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================
# Pre-flight Checks
# ============================================================
echo "============================================================"
echo "INTEGRATION TEST - Environment Verification"
echo "============================================================"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Python
echo "[TEST 1] Python availability..."
if python3 --version &>/dev/null; then
    echo "  PASSED: $(python3 --version)"
    ((TESTS_PASSED++))
else
    echo "  FAILED: Python not found"
    ((TESTS_FAILED++))
fi

# Test 2: Ninja in PATH
echo "[TEST 2] Ninja in PATH..."
if which ninja &>/dev/null; then
    echo "  PASSED: $(which ninja) - $(ninja --version)"
    ((TESTS_PASSED++))
else
    echo "  FAILED: Ninja not found in PATH"
    ((TESTS_FAILED++))
fi

# Test 3: Ninja subprocess check (PyTorch's method)
echo "[TEST 3] Ninja subprocess check (PyTorch JIT requirement)..."
if python3 -c "import subprocess; subprocess.check_output('ninja --version'.split())" 2>/dev/null; then
    echo "  PASSED: subprocess.check_output('ninja --version') works"
    ((TESTS_PASSED++))
else
    echo "  FAILED: Ninja subprocess check failed"
    ((TESTS_FAILED++))
fi

# Test 4: CUDA availability
echo "[TEST 4] CUDA availability..."
CUDA_RESULT=$(python3 -c "import torch; print('available' if torch.cuda.is_available() else 'not_available')" 2>/dev/null)
if [[ "$CUDA_RESULT" == "available" ]]; then
    DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  PASSED: CUDA available - $DEVICE_NAME"
    ((TESTS_PASSED++))
else
    echo "  FAILED: CUDA not available"
    ((TESTS_FAILED++))
fi

# Test 5: GCC compiler
echo "[TEST 5] GCC compiler..."
if gcc --version &>/dev/null; then
    GCC_VER=$(gcc --version | head -n1)
    echo "  PASSED: $GCC_VER"
    ((TESTS_PASSED++))
else
    echo "  FAILED: GCC not found"
    ((TESTS_FAILED++))
fi

# Test 6: ALF import
echo "[TEST 6] ALF framework import..."
if python3 -c "import alf" 2>/dev/null; then
    echo "  PASSED: ALF imported successfully"
    ((TESTS_PASSED++))
else
    echo "  FAILED: Cannot import ALF"
    ((TESTS_FAILED++))
fi

# Test 7: dismech import
echo "[TEST 7] dismech import..."
if python3 -c "import dismech" 2>/dev/null; then
    echo "  PASSED: dismech imported successfully"
    ((TESTS_PASSED++))
else
    echo "  FAILED: Cannot import dismech"
    ((TESTS_FAILED++))
fi

echo ""
echo "============================================================"
echo "Pre-flight Summary: $TESTS_PASSED passed, $TESTS_FAILED failed"
echo "============================================================"

if [[ $TESTS_FAILED -gt 0 ]]; then
    echo "INTEGRATION TEST: FAILED - Fix errors before running full training"
    exit 1
fi

# ============================================================
# Short Training Test (100 iterations, 32 envs)
# ============================================================
echo ""
echo "============================================================"
echo "Starting short training test (100 iterations, 32 envs)..."
echo "============================================================"
echo ""

cd "${PROJECT_ROOT}/dismech-rl"

export ROOT_DIR="${PROJECT_ROOT}/results/test_integration_$(date +%Y%m%d_%H%M%S)"
export NUM_PARALLEL_ENVS=32
export NUM_ITERATIONS=100
export SUMMARY_INTERVAL=10
export NUM_CHECKPOINTS=2

python3 -m alf.bin.train \
    --conf confs/snake_approach_ppo_conf.py \
    --root_dir "$ROOT_DIR" \
    --conf_param "_CONFIG._USER.render=False" \
    --conf_param "create_environment.num_parallel_environments=${NUM_PARALLEL_ENVS}" \
    --conf_param "TrainerConfig.num_iterations=${NUM_ITERATIONS}" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.summary_interval=${SUMMARY_INTERVAL}"

TRAIN_EXIT_CODE=$?

echo ""
echo "============================================================"
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "INTEGRATION TEST: PASSED"
    echo "Training completed successfully. Ready for full training."
    echo "Results saved to: $ROOT_DIR"
else
    echo "INTEGRATION TEST: FAILED"
    echo "Training exited with code $TRAIN_EXIT_CODE"
    echo "Check error log for details."
fi
echo "============================================================"

exit $TRAIN_EXIT_CODE
