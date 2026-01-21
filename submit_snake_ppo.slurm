#!/bin/bash
#=============================================================================
# USAGE:
#   sbatch submit_snake_ppo.slurm       # Submit this job to the queue
#
# FIND YOUR ACCOUNT (required for --account):
#   sacctmgr show associations user=$USER format=account%20 --noheader
#
# CHECK JOB STATUS:
#   squeue -u qifanwen                     # View all your jobs in the queue
#   squeue -j <job_id>                     # Check status of a specific job
#   sacct -j <job_id>                      # View job accounting info (after completion)
#   scancel <job_id>                       # Cancel a running/pending job
#
# CHECK HARDWARE RESOURCES:
#   # View job with resource allocation (CPUs, GPUs, memory, node)
#   squeue -u qifanwen -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %.4C %.15R %b"
#
#   # Detailed job info (GPU type, memory allocated, time limit, node features)
#   scontrol show job <job_id>
#
#   # View GPU partition hardware specs (all nodes, GPU models, memory per node)
#   sinfo -p gpu -o "%P %N %G %c %m %f"
#
#   # Check GPU utilization on your node (run from within job or ssh to node)
#   nvidia-smi
#
# OSC PITZER GPU PARTITION SPECS:
#   - GPU: NVIDIA Tesla V100-PCIE-16GB (2 per node)
#   - CPU: 40 cores per node (Intel Xeon)
#   - RAM: ~363 GB per node
#   - Nodes: p0225-p0256 (32 GPU nodes total)
#
# VIEW OUTPUT & ERRORS:
#   Output file: slurm_logs/snake_ppo_<job_id>.out
#   Error file:  slurm_logs/snake_ppo_<job_id>.err
#
#   ls slurm_logs/snake_ppo_*.out       # List all output files
#   cat slurm_logs/snake_ppo_<job_id>.out   # View full output
#   tail -f slurm_logs/snake_ppo_<job_id>.out  # Follow output in real-time (while running)
#   cat slurm_logs/snake_ppo_<job_id>.err   # View error log
#   tail -20 slurm_logs/snake_ppo_<job_id>.err  # View last 20 lines of errors
#=============================================================================

#SBATCH --job-name=snake_ppo           # Job name shown in squeue
#SBATCH --account=PAS3272              # Project/account to charge
#SBATCH --partition=gpu                # Use GPU partition
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Run 1 task
#SBATCH --cpus-per-task=8              # 8 CPU cores per task
#SBATCH --gpus-per-node=1              # Request 1 GPU
#SBATCH --time=04:00:00                # Max runtime: 4 hours
#SBATCH --output=slurm_logs/snake_ppo_%j.out  # Stdout file (%j = job ID)
#SBATCH --error=slurm_logs/snake_ppo_%j.err   # Stderr file (%j = job ID)

# ============================================================
# Environment Setup
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
# This ensures ninja and other tools are found by subprocesses (e.g., PyTorch JIT)
export PATH="${VENV_PATH}/bin:${PATH}"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Required for PyTorch deterministic algorithms with cuBLAS on CUDA >= 10.2
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================
# Pre-flight Checks
# ============================================================
echo "Environment verification..."
echo "  Python: $(which python3) - $(python3 --version)"
echo "  Ninja: $(which ninja) - $(ninja --version)"
echo "  CC: ${CC} ($(which ${CC}))"
echo "  CXX: ${CXX} ($(which ${CXX}))"

# Verify ninja works in subprocess (matches PyTorch's check)
python3 -c "import subprocess; subprocess.check_output('ninja --version'.split()); print('  Ninja subprocess check: PASSED')"

# Verify GPU is accessible
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

# ============================================================
# TensorBoard Server (Background)
# ============================================================
TB_PORT=$((6000 + (SLURM_JOB_ID % 1000)))
TB_HOST=$(hostname)
TB_IP=$(hostname -i)
TB_INFO_FILE="${PROJECT_ROOT}/.tensorboard_info"
TB_LOG_DIR="${PROJECT_ROOT}/dismech-rl/results"

# Write connection info for external scripts
cat > "$TB_INFO_FILE" << EOF
TB_JOB_ID=$SLURM_JOB_ID
TB_HOST=$TB_HOST
TB_IP=$TB_IP
TB_PORT=$TB_PORT
TB_LOG_DIR=$TB_LOG_DIR
EOF

echo "=============================================="
echo "TensorBoard Server Starting"
echo "=============================================="
echo "Node: $TB_HOST ($TB_IP)"
echo "Port: $TB_PORT"
echo "Log Dir: $TB_LOG_DIR"
echo ""
echo "To connect from your LOCAL machine, run:"
echo "  ssh -N -L ${TB_PORT}:${TB_HOST}:${TB_PORT} qifanwen@pitzer.osc.edu"
echo "Then open: http://localhost:${TB_PORT}"
echo "=============================================="

# Start TensorBoard in background
tensorboard --logdir="$TB_LOG_DIR" --port=$TB_PORT --bind_all 2>&1 | head -5 &
TB_PID=$!
sleep 3  # Give TensorBoard time to start

# Cleanup function
cleanup_tensorboard() {
    echo "Stopping TensorBoard (PID: $TB_PID)..."
    kill $TB_PID 2>/dev/null
    rm -f "$TB_INFO_FILE"
}
trap cleanup_tensorboard EXIT

# Run training
cd /users/PAS3272/qifanwen/RL_attempt1/dismech-rl/scripts
bash train_snake_approach_ppo.sh
