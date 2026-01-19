#!/bin/bash

# ============================================================
# HRL Training Script for Snake (Manager + Workers)
# ============================================================
#
# This script trains the hierarchical RL system:
# - Manager agent learns to select skills (approach/coil)
# - Workers can be pre-trained or trained jointly
#
# USAGE:
#   # Basic HRL training (workers use random policy)
#   bash train_snake_hrl_ppo.sh
#
#   # With pre-trained workers (recommended)
#   APPROACH_CHECKPOINT=results/snake_approach_ppo_*/ckpt-* \
#   COIL_CHECKPOINT=results/snake_coil_ppo_*/ckpt-* \
#   bash train_snake_hrl_ppo.sh
#
#   # Custom iterations
#   NUM_ITERATIONS=50000 bash train_snake_hrl_ppo.sh

# -------------------- Path Detection --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/train_snake_hrl_ppo.sh"

# -------------------- Virtual Environment Activation --------------------
VENV_PATH="${REPO_ROOT}/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated environment: ${VENV_PATH}"
else
    echo "Warning: venv not found at ${VENV_PATH}"
    echo "Please run setup.sh first"
fi

# -------------------- Configurable Parameters --------------------
export ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/results/snake_hrl_ppo_$(date +%Y%m%d_%H%M%S)}"
export NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-64}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-100000}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-256}"
export HIDDEN_LAYERS="${HIDDEN_LAYERS:-"(128, 128)"}"
export NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-50}"
export RENDER="${RENDER:-False}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# PPO-specific parameters
export UTD_RATIO="${UTD_RATIO:-4}"
export UNROLL_LENGTH="${UNROLL_LENGTH:-8}"
export ENTROPY_REG="${ENTROPY_REG:-0.1}"

# HRL-specific parameters
export WORKER_STEPS="${WORKER_STEPS:-50}"         # Worker steps per manager decision
export SWITCH_BONUS="${SWITCH_BONUS:-1.0}"        # Bonus for skill switching
export COMPLETION_BONUS="${COMPLETION_BONUS:-10.0}"  # Bonus for full task completion
export MANAGER_TIMEOUT="${MANAGER_TIMEOUT:-100}"  # Manager steps before timeout

# Worker checkpoints (optional - for loading pre-trained workers)
export APPROACH_CHECKPOINT="${APPROACH_CHECKPOINT:-}"
export COIL_CHECKPOINT="${COIL_CHECKPOINT:-}"

# TensorBoard settings
export SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-20}"

# Performance optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# -------------------- Helper Functions --------------------
print_resume_command() {
    ELAPSED_SECONDS=$(($(date +%s) - TRAIN_START_TIME))
    HOURS=$((ELAPSED_SECONDS / 3600))
    MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
    SECONDS=$((ELAPSED_SECONDS % 60))

    echo ""
    echo "============================================================"
    echo "HRL Training stopped. Results saved to: $ROOT_DIR"
    echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "============================================================"
    echo ""
    echo "To RESUME training from the last checkpoint, run:"
    echo "  ROOT_DIR=$ROOT_DIR bash $SCRIPT_PATH"
    echo ""
    echo "To resume with MORE iterations:"
    echo "  ROOT_DIR=$ROOT_DIR NUM_ITERATIONS=200000 bash $SCRIPT_PATH"
    echo ""
    echo "To monitor with TensorBoard:"
    echo "  tensorboard --logdir $ROOT_DIR --port 6006"
    echo ""
}

trap print_resume_command EXIT

# -------------------- Print Configuration --------------------
echo ""
echo "============================================================"
echo "Snake HRL PPO Training Configuration"
echo "============================================================"
echo "Output directory: $ROOT_DIR"
echo "Parallel environments: $NUM_PARALLEL_ENVS"
echo "Total iterations: $NUM_ITERATIONS"
echo "Worker steps per manager step: $WORKER_STEPS"
echo "Manager timeout steps: $MANAGER_TIMEOUT"
echo ""
if [ -n "$APPROACH_CHECKPOINT" ]; then
    echo "Approach checkpoint: $APPROACH_CHECKPOINT"
else
    echo "Approach checkpoint: Not specified (using random policy)"
fi
if [ -n "$COIL_CHECKPOINT" ]; then
    echo "Coil checkpoint: $COIL_CHECKPOINT"
else
    echo "Coil checkpoint: Not specified (using random policy)"
fi
echo "============================================================"
echo ""

# -------------------- Run Training --------------------
cd "$PROJECT_ROOT"

TRAIN_START_TIME=$(date +%s)

echo "Starting Snake HRL PPO training. Monitor with:"
echo "  tensorboard --logdir $ROOT_DIR --port 6006"
echo "Then open: http://localhost:6006"
echo ""

python3 -m alf.bin.train \
    --conf confs/snake_hrl_ppo_conf.py \
    --root_dir "$ROOT_DIR" \
    --conf_param "_CONFIG._USER.render=${RENDER}" \
    --conf_param "_CONFIG._USER.utd_ratio=${UTD_RATIO}" \
    --conf_param "create_environment.num_parallel_environments=${NUM_PARALLEL_ENVS}" \
    --conf_param "TrainerConfig.num_iterations=${NUM_ITERATIONS}" \
    --conf_param "TrainerConfig.mini_batch_size=${MINI_BATCH_SIZE}" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.summary_interval=${SUMMARY_INTERVAL}" \
    --conf_param "TrainerConfig.unroll_length=${UNROLL_LENGTH}" \
    --conf_param "PPOLoss.entropy_regularization=${ENTROPY_REG}" \
    --conf_param "SnakeHRLEnv.worker_steps_per_manager_step=${WORKER_STEPS}" \
    --conf_param "SnakeHRLEnv.switch_bonus=${SWITCH_BONUS}" \
    --conf_param "SnakeHRLEnv.completion_bonus=${COMPLETION_BONUS}" \
    --conf_param "SnakeHRLEnv.timeout_manager_steps=${MANAGER_TIMEOUT}"
