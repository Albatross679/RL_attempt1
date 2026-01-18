#!/bin/bash

# ============================================================
# SAC Training Script for Follow Task - DisMech Framework
# ============================================================

# -------------------- Path Detection --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/train_follow_sac.sh"

# USAGE:
#   Navigate to script directory and run:
#     source .venv/bin/activate  # from project root
#     cd dismech-rl/scripts
#     bash train_follow_sac.sh
#
#   With custom parameters:
#     NUM_ITERATIONS=1000 LEARNING_RATE=2e-3 bash train_follow_sac.sh
#
#   Run in background with logging:
#     nohup bash train_follow_sac.sh > train.log 2>&1 &
#
#   Custom workspace dimension (2D or 3D):
#     WS_DIM=2 bash train_follow_sac.sh
#
#   Use Elastica simulator instead of DisMech:
#     SIM_FRAMEWORK=elastica bash train_follow_sac.sh
#
#   Monitor training with TensorBoard (in separate terminal):
#     tensorboard --logdir $PROJECT_ROOT/results --port 6006
#   Then open browser to http://localhost:6006

# -------------------- Configurable Parameters --------------------
export ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/results/follow_sac_$(date +%Y%m%d_%H%M%S)}"
export SIM_FRAMEWORK="${SIM_FRAMEWORK:-dismech}"    # dismech or elastica
export WS_DIM="${WS_DIM:-3}"                        # 2 or 3 (workspace dimension)
export NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-300}"  # Reduced from 500 - pure Python dismech uses ~10-100x more memory
export NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2048}"
export HIDDEN_LAYERS="${HIDDEN_LAYERS:-"(256, 256, 256)"}"
export UTD_RATIO="${UTD_RATIO:-8}"
export REPLAY_BUFFER_LENGTH="${REPLAY_BUFFER_LENGTH:-4000}"
export NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
export INITIAL_COLLECT_STEPS="${INITIAL_COLLECT_STEPS:-1000}"
export RENDER="${RENDER:-False}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# TensorBoard settings
export SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-50}"

# Performance optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Required for CUDA determinism

# -------------------- Helper Functions --------------------
print_resume_command() {
    # Calculate elapsed time
    ELAPSED_SECONDS=$(($(date +%s) - TRAIN_START_TIME))
    HOURS=$((ELAPSED_SECONDS / 3600))
    MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
    SECONDS=$((ELAPSED_SECONDS % 60))
    
    echo ""
    echo "============================================================"
    echo "Training stopped. Results saved to: $ROOT_DIR"
    echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "============================================================"
    echo ""
    echo "To RESUME training from the last checkpoint, run:"
    echo "  ROOT_DIR=$ROOT_DIR bash $SCRIPT_PATH"
    echo ""
    echo "To resume with MORE iterations (e.g., 10000 total):"
    echo "  ROOT_DIR=$ROOT_DIR NUM_ITERATIONS=10000 bash $SCRIPT_PATH"
    echo ""
    echo "To monitor with TensorBoard:"
    echo "  tensorboard --logdir $ROOT_DIR --port 6006"
    echo ""
}

# Trap signals to print resume command on exit
trap print_resume_command EXIT

# -------------------- Run Training --------------------
cd "$PROJECT_ROOT"

# Capture training start time
TRAIN_START_TIME=$(date +%s)

echo ""
echo "Starting training. Monitor this run with:"
echo "  tensorboard --logdir $ROOT_DIR --port 6006"
echo "Then open: http://localhost:6006"
echo ""

python3 -m alf.bin.train \
    --conf confs/follow_conf.py \
    --root_dir "$ROOT_DIR" \
    --conf_param "_CONFIG._USER.sim_framework='${SIM_FRAMEWORK}'" \
    --conf_param "_CONFIG._USER.ws_dim=${WS_DIM}" \
    --conf_param "_CONFIG._USER.render=${RENDER}" \
    --conf_param "_CONFIG._USER.utd_ratio=${UTD_RATIO}" \
    --conf_param "create_environment.num_parallel_environments=${NUM_PARALLEL_ENVS}" \
    --conf_param "TrainerConfig.num_iterations=${NUM_ITERATIONS}" \
    --conf_param "TrainerConfig.mini_batch_size=${MINI_BATCH_SIZE}" \
    --conf_param "TrainerConfig.replay_buffer_length=${REPLAY_BUFFER_LENGTH}" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.summary_interval=${SUMMARY_INTERVAL}" \
    --conf_param "TrainerConfig.initial_collect_steps=${INITIAL_COLLECT_STEPS}" \
    --conf_param "get_alg_constructor.hidden_layers=${HIDDEN_LAYERS}"
    # Note: Learning rate is set in common_sac_training_conf.py (default 1e-3)
