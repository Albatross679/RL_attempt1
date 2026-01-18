#!/bin/bash

# ============================================================
# Policy Evaluation Script
# ============================================================

# -------------------- Path Detection --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# -------------------- Virtual Environment Activation --------------------
VENV_PATH="${REPO_ROOT}/.venv312"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated Python 3.12 environment: ${VENV_PATH}"
else
    echo "Warning: Python 3.12 venv not found at ${VENV_PATH}"
    echo "Please create it with: python3.12 -m venv ${VENV_PATH}"
fi

POLICY_DIR="${1:-$PROJECT_ROOT/results/follow_sac_latest}"
CHECKPOINT="${2:-latest}"      # latest, best, or step number
NUM_EPISODES="${NUM_EPISODES:-10}"
RECORD_VIDEO="${RECORD_VIDEO:-False}"
VIDEO_FILE="${VIDEO_FILE:-eval_video.mp4}"

cd "$PROJECT_ROOT"

CMD="python3 -m alf.bin.play \
    --root_dir $POLICY_DIR \
    --checkpoint_step $CHECKPOINT \
    --num_episodes $NUM_EPISODES \
    --conf_param render=True"

if [ "$RECORD_VIDEO" = "True" ]; then
    CMD="$CMD --record_file $VIDEO_FILE"
fi

eval $CMD
