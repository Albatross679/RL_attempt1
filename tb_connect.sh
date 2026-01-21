#!/bin/bash
# tb_connect.sh - Run on OSC login node
# Prints the SSH tunnel command for your running job's TensorBoard

TB_INFO_FILE="/users/PAS3272/qifanwen/RL_attempt1/.tensorboard_info"

if [[ -f "$TB_INFO_FILE" ]]; then
    source "$TB_INFO_FILE"
    echo "TensorBoard is running for job $TB_JOB_ID"
    echo ""
    echo "Run this on your LOCAL machine:"
    echo "  ssh -N -L ${TB_PORT}:${TB_HOST}:${TB_PORT} qifanwen@pitzer.osc.edu"
    echo ""
    echo "Then open: http://localhost:${TB_PORT}"
else
    echo "No TensorBoard info found. Is a training job running?"
    squeue -u $USER
fi
