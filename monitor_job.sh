#!/bin/bash
#=============================================================================
# Job Monitor Script
#
# USAGE:
#   ./monitor_job.sh              # Monitor most recent job
#   ./monitor_job.sh <job_id>     # Monitor specific job
#   watch -n 30 ./monitor_job.sh  # Auto-refresh every 30 seconds
#
# FEATURES:
#   - Auto-detects known error patterns
#   - Links to troubleshooting documentation
#   - Shows job status and recent output
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get job ID
if [[ -n "$1" ]]; then
    JOB_ID="$1"
else
    # Find most recent job from log files
    LATEST_LOG=$(ls -t "${LOG_DIR}"/snake_ppo_*.out "${LOG_DIR}"/test_integration_*.out 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        JOB_ID=$(basename "$LATEST_LOG" | grep -oP '\d+(?=\.out)')
    fi
fi

if [[ -z "$JOB_ID" ]]; then
    echo "No job ID provided and no recent logs found."
    echo "Usage: ./monitor_job.sh [job_id]"
    exit 1
fi

# Find log files
OUT_FILE=$(ls "${LOG_DIR}"/*_${JOB_ID}.out 2>/dev/null | head -1)
ERR_FILE=$(ls "${LOG_DIR}"/*_${JOB_ID}.err 2>/dev/null | head -1)

echo "============================================================"
echo -e "${BLUE}Job Monitor - Job ID: ${JOB_ID}${NC}"
echo "============================================================"
echo ""

# Check job status via squeue
JOB_INFO=$(squeue -j "$JOB_ID" -h -o "%T|%M|%N|%r" 2>/dev/null)

if [[ -n "$JOB_INFO" ]]; then
    IFS='|' read -r STATE TIME NODE REASON <<< "$JOB_INFO"
    echo -e "${GREEN}Status: $STATE${NC}"
    echo "  Runtime: $TIME"
    echo "  Node: ${NODE:-pending}"
    [[ -n "$REASON" && "$REASON" != "None" ]] && echo "  Reason: $REASON"
else
    # Job not in queue, check sacct
    SACCT_INFO=$(sacct -j "$JOB_ID" -n -o State,ExitCode,Elapsed --parsable2 2>/dev/null | head -1)
    if [[ -n "$SACCT_INFO" ]]; then
        IFS='|' read -r STATE EXIT_CODE ELAPSED <<< "$SACCT_INFO"
        if [[ "$STATE" == "COMPLETED" ]]; then
            echo -e "${GREEN}Status: COMPLETED${NC}"
        elif [[ "$STATE" == "FAILED" || "$STATE" == "CANCELLED" ]]; then
            echo -e "${RED}Status: $STATE (Exit: $EXIT_CODE)${NC}"
        else
            echo -e "${YELLOW}Status: $STATE${NC}"
        fi
        echo "  Total Runtime: $ELAPSED"
    else
        echo -e "${YELLOW}Status: Unknown (job not found in queue or history)${NC}"
    fi
fi

echo ""

# Error pattern detection
check_errors() {
    local file="$1"
    local found_error=0

    if [[ ! -f "$file" ]]; then
        return 1
    fi

    # Pattern: Ninja not found
    if grep -q "Ninja is required" "$file" 2>/dev/null; then
        echo -e "${RED}[ERROR DETECTED] Ninja is required to load C++ extensions${NC}"
        echo "  See: SLURM_Issue_Doc.md#issue-001-ninja-path-not-found"
        echo "  Fix: Ensure PATH export is before venv activation in SLURM script"
        found_error=1
    fi

    # Pattern: ICC deprecation
    if grep -q "icc: command line warning" "$file" 2>/dev/null; then
        echo -e "${YELLOW}[WARNING] ICC deprecation warning detected${NC}"
        echo "  See: SLURM_Issue_Doc.md#issue-002-icc-deprecation-warning"
        echo "  Fix: Load gcc module and set CC/CXX environment variables"
        found_error=1
    fi

    # Pattern: CUDA OOM
    if grep -q "CUDA out of memory" "$file" 2>/dev/null; then
        echo -e "${RED}[ERROR DETECTED] CUDA out of memory${NC}"
        echo "  See: SLURM_Issue_Doc.md#issue-004-cuda-out-of-memory"
        echo "  Fix: Reduce NUM_PARALLEL_ENVS or MINI_BATCH_SIZE"
        found_error=1
    fi

    # Pattern: Module not found
    if grep -q "ModuleNotFoundError" "$file" 2>/dev/null; then
        MODULE=$(grep "ModuleNotFoundError" "$file" | head -1 | grep -oP "No module named '\K[^']+")
        echo -e "${RED}[ERROR DETECTED] Module not found: $MODULE${NC}"
        echo "  See: SLURM_Issue_Doc.md#issue-005-module-import-errors"
        echo "  Fix: Check venv activation or re-run setup.sh"
        found_error=1
    fi

    # Pattern: Python exception
    if grep -qE "^(Traceback|Error:|Exception:)" "$file" 2>/dev/null; then
        if [[ $found_error -eq 0 ]]; then
            echo -e "${RED}[ERROR DETECTED] Python exception${NC}"
            echo "  Check error log for details: $file"
            found_error=1
        fi
    fi

    return $found_error
}

# Check for errors
echo "--- Error Detection ---"
ERRORS_FOUND=0
if [[ -f "$ERR_FILE" ]]; then
    check_errors "$ERR_FILE"
    ERRORS_FOUND=$?
fi
if [[ -f "$OUT_FILE" ]]; then
    check_errors "$OUT_FILE"
    [[ $? -eq 0 ]] && ERRORS_FOUND=0
fi

if [[ $ERRORS_FOUND -eq 1 ]]; then
    echo ""
else
    echo -e "${GREEN}No known error patterns detected${NC}"
    echo ""
fi

# Show recent output
if [[ -f "$OUT_FILE" ]]; then
    echo "--- Recent Output (last 15 lines) ---"
    tail -15 "$OUT_FILE"
    echo ""
fi

# Show recent errors (if any substantive content)
if [[ -f "$ERR_FILE" ]]; then
    ERR_LINES=$(wc -l < "$ERR_FILE")
    if [[ $ERR_LINES -gt 0 ]]; then
        # Filter out just warnings, show actual errors
        ACTUAL_ERRORS=$(grep -v "WARNING\|warning\|Hparam value cannot" "$ERR_FILE" | tail -10)
        if [[ -n "$ACTUAL_ERRORS" ]]; then
            echo "--- Recent Errors (last 10 lines, excluding warnings) ---"
            echo "$ACTUAL_ERRORS"
            echo ""
        fi
    fi
fi

# Training progress detection
if [[ -f "$OUT_FILE" ]]; then
    ITER_LINE=$(grep -oP "iter = \d+" "$OUT_FILE" | tail -1)
    if [[ -n "$ITER_LINE" ]]; then
        echo "--- Training Progress ---"
        echo "  Current: $ITER_LINE"

        # Try to get total iterations from config
        TOTAL_ITER=$(grep -oP "num_iterations=\d+" "$OUT_FILE" | head -1 | grep -oP "\d+")
        if [[ -n "$TOTAL_ITER" ]]; then
            CURRENT=$(echo "$ITER_LINE" | grep -oP "\d+")
            PERCENT=$((CURRENT * 100 / TOTAL_ITER))
            echo "  Total: $TOTAL_ITER iterations"
            echo "  Progress: ${PERCENT}%"
        fi
        echo ""
    fi
fi

echo "============================================================"
echo "Log files:"
echo "  Output: ${OUT_FILE:-not found}"
echo "  Errors: ${ERR_FILE:-not found}"
echo ""
echo "Commands:"
echo "  Full output: cat $OUT_FILE"
echo "  Full errors: cat $ERR_FILE"
echo "  Follow live: tail -f $OUT_FILE"
echo "============================================================"
