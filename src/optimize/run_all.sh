#!/usr/bin/env bash
# src/optimize/run_all.sh
#
# Runs the parameter optimizer for every supported strategy.
# Must be executed from the repository root directory.
#
# Usage:
#   bash src/optimize/run_all.sh [TRIALS] [METHOD] [EVAL_RUNS]
#
# Arguments (all optional):
#   TRIALS    Number of valid simulation trials per strategy  (default: 50)
#   METHOD    Optimization method: random | optuna | grid     (default: random)
#   EVAL_RUNS Pointing-offset scenarios evaluated per trial   (default: 30)
#
# Examples:
#   bash src/optimize/run_all.sh
#   bash src/optimize/run_all.sh 100 optuna 50
#   bash src/optimize/run_all.sh 20 grid

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRIALS="${1:-50}"
METHOD="${2:-random}"
EVAL_RUNS="${3:-30}"

OPTIMIZER="src/optimize/__main__.py"
SIM_CONFIG="config/Simulation.toml"
MC_CONFIG="config/MonteCarlo.toml"
LOG_DIR="src/optimize/logs"

STRATEGIES=(
    # dual_spiral
    # dual_raster
    lissajous_scan
    rosette_scan
    # concentric_shells
    # random_curve
    center_rebias
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ ! -f "$OPTIMIZER" ]]; then
    echo "ERROR: $OPTIMIZER not found. Run this script from the repository root." >&2
    exit 1
fi

if [[ ! -f "$SIM_CONFIG" ]]; then
    echo "ERROR: $SIM_CONFIG not found. Run this script from the repository root." >&2
    exit 1
fi

if [[ ! -f "$MC_CONFIG" ]]; then
    echo "ERROR: $MC_CONFIG not found. Run this script from the repository root." >&2
    exit 1
fi

case "$METHOD" in
    random|optuna|grid) ;;
    *)
        echo "ERROR: Unknown method '$METHOD'. Choose: random | optuna | grid" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR"

TOTAL=${#STRATEGIES[@]}
START_TIME=$(date +%s)

echo "========================================================"
echo " Strategy Optimizer — run_all.sh"
echo "========================================================"
echo " Strategies : ${TOTAL}"
echo " Method     : ${METHOD}"
echo " Trials     : ${TRIALS}"
echo " Eval runs  : ${EVAL_RUNS}"
echo " Sim config : ${SIM_CONFIG}"
echo " MC config  : ${MC_CONFIG}"
echo " Log dir    : ${LOG_DIR}"
echo " Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

FAILED=()

for i in "${!STRATEGIES[@]}"; do
    STRATEGY="${STRATEGIES[$i]}"
    IDX=$((i + 1))
    LOG_FILE="${LOG_DIR}/${STRATEGY}.log"

    echo "--------------------------------------------------------"
    echo " [$IDX/$TOTAL] Optimizing: $STRATEGY"
    echo " Log: $LOG_FILE"
    echo "--------------------------------------------------------"

    STRATEGY_START=$(date +%s)

    if PYTHONPATH=src python -m optimize \
            --strategy   "$STRATEGY" \
            --method     "$METHOD" \
            --trials     "$TRIALS" \
            --eval-runs  "$EVAL_RUNS" \
            --sim-config "$SIM_CONFIG" \
            --mc-config  "$MC_CONFIG" \
        2>&1 | tee "$LOG_FILE"; then

        STRATEGY_END=$(date +%s)
        ELAPSED=$(( STRATEGY_END - STRATEGY_START ))
        echo "  ✓ Completed in ${ELAPSED}s"
    else
        STRATEGY_END=$(date +%s)
        ELAPSED=$(( STRATEGY_END - STRATEGY_START ))
        echo "  ✗ FAILED after ${ELAPSED}s (see $LOG_FILE)" >&2
        FAILED+=("$STRATEGY")
    fi

    echo
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))

echo "========================================================"
echo " Summary"
echo "========================================================"
printf " Total time : %dm %ds\n" $(( TOTAL_ELAPSED / 60 )) $(( TOTAL_ELAPSED % 60 ))

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo " All ${TOTAL} strategies completed successfully."
else
    echo " ${#FAILED[@]}/${TOTAL} strategies FAILED:"
    for s in "${FAILED[@]}"; do
        echo "   - $s  (log: ${LOG_DIR}/${s}.log)"
    done
fi

echo
echo " Individual logs saved to: ${LOG_DIR}/"
echo "========================================================"

# Exit non-zero if any strategy failed
[[ ${#FAILED[@]} -eq 0 ]]
