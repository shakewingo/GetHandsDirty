#!/bin/bash
# Stage 10 Session F (docs/STAGE10_HARD_TASKS_PLAN.md, K3-K4): sample M1 on the calibration split
# (every shape at every level, seeds 100-105) at 8 seeds in parallel, then choose each shape's
# level with `levels`. The chosen levels are written to outputs/, to be reviewed before they
# replace evals/bench/levels.json.
#
# Usage, from the repository root with the venv active and no server running:
#   bash agent_from_scratch/scripts/stage10_calibrate.sh ADAPTER_DIR
# Writes outputs/s10-calib-s0..7, outputs/s10-calib-report.json and outputs/s10-calib-levels.json.
set -euo pipefail
. "$(dirname "$0")/stage10_lib.sh"
ADAPTER_DIR=${1:?ADAPTER_DIR, M1 (e.g. outputs/sft-mix-v3-r1/epoch-2)}

serve "M1=$ADAPTER_DIR"
pids=() runs=()
for s in $SEEDS; do
    bench "s10-calib-s$s" calib "$s" M1 & pids+=($!)
    runs+=("outputs/s10-calib-s$s")
done
wait_all "${pids[@]}"
stop_server
run_logged "$LOGS/s10-calib-levels.log" python -m agent_from_scratch.evals levels "${runs[@]}" \
    --output outputs/s10-calib-report.json --levels outputs/s10-calib-levels.json
dry || cat outputs/s10-calib-report.json
