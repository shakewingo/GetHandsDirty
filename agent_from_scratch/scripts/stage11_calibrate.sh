#!/bin/bash
# Stage 11 Session H (docs/STAGE11_PLAN.md, F2): calibrate M1 (mix-v3) on the three new hard-test
# shapes, every level, calibration seeds only, 8 samples each; then `levels` chooses their levels
# by C1. The chosen levels are written to outputs/ to be merged into levels.json after review.
#
# Usage, from the repository root with the venv active and no server running:
#   bash agent_from_scratch/scripts/stage11_calibrate.sh ADAPTER_DIR
# Writes outputs/s11-calib-s0..7, outputs/s11-calib-report.json and outputs/s11-calib-levels.json.
set -euo pipefail
. "$(dirname "$0")/stage10_lib.sh"
ADAPTER_DIR=${1:?ADAPTER_DIR, M1 (mix-v3)}
SHAPES_RE="^(config_patch_plan|shipping_fee_table|ledger_audit_fault)@"

tasks=$(python -c "
import re
from agent_from_scratch.evals.bench.run import specs
print(','.join(c.id for s, c in specs('calib') if re.match('$SHAPES_RE', c.name)))")
serve "M1=$ADAPTER_DIR"
pids=() runs=()
for s in $SEEDS; do
    bench "s11-calib-s$s" calib "$s" M1 "$tasks" & pids+=($!)
    runs+=("outputs/s11-calib-s$s")
done
wait_all "${pids[@]}"
stop_server
run_logged "$LOGS/s11-calib-levels.log" python -m agent_from_scratch.evals levels "${runs[@]}" \
    --output outputs/s11-calib-report.json --levels outputs/s11-calib-levels.json
dry || cat outputs/s11-calib-levels.json
