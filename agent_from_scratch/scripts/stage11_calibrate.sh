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
# The report's own C2 sees only these three shapes, so it always fails; the hard test's C2 (F2)
# counts them with the hardtest shapes levels.json already keeps (Stage 10's support_dialogue).
dry || python - outputs/s11-calib-levels.json <<'EOF'
import json, sys
from agent_from_scratch.evals.bench.skeletons import SHAPES, load_levels
chosen = set(load_levels()) | set(json.load(open(sys.argv[1])))
kept = sorted(s.name for s in SHAPES if s.split == "hardtest" and s.name in chosen)
print(f"C2 for the hard test (at least 2 kept): {kept}:", "passed" if len(kept) >= 2 else "FAILED")
EOF
