#!/bin/bash
# Stage 10's GRPO pilot (decisions G2-G6 of docs/STAGE10_PLAN.md). Starting from M1, each
# iteration serves the current adapter, samples its 42 train tasks at 8 seeds, stops the server
# and takes one `grpo step`; ten iterations make a pass. After each pass: the adapter is kept as
# outputs/PREFIX-pass-P, val seed 0 runs, and G6 may stop the pilot.
#
# Usage, from the repository root with the venv active and no server running:
#   bash agent_from_scratch/scripts/stage10_grpo.sh PASSES
# Environment: STAGE10_PREFIX (s10-grpo), STAGE10_ITERATIONS (10), STAGE10_VAL (1), STAGE10_LR
# (1e-5), STAGE10_START (M1's adapter). Session A's smoke: STAGE10_PREFIX=s10-grpo-smoke
# STAGE10_ITERATIONS=1 STAGE10_VAL=0 ... stage10_grpo.sh 1.
# Resumable: an iteration whose step directory exists is not run again. An iteration interrupted
# before its step leaves sample directories that `bench` will not overwrite; move them aside first.
set -euo pipefail
. "$(dirname "$0")/stage10_lib.sh"
PASSES=${1:?PASSES, at most 3}
PREFIX=${STAGE10_PREFIX:-s10-grpo}
ITERATIONS=${STAGE10_ITERATIONS:-10}
VAL=${STAGE10_VAL:-1}
LR=${STAGE10_LR:-1e-5}
current=${STAGE10_START:-$M1_ADAPTER}

val_seed0() {  # val_seed0 NAME: val seed 0 of the current adapter, unless it has run already
    [ -d "outputs/$1" ] && return
    serve "grpo=$current"
    bench "$1" val 0 grpo
    stop_server
}

[ "$VAL" = 1 ] && val_seed0 "$PREFIX-val-p0-s0"  # G6's reference: M1 on this server
for p in $(seq 1 "$PASSES"); do
    for i in $(seq 0 $((ITERATIONS - 1))); do
        step_dir="outputs/$PREFIX-p$p-i$i-step"
        if [ -d "$step_dir" ]; then current=$step_dir; continue; fi
        if dry && [ -n "${STAGE10_POOL:-}" ]; then
            tasks="<pass $p iteration $i of the pool>"
        else
            # shellcheck disable=SC2046
            tasks=$(python -m agent_from_scratch.evals grpo tasks --pass "$p" --iteration "$i" $(pool_args))
        fi
        serve "grpo=$current"
        pids=() runs=()
        for s in $SEEDS; do
            bench "$PREFIX-p$p-i$i-s$s" train "$s" grpo "$tasks" & pids+=($!)
            runs+=("outputs/$PREFIX-p$p-i$i-s$s")
        done
        wait_all "${pids[@]}"
        stop_server  # G5: the server and the trainer take turns on the GPU
        run_logged "$LOGS/$PREFIX-p$p-i$i-step.log" python -m agent_from_scratch.evals grpo step \
            --runs "${runs[@]}" --adapter-in "$current" --output "$step_dir" --lr "$LR"
        current=$step_dir
    done
    [ "$VAL" = 1 ] || continue
    [ -d "outputs/$PREFIX-pass-$p" ] || run_logged /dev/null cp -r "$current" "outputs/$PREFIX-pass-$p"
    val_seed0 "$PREFIX-val-p$p-s0"
    dry && continue
    # G6: stop when val seed 0 falls 10 or more tasks below M1's on this server, or the pass's
    # mean train reward falls 0.10 or more below M1's in Session A.
    if ! python3 - "$PREFIX" "$p" "$ITERATIONS" <<'EOF'
import json, sys
prefix, p, iterations = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def passed(name):
    return json.load(open(f"outputs/{name}/summary.json"))["passed"]

reference, now = passed(f"{prefix}-val-p0-s0"), passed(f"{prefix}-val-p{p}-s0")
groups = [json.load(open(f"outputs/{prefix}-p{p}-i{i}-step/step.json"))["groups"] for i in range(iterations)]
reward = sum(g["mean_reward"] * g["rollouts"] for g in groups) / sum(g["rollouts"] for g in groups)
try:
    m1 = json.load(open("outputs/s10-m1-train-strict.json"))["overall"]["mean_pass@1"]
except FileNotFoundError:
    m1 = None
print(f"pass {p}: val seed 0 passed {now} (M1 {reference}); mean train reward {reward:.3f} (M1 {m1})")
sys.exit(1 if now <= reference - 10 or (m1 is not None and reward <= m1 - 0.10) else 0)
EOF
    then
        echo "G6: the pilot stops after pass $p"
        break
    fi
done
echo "GRPO done: last adapter $current"
