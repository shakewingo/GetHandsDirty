#!/bin/bash
# Stage 10 expert iteration, round K (decisions E1-E3 of docs/STAGE10_PLAN.md):
#   M_K's verified train rollouts (export --track rft, teacher-wide's settings)
#   + M1's training data (teacher-wide x3, When2Call, Toucan), retrained from the base -> M_{K+1};
#   its epoch chosen on val seed 0, val seeds 1-2 for that epoch, then M_{K+1} sampled on the
#   train split for round K+1.
#
# Usage, from the repository root with the venv active and no server running:
#   bash agent_from_scratch/scripts/stage10_ei_round.sh K
# Needs outputs/s10-mK-train-s0..7 (Session A writes s10-m1-train-*; round K writes s10-m{K+1}-train-*).
# Writes outputs/s10-rft-mK, s10-sft-eiK, s10-m{K+1}-r1 (three epochs), s10-val-m{K+1}-*,
# s10-m{K+1} (the chosen epoch's adapter) and s10-m{K+1}-train-*.
set -euo pipefail
. "$(dirname "$0")/stage10_lib.sh"
K=${1:?round K, in which M_K generates the data for the next model}
NEXT=$((K + 1))

runs=()
for s in $SEEDS; do runs+=("outputs/s10-m$K-train-s$s"); done
run_logged "$LOGS/s10-rft-m$K.log" python -m agent_from_scratch.evals export --track rft \
    --max-per-task 1 --flawed-turns drop --output "outputs/s10-rft-m$K" "${runs[@]}"
run_logged "$LOGS/s10-sft-ei$K.log" python -m agent_from_scratch.evals mix --own "$TEACHER_WIDE" \
    --public "$WHEN2CALL" "$TOUCAN" --self "outputs/s10-rft-m$K" --own-share "$OWN_SHARE" --seed 0 \
    --output "outputs/s10-sft-ei$K"
run_logged "$LOGS/s10-m$NEXT-train.log" python -m agent_from_scratch.evals train \
    --examples "outputs/s10-sft-ei$K" --output "outputs/s10-m$NEXT-r1"

serve "m$NEXT-e1=outputs/s10-m$NEXT-r1/epoch-1" "m$NEXT-e2=outputs/s10-m$NEXT-r1/epoch-2" \
      "m$NEXT-e3=outputs/s10-m$NEXT-r1/epoch-3"
pids=()
for e in 1 2 3; do bench "s10-val-m$NEXT-e$e-s0" val 0 "m$NEXT-e$e" & pids+=($!); done
wait_all "${pids[@]}"
E=$(choose_epoch "s10-val-m$NEXT")
dry || echo "$E" > "outputs/s10-m$NEXT-epoch.txt"
pids=()
for s in 1 2; do bench "s10-val-m$NEXT-e$E-s$s" val "$s" "m$NEXT-e$E" & pids+=($!); done
wait_all "${pids[@]}"
sample_train "s10-m$NEXT" "m$NEXT-e$E"
stop_server
run_logged /dev/null cp -r "outputs/s10-m$NEXT-r1/epoch-$E" "outputs/s10-m$NEXT"
echo "round $K done: M$NEXT is epoch $E of outputs/s10-m$NEXT-r1"
