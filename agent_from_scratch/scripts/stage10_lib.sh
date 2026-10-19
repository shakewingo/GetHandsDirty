#!/bin/bash
# Shared pieces of the Stage 10 pod drivers (stage10_sample.sh, stage10_ei_round.sh,
# stage10_grpo.sh). Sourced, not run. Run the drivers from the repository root with the venv
# active and no server running; they start and stop the server themselves.
#
# STAGE10_DRY=1 prints each command instead of running it (no GPU, no server, no outputs).
# Design: docs/STAGE10_PLAN.md.

# M1 and the datasets M1 was trained on (Stage 9 Item 4), as the pod's outputs/ holds them.
M1_ADAPTER=${M1_ADAPTER:-outputs/sft-mix-r1/epoch-2}
TEACHER_WIDE=${TEACHER_WIDE:-outputs/stage9-item4-20261015-sft-teacher-wide}
WHEN2CALL=${WHEN2CALL:-outputs/stage9-item4-20261015-public-when2call}
TOUCAN=${TOUCAN:-outputs/stage9-item4-20261015-public-toucan}
LOGS=${LOGS:-/root/logs}
SEEDS="0 1 2 3 4 5 6 7"  # Stage 9 Item 2's sampling protocol (S2)
# Mix's long and short rows fragment the CUDA allocator (see train_stage9_adapters.sh).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SERVER_PID=""

dry() { [ "${STAGE10_DRY:-0}" = 1 ]; }
dry || mkdir -p "$LOGS"

run_logged() {  # run_logged LOG COMMAND...: run COMMAND with its output in LOG
    local log=$1; shift
    if dry; then echo "+ $* > $log"; else "$@" > "$log" 2>&1; fi
}

wait_all() {  # wait_all PID...: wait for every PID; fail if any of them failed
    local failed=0
    for pid in "$@"; do wait "$pid" || failed=1; done
    return $failed
}

serve() {  # serve NAME=DIR ...: the Stage 8 server command with these LoRA modules; waits until it answers
    if dry; then echo "+ serve $*"; return; fi
    VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup \
        vllm serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
        --dtype bfloat16 --max-model-len 32768 --enable-lora --max-lora-rank 64 --port 8000 \
        --lora-modules "$@" >> "$LOGS/vllm.log" 2>&1 < /dev/null &
    SERVER_PID=$!
    until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM exited while starting; see $LOGS/vllm.log" >&2; return 1; }
        sleep 5
    done
}

stop_server() {  # stop the server and wait until the GPU memory is free for training
    if dry; then echo "+ stop_server"; return; fi
    kill -- -"$SERVER_PID"  # setsid made it a process-group leader; its engine processes go with it
    while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 2; done
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 2; done
    SERVER_PID=""
}

bench() {  # bench OUT SPLIT SEED ADAPTER [TASKS]: one bench run against the server, logged
    local tasks=()
    [ -n "${5:-}" ] && tasks=(--tasks "$5")
    # ${a[@]+...} keeps an empty array legal under `set -u` in bash before 4.4 (macOS ships 3.2).
    run_logged "$LOGS/$1.log" python -m agent_from_scratch.evals bench --backend vllm --split "$2" --seed "$3" \
        --adapter "$4" ${tasks[@]+"${tasks[@]}"} --output "outputs/$1"
}

sample_train() {  # sample_train NAME ADAPTER: 8 seeds of the whole train split in parallel, then both counts
    local pids=() runs=()
    for s in $SEEDS; do
        bench "$1-train-s$s" train "$s" "$2" & pids+=($!)
        runs+=("outputs/$1-train-s$s")
    done
    wait_all "${pids[@]}"
    run_logged "$LOGS/$1-train-aggregate.log" python -m agent_from_scratch.evals aggregate "${runs[@]}" \
        --output "outputs/$1-train-aggregate.json"
    run_logged "$LOGS/$1-train-strict.log" python -m agent_from_scratch.evals aggregate --strict "${runs[@]}" \
        --output "outputs/$1-train-strict.json"
}

choose_epoch() {  # choose_epoch NAME: the epoch whose val seed 0 run (outputs/NAME-e<E>-s0) passed most, earlier on a tie
    if dry; then echo 1; return; fi
    python3 - "$1" <<'EOF'
import json, sys
passed = {e: json.load(open(f"outputs/{sys.argv[1]}-e{e}-s0/summary.json"))["passed"] for e in (1, 2, 3)}
print(max(passed, key=lambda e: (passed[e], -e)))
EOF
}
