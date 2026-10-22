#!/bin/bash
# Stage 11 Session I (docs/STAGE11_PLAN.md, F1-F5): the final runs, each arm once, on one server.
#   1. A new freeze (`dataset --check`); its 140 test tasks must match Stage 9's freeze, or stop.
#   2. One server: the base plus teacher-v3, mix, mix-v2, mix-v3 and M4.
#   3. Test: mix-v2, mix-v3, M4 (Stage 9's base, teacher-v3 and mix stand as reported).
#   4. Hard test: all six arms.
#   5. External: When2Call MCQ and replies for the judge, and BFCL, for mix-v2, mix-v3, M4.
# Seeds 0-2, `--final --manifest` for test and the hard test. Each stage is packed into
# $PACK/s11-STAGE.tgz and marked $PACK/s11-STAGE.done, and skipped when its marker exists.
#
# Usage, from the repository root with the venv active and no server running:
#   bash agent_from_scratch/scripts/stage11_session.sh
# Environment: the adapter directories TEACHER_V3, MIX, MIX_V2, MIX_V3, M4; STAGE9_MANIFEST (Stage 9's
# freeze); W2C_DIR (When2Call's test files); BFCL_ROOT (gorilla's berkeley-function-call-leaderboard);
# PACK (/root).
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
. "$S/stage10_lib.sh"
PACK=${PACK:-/root}
DATA=outputs/s11-data
MANIFEST=$DATA/manifest.json
NEW=(mix-v2 mix-v3 m4)
ARMS=(base teacher-v3 mix mix-v2 mix-v3 m4)

stamp() { if dry; then echo "# $*"; else echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/timeline.txt"; fi; }
finished() { [ -f "outputs/s11-$1.done" ]; }
finish() {  # finish STAGE PATH...: mark the stage done and pack it for the copy-back
    local stage=$1
    shift
    dry && { echo "+ finish $stage"; return; }
    touch "outputs/s11-$stage.done"
    tar czf "$PACK/s11-$stage.tgz" "$@" "$LOGS"
    touch "$PACK/s11-$stage.done"
}
final_run() {  # final_run OUT SPLIT SEED ARM: one frozen-protocol run (the base has no adapter)
    local adapter=()
    [ "$4" != base ] && adapter=(--adapter "$4")
    run_logged "$LOGS/$1.log" python -m agent_from_scratch.evals bench --backend vllm --split "$2" --final \
        --manifest "$MANIFEST" --seed "$3" ${adapter[@]+"${adapter[@]}"} --output "outputs/$1"
}

if ! finished freeze; then
    stamp "I: freeze"
    # The freeze hashes the working tree's sources, so it must be the committed tree (F4).
    dry || [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "Uncommitted changes; freeze from a clean clone" >&2; exit 2; }
    run_logged "$LOGS/dataset.log" python -m agent_from_scratch.evals dataset --output "$DATA" --check
    dry || grep -q '"gates_passed": true' "$DATA/summary.json"
    dry || python - "$MANIFEST" "${STAGE9_MANIFEST:?Stage 9 freeze}" <<'EOF'
import json, sys
FIELDS = ("prompt_sha256", "expect_sha256", "fault_sha256", "tools_sha256", "workspace_sha256")
new, old = (json.load(open(path)) for path in sys.argv[1:3])
test = lambda m: {t["id"]: tuple(t[f] for f in FIELDS) for t in m["tasks"] if t["split"] == "test"}
a, b = test(new), test(old)
if a != b:
    sys.exit(f"Test tasks differ from Stage 9's freeze: {sorted(set(a) ^ set(b)) or [i for i in a if a[i] != b.get(i)]}")
print(f"{len(a)} test tasks match Stage 9's freeze;",
      sum(t["split"] == "hardtest" for t in new["tasks"]), "hard-test tasks frozen")
EOF
    finish freeze "$MANIFEST" "$DATA/summary.json" "$DATA/metadata.json"
fi

serve "teacher-v3=${TEACHER_V3:?}" "mix=${MIX:?}" "mix-v2=${MIX_V2:?}" "mix-v3=${MIX_V3:?}" "m4=${M4:?}"

if ! finished test; then
    stamp "I: test"
    pids=()
    for arm in "${NEW[@]}"; do
        for s in 0 1 2; do final_run "s11-test-$arm-s$s" test "$s" "$arm" & pids+=($!); done
    done
    wait_all "${pids[@]}"
    finish test outputs/s11-test-*
fi

if ! finished hardtest; then
    stamp "I: hard test"
    for half in "base teacher-v3 mix" "mix-v2 mix-v3 m4"; do
        pids=()
        for arm in $half; do
            for s in 0 1 2; do final_run "s11-hard-$arm-s$s" hardtest "$s" "$arm" & pids+=($!); done
        done
        wait_all "${pids[@]}"
    done
    finish hardtest outputs/s11-hard-*
fi

if ! finished external; then
    stamp "I: external"
    for arm in "${NEW[@]}"; do
        pids=()
        run_logged "$LOGS/ext-mcq-$arm.log" python -m agent_from_scratch.evals when2call mcq \
            --data "${W2C_DIR:?}/when2call_test_mcq.jsonl" --adapter "$arm" --output "outputs/s11-ext-w2c-mcq-$arm" & pids+=($!)
        run_logged "$LOGS/ext-gen-$arm.log" python -m agent_from_scratch.evals when2call generate \
            --data "$W2C_DIR/when2call_test_llm_judge.jsonl" --adapter "$arm" --output "outputs/s11-ext-w2c-gen-$arm" & pids+=($!)
        run_logged "$LOGS/ext-bfcl-$arm.log" python -m agent_from_scratch.evals bfcl --bfcl-root "${BFCL_ROOT:?}" \
            --adapter "$arm" --output "outputs/s11-ext-bfcl-$arm" & pids+=($!)
        wait_all "${pids[@]}"
    done
    finish external outputs/s11-ext-*
fi
stop_server
stamp "I: session done"
