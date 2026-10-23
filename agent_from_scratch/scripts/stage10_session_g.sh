#!/bin/bash
# Stage 10 Session G (docs/STAGE10_HARD_TASKS_PLAN.md, section 5): mix-v3 is M1, the pool is the
# train shapes levels.json names. Stages: Item 1 on the pool and M1's val at three seeds; K8's
# gate, decided here; the GRPO pilot (up to three passes, G6); the self-SFT rounds (up to three,
# E3); and the comparison of every candidate on one server, val at three seeds.
#
# Each stage ends with a marker (outputs/s10g-STAGE.done) and is skipped when it exists, so a
# session on a host that dies can resume (move a half-written stage's run directories aside
# first: bench never overwrites). Each stage is packed into $PACK/s10g-STAGE.tgz and marked
# $PACK/s10g-STAGE.done for the copy-back.
#
# Usage, from the repository root with the venv active and no server running:
#   M1_ADAPTER=outputs/sft-mix-v3-r1/epoch-2 bash agent_from_scratch/scripts/stage10_session_g.sh
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
. "$S/stage10_lib.sh"
PACK=${PACK:-/root}
# M1's data (mix-v3): teacher-wide-v2 x3, When2Call, Toucan; each round adds M_k's verified set.
export TEACHER_WIDE=outputs/stage9-item7-20261019-sft-teacher-wide-v2 OWN_SHARE=0.445
STAGE10_POOL=${STAGE10_POOL:-$(python -c "from agent_from_scratch.evals.bench.skeletons import SHAPES, load_levels
print(','.join(s.name for s in SHAPES if s.split == 'train' and s.name in load_levels()))")}
export STAGE10_POOL
[ -n "$STAGE10_POOL" ] || { echo "levels.json names no train shape" >&2; exit 2; }

stamp() { if dry; then echo "# $*"; else echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/timeline.txt"; fi; }
finished() { [ -f "outputs/s10g-$1.done" ]; }
finish() {  # finish STAGE PATH...: mark the stage done and pack it for the copy-back
    local stage=$1
    shift
    dry && { echo "+ finish $stage"; return; }
    touch "outputs/s10g-$stage.done"
    tar czf "$PACK/s10g-$stage.tgz" "$@" "$LOGS" outputs/s10g-*.done
    touch "$PACK/s10g-$stage.done"
}
rate() {  # rate PREFIX: pass@1 over outputs/PREFIX-s0..2
    python -c "
import json, sys
runs = [json.load(open(f'outputs/{sys.argv[1]}-s{s}/summary.json')) for s in (0, 1, 2)]
print(sum(r['passed'] for r in runs) / sum(r['total'] for r in runs))" "$1"
}
stamp "G: pool $STAGE10_POOL"

if ! finished item1; then
    stamp "G: Item 1 and M1's val"
    serve "M1=$M1_ADAPTER"
    pids=()
    for s in 0 1 2; do bench "s10-val-m1-s$s" val "$s" M1 & pids+=($!); done
    sample_train s10-m1 M1
    wait_all "${pids[@]}"
    stop_server
    finish item1 outputs/s10-m1-train-* outputs/s10-val-m1-*
fi

if ! finished gate; then
    # K8: A1 (the ask side of long_edit_dialogue, when it is in the pool) gates both arms; G1 (3 in 8
    # of the pool mixed, in two shapes) gates GRPO; self-SFT also needs a pool below its ceiling.
    dry || python - <<'EOF'
import glob, json, math
strict = json.load(open("outputs/s10-m1-train-strict.json"))
overall = strict["overall"]
mixed_shapes = [name for name, row in strict["by_skeleton"].items() if row["mixed"] > 0]
g1 = overall["mixed"] >= math.ceil(0.375 * overall["tasks"]) and len(mixed_shapes) >= 2
earned = attempts = 0
for run in glob.glob("outputs/s10-m1-train-s[0-9]"):
    for record in json.load(open(f"{run}/results.json")):
        if record["skeleton"] == "long_edit_dialogue" and int(record["id"].split("-")[-1]) % 2 == 1:
            attempts += 1
            earned += int(record["passed"] and record["checks"].get("format_exact") is not False)
a1 = attempts == 0 or earned / attempts >= 0.05
below_ceiling = overall["mean_pass@1"] < 0.9
gate = {"g1": g1, "mixed": overall["mixed"], "tasks": overall["tasks"], "mixed_shapes": mixed_shapes,
        "a1": a1, "ask_side": [earned, attempts], "mean_pass@1": overall["mean_pass@1"],
        "grpo": g1 and a1, "sft": a1 and below_ceiling}
json.dump(gate, open("outputs/s10g-gate.json", "w"), indent=2)
print(json.dumps(gate))
EOF
    finish gate outputs/s10g-gate.json
fi
GRPO=$(dry && echo True || python -c "import json; print(json.load(open('outputs/s10g-gate.json'))['grpo'])")
SFT=$(dry && echo True || python -c "import json; print(json.load(open('outputs/s10g-gate.json'))['sft'])")
stamp "G: gate grpo=$GRPO sft=$SFT"

if [ "$GRPO" = True ]; then
    for p in 1 2 3; do
        finished "grpo-p$p" && continue
        stamp "G: GRPO pass $p"
        STAGE10_START=$M1_ADAPTER bash "$S/stage10_grpo.sh" "$p" 2>&1 | tee -a "$LOGS/grpo-driver.log"
        finish "grpo-p$p" outputs/s10-grpo-p$p-i*-s* outputs/s10-grpo-p$p-i*-step/step.json \
            "outputs/s10-grpo-pass-$p" "outputs/s10-grpo-val-p$p-s0" outputs/s10-grpo-val-p0-s0
        if grep -q "G6: the pilot stops after pass $p" "$LOGS/grpo-driver.log"; then
            stamp "G: G6 stopped the pilot after pass $p"
            break
        fi
    done
fi

if [ "$SFT" = True ]; then
    for K in 1 2 3; do
        NEXT=$((K + 1))
        if ! finished "round$K"; then
            stamp "G: self-SFT round $K"
            bash "$S/stage10_ei_round.sh" "$K"
            finish "round$K" "outputs/s10-rft-m$K" "outputs/s10-sft-ei$K/metadata.json" \
                "outputs/s10-m$NEXT-r1/metadata.json" "outputs/s10-m$NEXT-r1/train_log.jsonl" "outputs/s10-m$NEXT" \
                "outputs/s10-m$NEXT-epoch.txt" outputs/s10-val-m$NEXT-* outputs/s10-m$NEXT-train-*
        fi
        dry && continue
        EPOCH=$(cat "outputs/s10-m$NEXT-epoch.txt")
        NOW=$(rate "s10-val-m$NEXT-e$EPOCH")
        M1=$(rate s10-val-m1)
        stamp "G: E3 M$NEXT val pass@1 $NOW, M1 $M1"
        if python -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) - 0.05 else 1)" "$NOW" "$M1"; then
            stamp "G: E3 stops the rounds after round $K"
            break
        fi
    done
fi

if ! finished compare; then
    # Every candidate on one server (C1 of STAGE10_PLAN.md): M1, the rounds' models, the GRPO passes.
    modules=("m1=$M1_ADAPTER")
    for k in 2 3 4; do [ -d "outputs/s10-m$k" ] && modules+=("m$k=outputs/s10-m$k"); done
    for p in 1 2 3; do [ -d "outputs/s10-grpo-pass-$p" ] && modules+=("grpo$p=outputs/s10-grpo-pass-$p"); done
    dry && modules+=("m2=outputs/s10-m2" "grpo1=outputs/s10-grpo-pass-1")
    stamp "G: comparison of ${modules[*]}"
    serve "${modules[@]}"
    pids=()
    for module in "${modules[@]}"; do
        name=${module%%=*}
        for s in 0 1 2; do bench "s10-final-$name-s$s" val "$s" "$name" & pids+=($!); done
    done
    wait_all "${pids[@]}"
    stop_server
    finish compare outputs/s10-final-*
fi
stamp "G: session done"
