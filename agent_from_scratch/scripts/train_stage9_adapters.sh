 #!/bin/bash
# Train the Stage 9 SFT candidates on a GPU pod, one after the other, with Item 3's recipe
# (the defaults of `train`: LoRA rank 16, rate 1e-4, 3 epochs, one adapter per epoch).
#
# Usage, from the repository root on the pod, with the datasets copied into outputs/:
#   setsid nohup bash agent_from_scratch/scripts/train_stage9_adapters.sh [CANDIDATE ...] \
#     > train.log 2>&1 < /dev/null &
# CANDIDATE is teacher-v3, teacher-wide, mix or mix-v2 (Item 7); with none, the first three are trained.
#
# Runs are sequential on purpose: the server cannot share the GPU, and two runs on one GPU take
# about as long as one after the other. On an A100 80 GB (0.35 ms per token, measured on
# teacher-wide): teacher-wide 28 minutes; teacher-v3 about 8 minutes and mix about 5 hours,
# estimated. Mix's longest rows need about 52-63 GiB, more than an A40 has; teacher-v3 and
# teacher-wide fit an A40. Mix's long and short rows fragment the CUDA allocator (Session A
# logged "allocation failed with OOM" warnings that the allocator retried and survived), so
# expandable segments are turned on.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Design and results: docs/STAGE9_ITEM3_PLAN.md and docs/STAGE9_ITEM4_PLAN.md.
set -euo pipefail

TEACHER_V3=outputs/stage9-item2-20261012-sft-teacher-v3   # 255 examples, 4 train shapes
TEACHER_WIDE=outputs/stage9-item4-20261015-sft-teacher-wide # 956 examples, 16 train shapes
MIX=outputs/stage9-item4-20261015-sft-mix-v1                # teacher-wide x3 + When2Call + Toucan
MIX_V2=outputs/stage9-item7-20261019-sft-mix-v2             # teacher-wide-v2 x2 + capped When2Call + Toucan

train() {  # train DATASET OUTPUT
    python -m agent_from_scratch.evals train --examples "$1" --output "$2"
    echo "DONE $2"
}

candidates=("$@")
[ ${#candidates[@]} -eq 0 ] && candidates=(teacher-v3 teacher-wide mix)
for name in "${candidates[@]}"; do  # check every name before any training starts
    case "$name" in teacher-v3|teacher-wide|mix|mix-v2) ;; *)
        echo "Unknown candidate: $name (teacher-v3, teacher-wide, mix or mix-v2)" >&2; exit 2 ;; esac
done
for name in "${candidates[@]}"; do
    case "$name" in
        teacher-v3) train "$TEACHER_V3" outputs/sft-teacher-r1 ;;
        teacher-wide) train "$TEACHER_WIDE" outputs/sft-wide-r1 ;;
        mix) train "$MIX" outputs/sft-mix-r1 ;;
        mix-v2) train "$MIX_V2" outputs/sft-mix-v2-r1 ;;
    esac
done
