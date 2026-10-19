#!/bin/bash
# Sample one adapter on the whole train split: 8 seeds in parallel, then `aggregate` and
# `aggregate --strict` (Stage 10 Item 1, and every model's row of the plan's Table 1).
#
# Usage, from the repository root with the venv active and no server running:
#   bash agent_from_scratch/scripts/stage10_sample.sh NAME ADAPTER_DIR
# e.g. NAME=s10-m1 ADAPTER_DIR=outputs/sft-mix-r1/epoch-2 writes outputs/s10-m1-train-s0..7,
# outputs/s10-m1-train-aggregate.json and outputs/s10-m1-train-strict.json.
set -euo pipefail
. "$(dirname "$0")/stage10_lib.sh"
NAME=${1:?NAME, e.g. s10-m1}
ADAPTER_DIR=${2:?ADAPTER_DIR, e.g. outputs/sft-mix-r1/epoch-2}

serve "sampled=$ADAPTER_DIR"
sample_train "$NAME" sampled
stop_server
