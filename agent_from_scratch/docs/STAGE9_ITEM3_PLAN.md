# Stage 9 Item 3 — adapter SFT on the verified examples

Written October 12, 2026 on branch `feat/stage9-adapt-on-sft-examples`, and rewritten October 14
after the work was done, so that it describes Item 3 as decided and as run. Spec and results:
[STAGE.md](STAGE.md), Stage 9 ("Item 3 complete"). Item 2's guide:
[STAGE9_ITEM2_PLAN.md](STAGE9_ITEM2_PLAN.md).

> **Who wrote what.** The coding rules in STAGE.md make training-data logic the user's to write.
> The user wrote `fit`. At the user's explicit request, AI wrote `encode_example` and
> `select_examples`, revised `fit`, and wrote the plumbing (`train`, `replay`, their tests).
> The code is committed; section 6 lists where.

**Goal.** Train one LoRA adapter on the teacher examples and one on the RFT examples, load each
into the ordinary harness through the Stage 8 server, and compare the base, teacher-SFT and
RFT-SFT on val (the `val` split).

**Status: done.** The recipe for later candidates is confirmed in Stage 9 Item 4.

**Not in Item 3.**

- The test set and the external test: Stage 9 Items 5 and 6.
- Sampling the train pool with the chosen adapter (M1's yield): Stage 10 Item 1.
- Expert iteration and GRPO: Stage 10.
- Public data and new train shapes: Stage 9 Item 4.
- Merging an adapter into a GGUF.

## 1. What Item 3 delivers

| Requirement (STAGE.md) | What satisfies it | Result |
|---|---|---|
| Loss on assistant tokens only | `encode_example`: labels come from the character offsets of one joint tokenization; prompt tokens are ignored | 13,393 of 451,019 tokens are trained; they decode back to every completion |
| Overfit → save → reload smoke | `train` on 8 examples the base model gets wrong, served by vLLM, checked with `replay` | adapter 8 of 8, base 0 of 8 |
| A LoRA pilot (D1: bf16 LoRA instead of 4-bit QLoRA) | `train` on the teacher set and on the RFT set | two adapters, one per epoch |
| Adapters reloaded into the ordinary loop | `bench --backend vllm --adapter NAME` | used for every val run |
| Dataset, config and cost records | `metadata.json` and `train_log.jsonl` per run; cost in STAGE.md | \$1.69 |
| Val comparison at the same checkpoint, backend and precision | base, teacher-SFT and RFT-SFT, three seeds each, one server process | 0.573, 0.953, 0.375 |

```
sft-teacher-v3/ ──► train ──► teacher adapters (one per epoch) ─┐
sft-rft/ ─────────► train ──► RFT adapters ─────────────────────┤
                                                                ├─► vLLM (Stage 8 command + --lora-modules)
                                                                │      ├─► replay  (does the server apply the adapter?)
                                                                │      └─► bench --split val --adapter NAME ──► aggregate, compare
base checkpoint, adapter off ───────────────────────────────────┘
```

## 2. Facts the design rests on

Each was checked, not assumed.

1. **The datasets are small.** Counted with the Hugging Face tokenizer in Step 5 (the same
   counts as the GGUF vocabulary):

   | Dataset | Examples | Tokens | Trained tokens | Longest row |
   |---|---|---|---|---|
   | `sft-teacher-v3` | 255 | 451,019 | 13,393 (3.0%) | 3,599 |
   | `sft-rft` | 63 | 81,175 | 1,956 (2.4%) | 2,857 |

2. **The prompt/completion boundary is clean.** Every prompt ends with `<|im_start|>assistant\n`
   and every completion with one `<|im_end|>`, so no token crosses the boundary on this data.
3. **vLLM serves adapters at start-up**, with `--enable-lora --lora-modules NAME=PATH ...`.
   Several pairs after one flag work, and each adapter is listed in `/v1/models` with the base
   as `parent`. `VLLMClient(adapter=NAME)` sends the adapter's name as the request's model, and
   `protocol_mismatch` does not look at the adapter.
4. **The base model** is `Qwen3ForCausalLM`: 36 layers, hidden size 2,560, bf16, at the frozen
   revision `cdbee75f`.
5. **Library versions.** With `vllm==0.30.0` the pod resolves torch 2.13.0+cu130, transformers
   5.19.0 and peft 0.21.2. `requirements-train.txt` pins the same versions for the Mac.
6. **Memory.** One training step on the longest row needs more than the A40's 44 GiB without
   gradient checkpointing (41.5 GiB were allocated when it failed), and 14.4 GiB with it.
   Checkpointing raises the time per row from 0.96 to 1.33 seconds.
7. **The control.** The Stage 8 base scored 0.542 on val (September 27). The base is run again
   in the same session (D7).

## 3. Decisions

All ten were approved on October 12, 2026. D9 and D10 were revised on October 13.

| # | Decision | How it ran |
|---|---|---|
| D1 | LoRA on the bf16 base, no 4-bit quantization: the 8 GB of weights fit the A40, and training then uses the same base as serving | as decided |
| D2 | One row per forward pass, no padding; an optimizer step averages 8 rows | as decided |
| D3 | A row's loss is the mean over its own completion tokens, and rows are averaged, so a 3-token final answer weighs as much as a 200-token tool call | as decided |
| D4 | Rank 16, alpha 32, dropout 0.05 on all seven projections; AdamW at 1e-4 without weight decay; 10% linear warmup, then cosine decay; clipping at 1.0; 3 epochs; seed 0 | as decided, with gradient checkpointing added (fact 6) |
| D5 | Save an adapter per epoch and choose the epoch on val seed 0: most tasks passed, the earlier on a tie | epochs 1–3 passed 52, 59 and 59 of 64: epoch 2 |
| D6 | The RFT arm uses the teacher's recipe and epoch, capped at 10 examples per skeleton | 33 of the 63 examples (10, 10, 6, 7) |
| D7 | Run the base again on val in the same session | 0.573, against Stage 8's 0.542 |
| D8 | Serve every arm with the Stage 8 command plus `--lora-modules` | as decided |
| D9 | One rental for training and val | Session B (M1's yield) moved to Stage 10 Item 1 |
| D10 | Debug `fit` on the Mac in a fresh environment: `requirements.txt` plus `requirements-train.txt`, pinned to the pod's versions | as decided |

The learning rate of optimizer step $s$, counted from 0, with $total$ steps and
$warm = \max(1, \operatorname{round}(0.1 \cdot total))$:

$$\text{rate}(s) = \begin{cases} lr \cdot \dfrac{s + 1}{warm} & s < warm \\[2ex] \dfrac{lr}{2}\left(1 + \cos\dfrac{\pi\,(s - warm)}{\max(1,\ total - warm)}\right) & s \ge warm \end{cases}$$

This is linear warmup followed by cosine decay, the schedule of Hugging Face's
`get_cosine_schedule_with_warmup`. The only difference is $s + 1$, so that the first step already
has a rate.

## 4. Constraints

- Only train examples train. `train` refuses a pilot dataset without `--pilot`, and an adapter
  trained with `--pilot` or `--limit` is a pipeline check, never a candidate.
- Item 3 does not run the test split.
- No change to the verifier, the skeletons, `prompts/`, the `AgentLimits` defaults or `BENCH_*`.
- One renderer and one template: `train` refuses a dataset exported under a different
  `prompts/qwen_chat.jinja`.
- A row is never truncated; one longer than the window is an error.
- Evidence is never overwritten, and every command writes outside the package.
- torch, transformers and peft are imported only when training starts. The tests that need them
  skip without them.
- Adapter weights are not committed; `metadata.json` records their SHA-256.
- Every rental has a cost guard and ends with the pod deleted and `runpodctl pod list --all`
  empty.

## 5. Review focus

Each of these inputs would spoil the result without an error, and each has a test:

| Risk | Expected | Test |
|---|---|---|
| A token spans the prompt/completion boundary | an error, not a guess | `EncodeExampleTests.test_a_token_across_the_boundary_is_an_error` |
| A row is longer than the window | an error, never truncation | `EncodeExampleTests.test_a_row_longer_than_the_limit_is_refused_not_truncated` |
| The base weights move, or the saved adapter is not what was trained | only `lora_` parameters change; a reload gives the same outputs | `FitTests.test_only_the_adapter_moves`, `FitTests.test_a_saved_adapter_reloads_onto_a_fresh_base_with_the_same_outputs` |
| The recorded loss, rate or row order is not what the contract says | the loss is the mean of the group's rows; the rate follows the formula from step 0; the seed alone decides the order | `FitTests.test_the_recorded_loss_is_the_mean_of_the_rows_own_losses`, `test_the_rate_warms_up_from_the_first_step_and_decays_toward_zero`, `test_the_seed_alone_decides_the_row_order` |
| The server answers with the base while the run is labelled with an adapter | `replay` and `bench` refuse an unserved name; the smoke adapter and the base disagree | `ReplayTests.test_an_adapter_the_server_does_not_serve_is_an_error`; Step 8 |
| A pilot dataset, or one exported under another template, is trained on | refused before any model loads | `LoadDatasetTests` |
| The training libraries become a hard dependency | the module imports without them | `LoadDatasetTests.test_the_module_loads_without_the_training_libraries` |

## 6. Files

| File | Responsibility | Commit |
|---|---|---|
| `evals/sft/tensorize.py` | `encode_example` (one example → token row with labels), `select_examples` (per-skeleton cap, limit) | `d811dff` |
| `evals/sft/train.py` | dataset guards, `fit`, LoRA set-up, the `train` command | `b1213c3`, `a82558b` |
| `evals/sft/replay.py` | greedy continuations against targets; the `replay` command | `5dda2bb` |
| `evals/__main__.py` | registers `train` and `replay` | `b1213c3`, `5dda2bb` |
| `tests/test_sft_tensorize.py`, `test_sft_train.py`, `test_sft_replay.py` | 9, 13 and 4 tests | `d811dff`, `b1213c3`, `5dda2bb` |
| `requirements.txt`, `requirements-train.txt` | runtime packages in one file; training packages pinned to the pod's | `3d7c135`, `b1213c3` |
| `evals/README.md`, `docs/benchmark.md` | the `train` and `replay` commands; the GPU runbook's adapter section | `7341fa4` and the Item 3 docs commit |

## 7. Steps

Every step below is done. The commands as run in Session A are in the appendix.

**Step 0 — baseline.** Branched from `main` at `e45d39f`: 388 tests passed.

**Step 1 — `encode_example`.** Tokenize `prompt + completion` as one string. A token whose span
starts before `len(prompt)` gets `IGNORE` (-100); every other token is a target. The function
raises on a token across the boundary, a row longer than `max_tokens`, a tokenizer whose ids and
offsets disagree, and a completion that does not end with one `<|im_end|>`, since the stop token
must be learned. On the average teacher row (1,769 tokens, 52 targets) 2.9% of the positions
carry a loss term. The rest is context: it must be computed, but unlike padding it cannot be
skipped. That is why D2 does not optimise padding.

**Step 2 — `select_examples`.** Whole trajectories (skeleton, run, task) are kept in file order
while their skeleton stays within `max_per_skeleton`; then the first `limit` examples are kept.
On the RFT set, a cap of 10 keeps 33 examples.

**Step 3 — `train`, everything but the loop.** Dataset guards (pilot flag, template hash),
`hf_tokenize` (offsets, no added special tokens), `add_lora` and the command. The training
libraries are imported inside `main`, after every argument and dataset check.

**Step 4 — `fit`.** AdamW over the trainable parameters, without weight decay. One generator,
`random.Random(seed)`, orders the rows of every epoch. Each row is one forward pass, with its
loss divided by the size of its group. After each group: clip at 1.0, set the rate above, step,
and record the group's loss and rate. `on_epoch` saves the adapter. 410 tests passed, in a fresh
pinned environment and in the old conda environment once its transformers was upgraded.

**Step 5 — the real tokenizer.** Both datasets encoded with the Hugging Face tokenizer gave
exactly the counts in fact 1, and the trained tokens decoded to the completion on 255 of 255 and
63 of 63 rows.

**Step 6 — `replay`.** It sends each prompt to `/v1/completions` at temperature 0 and compares
`text + "<|im_end|>"` with the target. 414 tests pass; without torch, 8 of them skip.

**Step 7 — commit before renting.** A shallow clone of the branch passes all 414 tests.

**Step 8 — Session A.** One A40 in EU-SE-1 (CA-MTL-1 had none free), pod `4lwdv5hhkaq147`,
05:13 to 08:07 UTC on October 14, \$1.69.

1. **Install and gate.** `uv pip install vllm==0.30.0 peft -r requirements.txt`. The versions
   matched `requirements-train.txt`, and the 26 new tests passed on the pod.
2. **Smoke.** The base model's loss on the first eight teacher examples, all
   `address_conflict_stop`, was 0.0002, so an overfit on them tests nothing. The base loss was
   measured on all 255 teacher examples. Mean per completion token by skeleton:
   `address_conflict_stop` 0.25, `points_save_recovery` 0.56, `eligibility_record_update` 1.09,
   `inventory_reconciliation` 1.94. The eight worst are bare final answers, four `NONE` and four
   `BLOCKED`, with losses of 6 to 14; they became the smoke dataset `sft-smoke-hard8`. The
   overfit took its loss from 9.38 to 0.0.
3. **Memory.** A one-step probe on the longest row failed for lack of memory (fact 6), so
   `train` enables gradient checkpointing (`a82558b`). The pod's clone was moved to that commit
   with a git bundle, so its hash matches the Mac's, and the smoke was repeated on it
   (`sft-smoke3`).
4. **Training.** Teacher arm: 96 steps in 997 seconds, mean loss by epoch 0.273, 0.047, 0.030.
   RFT arm: 15 steps in 111 seconds, loss 0.018 to 0.0001. Its targets are the base model's own
   samples, so they start almost fitted.
5. **Serving and reload.** The Stage 8 command plus the smoke adapter and six arm adapters. The
   smoke adapter reproduced 8 of 8 targets and the base 0 of 8. The teacher adapter at epoch 3
   reproduces 181 of its 255 training targets.
6. **Epoch, and the comparison.** Epoch 2 was chosen (D5). Then the base, teacher-SFT and RFT-SFT
   ran on val with three seeds each, `aggregate` combined each arm, and `compare` paired the
   base with each adapter per seed. Measured and served prompt tokens differed on 0 of 2,619
   requests. The results and their reading are in STAGE.md.
7. **Copy back and delete.** Runs, reports, logs and the two epoch-2 adapters (132 MB each) are
   in `outputs/stage9-item3-20261014-session-a/`. The archive checksums matched the pod, and
   the adapter hashes matched their `metadata.json`. Copying 244 MB of adapters took about half
   an hour.

**Step 9 — record.** STAGE.md ("Item 3 complete"), the GPU runbook's adapter section in
`docs/benchmark.md`, and a short "Training on a rented GPU" section in `evals/README.md`.

## 8. Exit gate

- [x] 414 tests pass on the Mac, and the 26 new ones pass on the pod with none skipped.
- [x] The smoke adapter reproduced 8 of 8 targets through the server, and the base fewer (0).
- [x] Two adapters exist with `metadata.json` and `train_log.jsonl`, trained from datasets with
  `pilot: false`. Their epoch-2 SHA-256 values: teacher `7a7357fdc0b97e57…`, RFT
  `47354fe8dbc677a2…`.
- [x] The base, teacher-SFT and RFT-SFT ran on val with three seeds each, in one session, on one
  server process.
- [x] No test task was run.
- [x] Results, costs and limits are written in STAGE.md, including the arm that shows no gain.

## 9. Handoff

- **Stage 9 Item 4** trains its candidates (teacher-wide, mix) with this recipe, after
  confirming it, and compares them on val with teacher-v3, this item's teacher adapter.
- **Stage 10 Item 1** samples the train pool with the adapter Item 4 chooses (M1). The commands
  are there. The RFT result is the evidence for its round design: a round cannot train on M_k's
  own verified samples alone.
- **Every later session** reuses the server command, the adapter naming, the recipe in
  `metadata.json` and the smoke method: overfit on examples the base model gets wrong.

## Appendix — Session A as run

Files written on the Mac and copied to `/root` on the pod. `train.sh`, `select.sh` and
`compare.sh` are also kept in the evidence directory under `session-a-logs/`.

**Pack and ship (Mac).**

```sh
git clone --depth 1 --branch feat/stage9-adapt-on-sft-examples "file://$PWD" SCRATCH/stage9-src
COPYFILE_DISABLE=1 tar czf SCRATCH/stage9-src.tgz -C SCRATCH stage9-src
COPYFILE_DISABLE=1 tar czf SCRATCH/stage9-data.tgz \
  outputs/stage9-item2-20261012-sft-teacher-v3 outputs/stage9-item2-20261012-sft-rft
runpodctl pod create --name stage9-sft --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --gpu-id "NVIDIA A40" --cloud-type SECURE --data-center-ids EU-SE-1 \
  --min-cuda-version 13.0 --container-disk-in-gb 60 --ports "22/tcp" --wait --wait-timeout 15m
sleep 7200 && runpodctl pod stop POD_ID     # cost guard, in a second terminal
scp -P PORT -i KEY SCRATCH/stage9-src.tgz SCRATCH/stage9-data.tgz SCRATCH/setup.sh root@HOST:/root/
ssh -o ServerAliveInterval=15 -p PORT -i KEY root@HOST "bash /root/setup.sh"
```

macOS `tar` adds `com.apple.provenance` headers, which GNU `tar` on the pod reports and ignores.

**`setup.sh`.**

```sh
#!/bin/bash
set -euo pipefail
cd /root
tar xzf stage9-src.tgz --no-same-owner
# The archive carries the Mac's user ID; git refuses a repository owned by someone else.
chown -R root:root stage9-src
git -C stage9-src rev-parse HEAD
echo "untracked or modified files on the pod: $(git -C stage9-src status --short | wc -l)"
tar xzf stage9-data.tgz -C stage9-src --no-same-owner
python3 -m venv /root/venv
/root/venv/bin/pip install -q uv
. /root/venv/bin/activate
uv pip install vllm==0.30.0 peft -r stage9-src/agent_from_scratch/requirements.txt 2>&1 | tail -3
python -c "import vllm, torch, transformers, peft; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
cd stage9-src
python -m unittest agent_from_scratch.tests.test_sft_tensorize agent_from_scratch.tests.test_sft_train agent_from_scratch.tests.test_sft_replay 2>&1 | tail -4
echo SETUP_DONE
```

**The smoke dataset: the eight examples the base model predicts worst** (`base_loss.py`, run
from the clone's root with `PYTHONPATH=.`).

```python
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import torch
import transformers

from agent_from_scratch.config import BENCH_MODEL, BENCH_N_CTX
from agent_from_scratch.evals.sft.tensorize import encode_example
from agent_from_scratch.evals.sft.train import hf_tokenize, load_dataset

source = Path("outputs/stage9-item2-20261012-sft-teacher-v3")
examples, metadata = load_dataset(source)
tokenizer = transformers.AutoTokenizer.from_pretrained(BENCH_MODEL["id"], revision=BENCH_MODEL["revision"])
model = transformers.AutoModelForCausalLM.from_pretrained(
    BENCH_MODEL["id"], revision=BENCH_MODEL["revision"], dtype=torch.bfloat16).to("cuda").eval()
losses = []
with torch.no_grad():
    for example in examples:
        row = encode_example(example, hf_tokenize(tokenizer), max_tokens=BENCH_N_CTX)
        loss = model(input_ids=torch.tensor([row["input_ids"]], device="cuda"),
                     labels=torch.tensor([row["labels"]], device="cuda")).loss.item()
        losses.append({"id": example["id"], "skeleton": example["provenance"]["skeleton"], "loss": loss})
Path("outputs/base-loss-teacher-v3.json").write_text(json.dumps(losses, indent=1))
worst = {entry["id"] for entry in sorted(losses, key=lambda entry: -entry["loss"])[:8]}
subset = Path("outputs/sft-smoke-hard8")
subset.mkdir()
lines = source.joinpath("examples.jsonl").read_text(encoding="utf-8").splitlines()
subset.joinpath("examples.jsonl").write_text("".join(line + "\n" for line in lines if json.loads(line)["id"] in worst))
subset.joinpath("metadata.json").write_text(json.dumps(
    {**metadata, "subset_of": source.name,
     "subset_rule": "the 8 examples with the highest base-model loss"}, indent=2))
```

**Train (pod, from `/root/stage9-src` with the venv active).**

```sh
python -m agent_from_scratch.evals train --examples outputs/sft-smoke-hard8 \
  --epochs 40 --accumulate 4 --lr 3e-4 --save-last --output outputs/sft-smoke3
# train.sh, started with: setsid nohup bash /root/train.sh > /root/train.log 2>&1 < /dev/null &
python -m agent_from_scratch.evals train --examples outputs/stage9-item2-20261012-sft-teacher-v3 --output outputs/sft-teacher-r1
python -m agent_from_scratch.evals train --examples outputs/stage9-item2-20261012-sft-rft --max-per-skeleton 10 --output outputs/sft-rft-r1
```

`setsid nohup … &` detaches a job from the SSH session: Ctrl-C on `tail -f` stops only the
watching.

**Serve, reload check, val (pod).**

```sh
PATH=/root/venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup \
  vllm serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --dtype bfloat16 --max-model-len 32768 --enable-lora --max-lora-rank 64 --port 8000 \
  --lora-modules smoke=outputs/sft-smoke3/epoch-40 \
    teacher-e1=outputs/sft-teacher-r1/epoch-1 teacher-e2=outputs/sft-teacher-r1/epoch-2 \
    teacher-e3=outputs/sft-teacher-r1/epoch-3 rft-e1=outputs/sft-rft-r1/epoch-1 \
    rft-e2=outputs/sft-rft-r1/epoch-2 rft-e3=outputs/sft-rft-r1/epoch-3 \
  > /root/vllm.log 2>&1 < /dev/null &
python -m agent_from_scratch.evals replay --examples outputs/sft-smoke-hard8 \
  --ids outputs/sft-smoke3/metadata.json --adapter smoke --output outputs/replay-smoke-adapter.json
python -m agent_from_scratch.evals replay --examples outputs/sft-smoke-hard8 \
  --ids outputs/sft-smoke3/metadata.json --output outputs/replay-smoke-base.json
# select.sh: val seed 0 for each teacher epoch
for e in 1 2 3; do
  python -m agent_from_scratch.evals bench --backend vllm --split val --seed 0 --adapter teacher-e$e --output outputs/val-teacher-e$e-s0
done
# compare.sh 2: three seeds per arm, then aggregate and pair
K=2
for s in 0 1 2; do
  python -m agent_from_scratch.evals bench --backend vllm --split val --seed $s --output outputs/val-base-s$s
  python -m agent_from_scratch.evals bench --backend vllm --split val --seed $s --adapter rft-e$K --output outputs/val-rft-s$s
done
for s in 1 2; do
  python -m agent_from_scratch.evals bench --backend vllm --split val --seed $s --adapter teacher-e$K --output outputs/val-teacher-e$K-s$s
done
python -m agent_from_scratch.evals aggregate outputs/val-base-s? --output outputs/val-base.json
python -m agent_from_scratch.evals aggregate outputs/val-teacher-e$K-s? --output outputs/val-teacher.json
python -m agent_from_scratch.evals aggregate outputs/val-rft-s? --output outputs/val-rft.json
for s in 0 1 2; do
  python -m agent_from_scratch.evals compare outputs/val-base-s$s outputs/val-teacher-e$K-s$s > outputs/compare-base-teacher-s$s.json
  python -m agent_from_scratch.evals compare outputs/val-base-s$s outputs/val-rft-s$s > outputs/compare-base-rft-s$s.json
done
```

**Copy back and delete.**

```sh
# pod
tar czf /root/session-a.tgz --exclude='outputs/stage9-item2-*' --exclude='outputs/sft-smoke*' \
  --exclude='outputs/base-loss-*' --exclude='outputs/smoke-logs' --exclude='outputs/sft-*-r1/epoch-*' outputs
tar czf /root/adapters.tgz outputs/sft-teacher-r1/epoch-2 outputs/sft-rft-r1/epoch-2
# Mac
scp -P PORT -i KEY root@HOST:/root/session-a.tgz root@HOST:/root/adapters.tgz outputs/stage9-item3-20261014-session-a/
runpodctl pod delete POD_ID && runpodctl pod list --all && runpodctl user
```
