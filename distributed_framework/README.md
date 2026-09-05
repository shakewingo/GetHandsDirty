# torchfeather

A from-scratch distributed training framework for a DeepSeek-V3-style Mixture-of-Experts
model, and the 1.3B model trained with it.

Everything here is built rather than imported: MLA attention with weight absorption,
auxiliary-loss-free MoE routing, and the full parallelism stack — FSDP, tensor, pipeline,
expert and expert-tensor parallelism — over PyTorch's `DTensor` / `DeviceMesh` primitives.

---

## The trained model

| | |
|---|---|
| parameters | 1,319,310,336 total · 410,981,376 active per token |
| architecture | 16 layers (1 dense + 15 MoE), dim 1024, 8 heads, MLA with `kv_lora_rank=256` |
| MoE | 32 experts, top-4, 1 shared expert, auxiliary-loss-free balancing |
| tokenizer | DeepSeek-MoE-16B, 102,400 vocab |
| training data | FineWeb-Edu `sample-10BT`, streamed |
| tokens seen | **2.10B** (16,000 steps × 131,072) |
| final loss | **2.920** (200-step average); best single step 2.7661 |
| perplexity | ≈ 18.5 |

At ~5 tokens per active parameter this is deliberately undertrained against the
Chinchilla-optimal ~20. It produces fluent, grammatical English with real but unreliable
factual recall — `The capital of France is Paris.` is correct; `Water boils at 212°C`
has the right number and the wrong unit. The loss was still falling when the budget ran out.

---

## Running inference

```bash
cd distributed_framework
python -m torchfeather.generate --prompt "The capital of France is"
python -m torchfeather.generate --interactive --device cuda
```

Weights default to `./artifacts/torchfeather-1b-16000.pt` (2.64 GB, bf16).

**On a low-memory machine.** An MoE saves compute, not weights: all 1.3B parameters must be
resident even though only 0.41B are active per token. `generate.py` builds the model on the
meta device and loads with `assign=True` + `mmap=True`, so parameters *become* the
checkpoint's file-backed tensors instead of copies of them. Measured on a 3.9 GB box with
2 CPUs: loads in ~1.7 s, 1.2 GB resident, weights in evictable page cache, **1.3 tok/s**.

Inference uses `attn_impl="absorb"`, which folds `W^UK`/`W^UV` into the q and o projections
so the KV cache stores the latent rather than per-head keys and values. Training uses
`attn_impl="naive"` instead — absorb contracts scores over 288 dims against naive's 96, which
is 3–4× the attention FLOPs for a saving that only exists when there is a cache to reuse.

---

## Deployment: how the model was trained

### Hardware

| stage | hardware | rate | duration | cost |
|---|---|---|---|---|
| 0 — local fixes + CPU gate | this box | — | ~1 day | $0 |
| 1 — single-GPU smoke | 1× RTX 4090, secure | $0.74/h | 16 min | $0.27 |
| 2 — parallelism matrix | 8× A40 48 GB, secure | $3.92/h | ~1.5 h | $6 |
| 3 — the training run | 8× A40 48 GB, secure | $3.92/h | 17.9 h | $70 |
| 4 — export + sampling | same pod | $3.92/h | 20 min | $1.3 |
| | | | **total** | **≈ $78** |

RunPod, EU-SE-1. **No 8× RTX 4090 was available at any tier**, and it would not have fit
regardless: peak memory reached 37 GiB/GPU against a 4090's 23.5 GiB usable. The A40 at
$3.92/h for 8 cards was both cheaper than 8× 4090 secure and the only 48 GB option under $8/h.

### Machine-specific settings that are not optional

```bash
export NCCL_P2P_DISABLE=1     # see F9 below
export WANDB_MODE=offline     # or set WANDB_API_KEY
```

Without `NCCL_P2P_DISABLE=1` every multi-GPU run hangs on its *first* collective and dies to
the 300 s watchdog with zero training steps. This machine has no NVLink and splits GPUs 0–3 /
4–7 across NUMA nodes over `SYS`. **Re-check this on any new pod** — it is a property of the
topology, not of this code. A 20-line pure-NCCL script is the fastest way to bisect it.

### Pod setup

```bash
git clone -b <branch> <repo> && cd GetHandsDirty/distributed_framework
bash setup_pod.sh
```

`setup_pod.sh` installs torch from the CUDA-matched index, installs the rest, asserts
`torch.cuda.is_available()`, downloads the tokenizer, and runs both hardware gates
(`torch._grouped_mm` on this SM, and that FlashAttention is genuinely selected rather than
silently degraded to the math backend).

Two environment traps it encodes:

- The image's Python is PEP 668 externally-managed; pip needs `--break-system-packages`.
- **`torch==2.13` does not exist for CUDA 12.8** — that index stops at 2.11.0, and 2.13 ships
  only as `cu130`, which needs driver ≥ 580. Listing torch in `requirements.txt` made pip
  re-resolve it against default PyPI and silently install a `cu130` build that reported
  `cuda False`, so every GPU gate ran on CPU without failing. `requirements.txt` therefore
  does not name torch. The real floor is **2.9** (2.8 lacks `DeviceMesh._unflatten`).

### Launch

```bash
export TORCHFEATHER_CONFIG=tf1b_run
export TF_RUN_STEPS=16000            # sized from measured throughput, not guessed
until torchrun --standalone --nproc_per_node=8 --max-restarts=3 -m torchfeather.train; do
  echo "trainer exited $?; resuming in 30s"; sleep 30
done
```

`--max-restarts` covers one rank dying; the outer loop covers the whole job dying. Resume is
automatic — `CheckpointManager.load(step=-1)` takes the highest `step-N` carrying a
`.metadata` file. Over 17.9 hours this fired exactly once, on the final teardown.

---

## Performance

Measured on 8× A40, `tf1b_run` (dp_shard=8, ep=8):

| | |
|---|---|
| throughput | ~36,000 tok/s aggregate (~4,500/GPU) |
| step time | ~3.6 s for 131,072 tokens |
| MFU | ~12.6% against 74.8 TFLOPS bf16 |
| peak memory | 31–37 GiB/GPU of 44.4 available |
| checkpoint | 15 GB, 1.5 s blocking (async), ~107 saves ≈ 2.7 min total |

### The cost of each parallelism dimension

From the Stage 2 matrix, 50 steps each, same seed and global batch:

| config | dp | tp | ep | etp | pp | tps/GPU | MFU |
|---|---|---|---|---|---|---|---|
| FSDP baseline | 8 | 1 | 1 | 1 | 1 | 6,079 | 16.97% |
| + EP 2 | 8 | 1 | 2 | 1 | 1 | 5,062 | 14.13% |
| + EP 8 | 8 | 1 | 8 | 1 | 1 | 4,062 | 11.34% |
| TP only | 4 | 2 | 1 | 1 | 1 | 2,685 | 7.50% |
| EP borrows TP | 4 | 2 | 4 | 1 | 1 | 2,580 | 7.20% |
| ETP | 4 | 2 | 2 | 2 | 1 | 2,009 | 5.61% |
| PP + EP | 2 | 2 | 2 | 1 | 2 | 1,169 | 3.26% |

On a box with no NVLink, **each added communication dimension costs roughly a third of
throughput**. Cheap $/hour is not cheap $/token. Single-GPU MFU was 30–34%; the drop to
11–17% at 8 GPUs is entirely collectives.

Note the loss column is only comparable *within* a dp degree — `split_dataset_by_node`
shards by dp world size, so dp=8 and dp=4 runs stream different documents.

---

## Continuing from a checkpoint

The full DCP checkpoint (`artifacts/step-16000/`, 15 GB) carries model, optimizer, LR
schedule, dataloader position and step count.

```bash
# Put it back where the trainer looks, then raise the step budget and relaunch.
mkdir -p outputs/tf1b_run/checkpoint
cp -r artifacts/step-16000 outputs/tf1b_run/checkpoint/
TF_RUN_STEPS=32000 TORCHFEATHER_CONFIG=tf1b_run \
  torchrun --standalone --nproc_per_node=8 -m torchfeather.train
```

Three constraints:

- **The GPU count cannot change.** `ParallelAwareDataloader.load_state_dict` asserts
  `dp_world_size == state_dict["world_size"]`. To resume on a different world size, load with
  `checkpoint.load_step=0` (model-only) and accept losing optimizer state and data position.
- **The expert-parallel degree *can* change.** Verified: written at `ep=2`, read back at
  `ep=4`, loss continued 9.6944 → 9.4042 rather than resetting. DCP reshards through DTensor
  metadata.
- **The LR schedule is defined over `training.steps`.** Raising it mid-run restarts the
  cosine from wherever the schedule now says, which is not the same as extending the original
  curve. For a genuine continuation, decide the final step count up front.

### Checkpoint integrity

`_find_load_step` only accepts a `step-N` directory carrying a `.metadata` file, so an async
save killed mid-write is correctly skipped. **`keep_latest_k >= 2` is load-bearing**: it is
what provides a complete older checkpoint to fall back to. With only one checkpoint and that
one torn, training silently restarts from step 1 — `load()` now logs a loud warning naming
the orphaned folders (F7). `enable_first_step_checkpoint=True` closes the same window at the
start of a run.

---

## Configs

`TORCHFEATHER_CONFIG` selects from `config/default_configs.py`:

| name | purpose |
|---|---|
| `tf1b_run` | the real run — 8 GPUs, FSDP + EP 8, 16,000 steps |
| `tf1b_smoke` | single GPU, 4 layers, 100 steps — measure before you spend |
| `tf1b_m1` … `tf1b_m7` | the Stage 2 parallelism matrix |
| `tf1b_reshard` | write a checkpoint at one EP degree, read it at another |

---

## Verification

Run these after any change to the model or the parallelism code:

```bash
python -m torchfeather.minimal_examples.verify_flash_attention   # asserts flash is selected
python -m torchfeather.minimal_examples.verify_ep_gloo           # EP on CPU/gloo, 8 assertions
python -m torchfeather.model.moe.moe                             # MoE self-checks
python -m torchfeather.model.model                               # forward/backward smoke
```

`verify_flash_attention` matters more than it looks: SDPA degrades to the math backend
*silently* when a constraint fails, restoring the `(B,H,S,T)` score tensor the wrapper exists
to avoid. It forces `sdpa_kernel([FLASH_ATTENTION])` so a rejected constraint raises instead.

---

## Notes on what does not work

- **Context parallelism is not usable.** `train.py` passes a `freqs_cis` buffer that does not
  exist in this repo, and `RotaryEmbedding._rotate` indexes by absolute position, which CP's
  load-balanced round-robin sharding breaks. Roughly half a day in `rope.py` to fix.
- **`torch.compile` is off by default.** The MoE has data-dependent shapes and the line that
  makes that tolerable (`capture_scalar_outputs`) is commented out. Turn it on as a measured
  experiment.
- **Vocabulary is oversized for the corpus.** 102,400 × 1024 × 2 = 210M parameters, 15% of the
  model, on a Chinese-and-English tokenizer training on English-only text. A 32k–50k tokenizer
  would buy back ~12% throughput and 100M parameters.
