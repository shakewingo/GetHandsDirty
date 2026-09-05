# torchfeather

A from-scratch distributed training framework for a DeepSeek-V3-style Mixture-of-Experts
model, and the 1.3B model trained with it. Original repo forked from [here](https://github.com/hkproj/torchfeather/tree/main) with some model architecture customized.

Everything here is built rather than imported: MLA attention with weight absorption,
auxiliary-loss-free MoE routing, and the full parallelism stack — FSDP, tensor, pipeline,
expert and expert-tensor parallelism — over PyTorch's `DTensor` / `DeviceMesh` primitives.

The full engineering record — the audit that found ten defects, the staged plan, and every
measurement behind the numbers below — is written up as a runbook:
[TorchFeather Phase 3 Runbook](https://claude.ai/code/artifact/391c0bac-d9f2-48e9-a1dc-b644a0d27a28).
*(Private link — visible to the repo owner unless explicitly shared.)*

---

## The trained model

| | |
|---|---|
| parameters | 1.3B total · 411M active per token |
| architecture | 16 layers (1 dense + 15 MoE), dim 1024, 8 heads, MLA with `kv_lora_rank=256` |
| MoE | 32 experts, top-4, 1 shared expert, auxiliary-loss-free balancing |
| tokenizer | DeepSeek-MoE-16B, 102,400 vocab |
| training data | FineWeb-Edu `sample-10BT`, streamed |
| tokens seen | **2.10B** (16,000 steps × 131,072) |
| final loss | **2.920** · perplexity ≈ 18.5 |
| trained on | 8× A40, 17.9 h, **≈ $78** |

At ~5 tokens per active parameter this is deliberately undertrained against the
Chinchilla-optimal ~20. It produces fluent, grammatical English with real but unreliable
factual recall — `The capital of France is Paris.` is correct; `Water boils at 212°C` has
the right number and the wrong unit. The loss was still falling when the budget ran out.

---

## Inference

```bash
cd distributed_framework
python -m torchfeather.generate --prompt "The capital of France is"
python -m torchfeather.generate --interactive --device cuda
```

Weights default to `./artifacts/torchfeather-1b-16000.pt` (2.64 GB, bf16).

**Low-memory machines work.** An MoE saves compute, not weights — all 1.3B parameters must
be resident even though only 0.41B are active per token. `generate.py` loads with
`assign=True` + `mmap=True`, so parameters become the checkpoint's file-backed tensors
rather than copies. Runs on a 3.9 GB box at 1.3 tok/s with 1.2 GB resident.

Inference uses `attn_impl="absorb"`, which folds `W^UK`/`W^UV` into the q and o projections
so the KV cache stores the latent rather than per-head keys and values. Training uses
`attn_impl="naive"`: absorb contracts scores over 288 dims against naive's 96, which only
pays off when there is a cache to reuse.

---

## Training

### 1. Provision

8 GPUs with **≥ 40 GB each** — peak memory is 31–37 GiB per GPU. This model was trained on
8× A40 48 GB on RunPod at $3.92/h.

### 2. Set up

```bash
git clone -b <branch> <repo> && cd GetHandsDirty/distributed_framework
bash setup_pod.sh
```

Installs dependencies against the CUDA version the machine actually has, downloads the
tokenizer, and runs two hardware gates: that `torch._grouped_mm` works on this SM, and that
FlashAttention is genuinely selected rather than silently degraded to the math backend.
**If either gate fails, stop** — training will run, just far slower than it should.

Requires torch ≥ 2.9 (2.8 lacks `DeviceMesh._unflatten`). `setup_pod.sh` picks the right
build; don't install torch by hand.

### 3. Launch

```bash
export TORCHFEATHER_CONFIG=tf1b_run
export TF_RUN_STEPS=16000
export NCCL_P2P_DISABLE=1        # required on machines without NVLink
export WANDB_MODE=offline        # or set WANDB_API_KEY

until torchrun --standalone --nproc_per_node=8 --max-restarts=3 -m torchfeather.train; do
  echo "trainer exited $?; resuming in 30s"; sleep 30
done
```

`NCCL_P2P_DISABLE=1` is not optional on a machine without NVLink: without it every run
hangs on its first collective and dies to the watchdog with zero steps. Check your topology
with `nvidia-smi topo -m` — if you see `SYS` between GPUs, you need it.

The `until` loop plus `--max-restarts` makes the run survive crashes. Resume is automatic:
the trainer loads the newest complete checkpoint and continues.

### Sizing the run

Set `TF_RUN_STEPS` from measured throughput, not from a guess:

```
steps = budget_hours × 3600 × aggregate_tokens_per_sec / 131072
```

Run `TORCHFEATHER_CONFIG=tf1b_smoke` on one GPU first to get that number. The LR schedule
is a cosine defined over `steps`, so decide the final count before starting — a truncated
run ends parked at a high learning rate.

### Measured throughput

| | |
|---|---|
| throughput | ~36,000 tok/s aggregate (~4,500/GPU on A40) |
| step time | ~3.6 s for 131,072 tokens |
| MFU | ~12.6% |
| checkpoint | 15 GB, 1.5 s blocking (async) |

---

## Continuing from a checkpoint

The DCP checkpoint carries model, optimizer, LR schedule, dataloader position and step count.

```bash
mkdir -p outputs/tf1b_run/checkpoint
cp -r artifacts/step-16000 outputs/tf1b_run/checkpoint/
TF_RUN_STEPS=32000 TORCHFEATHER_CONFIG=tf1b_run \
  torchrun --standalone --nproc_per_node=8 -m torchfeather.train
```

- **The GPU count cannot change.** The dataloader asserts it. To resume on a different world
  size, set `checkpoint.load_step=0` for a model-only load and accept losing optimizer state
  and data position.
- **The expert-parallel degree can change.** DCP reshards through DTensor metadata — verified
  by writing at `ep=2` and reading back at `ep=4`.
- **Keep `keep_latest_k >= 2`.** A checkpoint interrupted mid-write is unloadable; the second
  one is what the trainer falls back to.

---

## Configs

Select with `TORCHFEATHER_CONFIG`:

| name | purpose |
|---|---|
| `tf1b_run` | the full run — 8 GPUs, FSDP + EP 8 |
| `tf1b_smoke` | one GPU, 4 layers, 100 steps — measure before you spend |
| `tf1b_m1` … `tf1b_m7` | parallelism matrix: FSDP / EP / TP / ETP / PP combinations |
| `tf1b_reshard` | write a checkpoint at one EP degree, read it at another |

Model and training hyperparameters live in `config/default_configs.py`.

---

## Known limitations

- **Context parallelism does not work.** `train.py` references a `freqs_cis` buffer that does
  not exist here, and RoPE indexes by absolute position, which CP's load-balanced sharding
  breaks. Leave `context_parallel_degree=1`.
- **`torch.compile` is off by default.** The MoE has data-dependent shapes; enable it as a
  measured experiment, not a default.
- **No validation loop.** There is no held-out split or eval during training — every reported
  number is training loss.
- **The vocabulary is oversized for English-only data.** 102,400 tokens costs 210M parameters,
  15% of the model. A 32k–50k tokenizer would buy back throughput and parameters.
