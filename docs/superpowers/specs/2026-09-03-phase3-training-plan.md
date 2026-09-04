# Phase 3: Train a 1.4B MoE on Rented GPUs

Date: 2026-09-03
Scope: `distributed_framework/torchfeather`
Status: **Stage 0 complete and verified on branch `feat/phase3-training-prep`.** Every F/H item
below is fixed; the M/L items are fixed except where noted. The local gate passes: 4 steps of the
real trainer on CPU/gloo with decreasing loss, a checkpoint written, and a clean resume from it.
Stage 1 (first rented GPU) is the next action.

Supersedes the empty `# Phase 3` heading at the end of `2026-09-02-moe-design.md`.
Corrects one stale finding in `2026-09-03-parallelism-structure-analysis.md` (B1 is fixed).

---

## 0. Verdict

The framework is close, but **`train.py` has never executed a single step** and cannot today.
Four independent fatal defects sit between here and step 1, and none of them is in the
parallelism code — they are in the dataloader, two never-written helpers, and a dtype.

Separately, the shipped `deepseek_v3` config builds a **16.2B-parameter** model (2.66B active),
not the 1–3B the goal calls for. Memory is driven by *total* parameters, so that config needs
~260 GB of optimizer state. A new config is required, not a tweak.

Budget verdict: **~$43 of RunPod time** gets you a verified parallelism matrix plus a ~1.3B-token
pretraining run of a 1.4B-total / 0.41B-active MoE. That is undertrained by Chinchilla standards
(~3 tokens per active parameter) but produces a model that generates fluent English and a loss
curve you can defend.

---

## 1. Audit

**Fix status, 2026-09-03.** F2/F3 were written by the repo owner before this pass; everything
else here was fixed and verified during it. Two findings were added *by* the fix work:

| | status | evidence |
|---|---|---|
| F1 dataset methods nested in `__init__` | fixed | `class methods: ['__init__', '_get_data_iter', '__iter__', 'state_dict', 'load_state_dict']` |
| F2 `clip_grad_norm_` | fixed by owner | present, incl. the EP-mesh split and PP all-reduce |
| F3 `set_pg_timeouts` | fixed by owner | present, barriers before changing timeouts |
| F4 float expert counts | fixed | `torch.bincount`; router counts now `torch.int64` |
| F5 environment | fixed | `requirements.txt`, `setup_pod.sh`, wandb falls back to console |
| **F6 MoE autocast dtype** (new) | fixed | found by the Stage 0 dry run; see below |
| H1 16.2B config | fixed | `tf1b_run`: 1.319B total, 2.09 GFLOP/token |
| H2 no SDPA / CP wrong | **partly** | SDPA done and verified; CP still blocked on rope, see below |
| H3 PP renumbers layers | fixed | ModuleDict; stage 0 owns keys 0-3, stage 1 owns 4-7, no overlap |
| H4 `n_dense_layers` unwired | fixed | layer 0 builds `ffn`, layers 1+ build `moe` |
| M1 absorb in training | fixed | `attn_impl="naive"`, TP-adaptive; absorb untouched for inference |
| M2-M6, L1-L3 | fixed | see §1.1 |

### F6 — MoE output dtype assumes FSDP, breaks under autocast (found by the Stage 0 dry run)

`MoE.forward` cast the scaled routed output to `x.dtype` before `index_add`-ing it into the
shared-expert output. Under FSDP mixed precision `x` is already bf16 and the two agree. Under
`torch.autocast` -- which is what `maybe_enable_amp` selects for single-device and DDP runs,
i.e. **exactly the Stage 1 smoke config** -- `x` stays fp32 while the shared-expert `nn.Linear`
returns bf16:

```
RuntimeError: index_add_(): self (BFloat16) and source (Float) must have the same scalar type
  moe.py:203  out = out.index_add(0, token_indices_experts_sorted, routed_output)
```

**Fix:** cast to `out.dtype`, the tensor actually being added into, not to `x.dtype`.

This is the argument for the free local gate in one bug: it is invisible to every multi-GPU
config and fatal to the first single-GPU run.

### H2 — SDPA is in; context parallelism needs a rope refactor on top

`ScaledDotProductAttentionWrapper` now has a real `forward` and the naive path routes through
the module-level `F.scaled_dot_product_attention` (verified: 2 calls per 2-layer forward,
`is_causal=True`, V zero-padded 32 -> 48 so the flash backend's `Dq == Dk == Dv` holds). Output
is unchanged against the absorbed path at `3.35e-08`.

That fixes the memory and speed problem. It does **not** by itself fix CP, which has two further
blockers found while wiring this:

1. `train.py:340` passes `m.freqs_cis` into `cp_buffers`. `RotaryEmbedding` registers
   `cos_cached` and `sin_cached`; **`freqs_cis` does not exist anywhere in the repo.**
   `AttributeError` the moment `cp > 1`.
2. `RotaryEmbedding._rotate` indexes the tables by absolute position
   (`cos_cached[start_pos : start_pos + seqlen]`). Under CP with load balancing each rank holds a
   round-robin-permuted, non-contiguous slice of the sequence, so absolute indexing applies the
   wrong rotation angles. The tables have to become a model-level buffer that CP shards under the
   same permutation -- which is precisely why torchtitan routes `freqs_cis` through `cp_buffers`.

So CP stays out of the Stage 2 matrix. Cost to close: roughly half a day, all in `rope.py`.

### 1.1 What the M/L pass actually changed

- **M1** `attn_impl="naive"` in the new config, and the naive path is now TP-adaptive: `k_pe`
  arrives `Replicate()` while `k_nope` is `Shard(2)`, so it is expanded to the *local* head count
  (`local_head_count`, read off the shard) and re-wrapped, instead of expanding to the global
  count and forcing DTensor to reconcile placements. Naive + KV cache under TP now raises a
  clear `NotImplementedError` instead of silently writing a wrong-shaped cache.
- **M2** `max_batch_size=1` in the new config.
- **M3** peak-FLOPS entries for RTX 4090 / 5090 / 3090, A6000 / A40 / A5000. The 4090 entry is
  82.6 TFLOPS -- bf16 with fp32 accumulate, which is what PyTorch matmul actually reaches; the
  marketed 165 TFLOPS is fp16-accumulate. MFU now reads honestly on a rented 4090.
- **M4** `keep_latest_k = 3`.
- **M5** sigmoid + `route_norm=True`, in the new config *and* in `get_deepseek_v3_model_args`,
  which was contradicting its own dataclass default.
- **M6** resolved by implementing the wrapper rather than deleting it.
- **L1** `.gitignore` rewritten -- `*model*` narrowed to actual weight/asset patterns, so new
  source files under `model/` are no longer silently ignored; 14 tracked `.pyc` files untracked.
- **L2** `compile.enable = False` in the new configs.
- **L3** `training.dataloader_num_workers`, 0 by default, 2 in the run config.
- **Not changed, flagged for a decision:** `Attention.__init__` computes `self.softmax_scale`
  with a YaRN mscale correction and **neither path has ever used it** -- both divide by
  `sqrt(qk_head_dim)`. Switching to it would silently change training math and break
  absorb/naive equivalence, so it stays as-is. Inactive for `tf1b` anyway
  (`max_seq_len=2048 < original_seq_len=4096`); it only matters if you extend context later.
- **Not changed:** B4/B5 from the parallelism analysis (`_apply_full_ac` ignores `ac_config`;
  `model_compile_enabled` accepted and unused). Both harmless.

### 1.2 Supporting changes that made the local gate possible

The Stage 0 gate needs the trainer to run without a GPU. Three small changes:

- `device_utils.get_device_info` returned `"cuda"` on a box with no accelerator. It now returns
  `"cpu"` with a no-op memory-stats shim, so failures are honest instead of obscure.
- `init_process_group` hardcoded `backend="nccl"`; it now follows the device type.
- `DeviceMemoryMonitor._to_pct` guarded against a zero device capacity.

### 1.3 Verification run on this branch

```
EP (gloo, unpatched source)   8/8 assertions at 2 and 4 ranks, EP-vs-baseline maxdiff 0.00000
absorb vs naive               3.35e-08 (identical model, both paths)
SDPA actually called          2 calls / 2-layer fwd, is_causal=True, V padded 32->48
autocast fwd+bwd              loss finite, every grad finite   (the F6 regression test)
PP stage split                stage 0 keys ['0'..'3'], stage 1 keys ['4'..'7'], overlap: none
tf1b_run sizing               1.319B total, 2.09 GFLOP/token
trainer, CPU/gloo, 4 steps    loss 11.5353 -> 11.4837, checkpoint written
trainer, resume               loaded step-4, "Training starts at step 5", ran to 6
```

Severity: **F** = fatal, blocks step 1. **H** = high, wrong results or wrong scale.
**M** = medium, cost/quality. **L** = low, hygiene.

### F1 — `HuggingFaceDataset` has no methods

`datasets/hf_datasets.py:79-145`. `_get_data_iter`, `__iter__`, `state_dict` and
`load_state_dict` are indented **inside `__init__`**. They are local functions that are defined
and discarded on every construction. The class body contains only `__init__`:

```
$ python -c "...ast..."
class methods: ['__init__']
nested in __init__: ['_get_data_iter', '__iter__', 'state_dict', 'load_state_dict']
```

Consequence: iterating the dataloader hits `IterableDataset.__iter__` and raises. Checkpoint
resume of data position is also dead code.

**Fix:** dedent all four by four spaces.

### F2 — `dist_utils.clip_grad_norm_` does not exist

`train.py:456` calls it every step. `distributed/utils.py` defines
`_dist_reduce / dist_max / dist_sum / dist_mean / set_determinism / create_context_parallel_ctx /
get_train_context / maybe_enable_amp` and nothing else.

**Fix:** write it. It must (a) separate DTensor from plain params, (b) compute the total norm
across the PP mesh (all-reduce the per-stage norms with `pow(norm, 2)` sum then sqrt, or `max`
for inf-norm), and (c) when `ep_enabled`, handle expert params living on a *different* mesh from
the rest — compute the two norms separately and combine, rather than letting `foreach` batch
across meshes. Port from torchtitan `distributed/utils.py::clip_grad_norm_`. This is the one
item in the audit with real design content; the prior analysis doc flagged it as such (§B2).

### F3 — `dist_utils.set_pg_timeouts` does not exist

`train.py:546`, called once after step 1. Same file, same absence.

**Fix:** small — iterate `parallel_dims.get_all_one_dimensional_meshes()`, call
`torch.distributed.distributed_c10d._set_pg_timeout(timeout, mesh.get_group())` for each, plus
the default group. Must be preceded by a `barrier()` and a device sync so no rank is mid-collective.

### F4 — expert token counts are `float32`; the EP all-to-all requires ints

Known as §12.8 of the MoE design, still open in the source. `torch.histc`'s CPU kernel is
float-only, so `moe.py:76` and `moe.py:99` both do `.view(-1).float()` and return `float32`
counts. `ExpertParallel._token_dispatch` derives its split sizes from those counts:

```
router counts dtype: torch.float32 -> splits would be: [38.0, 26.0]
```

`all_to_all_single_autograd` then raises `All output_split_sizes must be int or SymInt`.

**Fix:** `torch.bincount(idx.view(-1), minlength=self.num_experts)` at both sites. Integral by
construction, no bin arithmetic, and it raises rather than silently accepting float scores.
Everything downstream (`cumsum(0, dtype=torch.int32)`, the `float32 tokens_per_expert` buffer)
already accepts int64.

### F5 — the environment cannot run the trainer

- No `requirements.txt` / `pyproject.toml` anywhere in the repo.
- `datasets`, `tokenizers`, `torchdata`, `wandb` are all absent from the active env.
- `config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"` — that directory does not
  exist. `DeepSeekV3Tokenizer.__init__` raises `FileNotFoundError` before anything else runs.
- `metrics.py::_build_metric_logger` constructs `WandBLogger` unconditionally on the logging
  rank. There is no console-only path, so a pod without `wandb` credentials dies at
  `Trainer.__init__`.

**Fix:** add `requirements.txt`; download the tokenizer assets; either set
`WANDB_MODE=offline` or add a `BaseLogger` fallback when the `wandb` import fails.

Verified the tokenizer assets *will* satisfy the loader once present — `bos_token`/`eos_token`
in `tokenizer_config.json` are dicts with a `content` key (what `_get_token_from_config`
asserts), and `vocab_size` is 102400, matching the config.

### H1 — the shipped config is 16.2B, not 1–3B

`get_deepseek_v3_model_args()` measured on the meta device:

```
dense 791,145,984 | sparse 15,419,179,008 | active 2,663,247,360 | TOTAL 16.210B
```

FSDP shards optimizer state by *total* params: 16.2B x 16 B/param (fp32 master + fp32 grad +
Adam m,v) = 259 GB, before activations. See §2 for the replacement config.

### H2 — context parallelism is silently wrong

`model/attention.py` defines `ScaledDotProductAttentionWrapper` with **no `forward` method**, and
`Attention.forward` never calls it — the score math is explicit `matmul` + `softmax`. But
`create_context_parallel_ctx` works by monkey-patching `F.scaled_dot_product_attention` to insert
the ring-attention exchange. Since that function is never called, CP does not shard attention: each
rank computes attention over only its local sequence chunk with no cross-rank KV exchange, and the
run trains happily on wrong gradients.

**Fix (or avoid):** either route the score math through `F.scaled_dot_product_attention`, or drop
`cp` from the test matrix and delete `context_parallel_degree` from the configs. **Do not run
`fsdp_cp` and believe the loss curve.**

Secondary consequence at audit time: no FlashAttention, with the `(B, S, H, T)` score tensor
materialized — 537 MB per layer at `S=2048, B=8, H=8`, 2.1 GB at `S=4096`, which is what capped
sequence length rather than the parameters. **Now fixed**; see the FlashAttention subsection of §7
for what does and does not get the flash backend.

### H3 — pipeline parallelism renumbers layers, breaking checkpoints

`pipeline_parallel.py:258-269` rebuilds `model.layers` as
`nn.ModuleList([layer for i, layer in enumerate(module_value) if i in indices_to_keep])`. Stage 1's
`layers.8..15` become `layers.0..7` in its chunk. Every stage therefore writes parameters named
`layers.0.*`, and DCP silently merges them. Upstream uses `nn.ModuleDict` keyed by the original
index precisely to avoid this.

**Avoid:** run PP with `checkpoint.enable = False`, or switch `model.layers` to a `ModuleDict`.
PP is not needed for a 1.4B model; it is a matrix-exercise only.

### H4 — `n_dense_layers` is unwired

`DecoderLayer.__init__(self, args)` takes no `layer_id` and sets
`self.moe_enabled = args.moe_enabled` — a single global flag. The MoE design (§4) specifies
`moe_enabled = layer_id >= args.n_dense_layers`. Consequences today:

- every layer is MoE; the first-dense-layers pattern of DeepSeek-V3 is absent;
- `inter_dim` (10944) is dead — `FeedForward` is never constructed;
- the `not transformer_block.moe_enabled` branch of the TP plan
  (`parallelize.py:268-281`) is never taken, so the dense-FFN TP path is untested.

**Fix:** three lines — `DecoderLayer(layer_id, args)`, the flag, and the loop in
`DeepSeekV3Model.__init__`. Worth doing: it is cheap and it turns on a code path the matrix is
supposed to cover.

### M1 — training with `attn_impl="absorb"` wastes attention FLOPs

Weight absorption is an *inference* optimization: it trades score-space width for not
re-expanding the KV cache. In training there is no cache to save, and it contracts scores over
`kv_lora_rank + qk_rope_head_dim = 288` instead of `qk_nope_head_dim + qk_rope_head_dim = 96`
— 3x on QK and 4x on AV, plus a decompression einsum.

Verified numerically equivalent and measurably cheaper:

```
absorb vs naive maxdiff: 3.07e-06   (output scale 1.5e-02)
absorb   fwd 0.659s
naive    fwd 0.524s     (CPU, S=1024, dim=512, kv_lora=256)
```

The gap widens with sequence length. **Set `attn_impl="naive"` for training runs.** Caveat: the
naive branch does `k_pe.expand(-1, -1, self.n_heads, -1)` with the *global* head count and cats it
onto a head-sharded DTensor — likely broken under TP. Use `naive` only when `tp = 1` (which is the
right choice for a 1.4B model anyway), and keep `absorb` for the TP matrix runs.

### M2 — KV cache buffers are allocated during training

`Attention.__init__` registers `kv_cache` (`max_batch_size x max_seq_len x kv_lora_rank`) and
`pe_cache` unconditionally, and `to_empty()` materializes them on the GPU. Training never sets
`use_kv_cache=True`. At `max_batch_size=2, max_seq_len=2048, kv_lora_rank=256` in fp32 that is
~4 MB + 0.5 MB per layer — small at this scale, but it scales with `max_seq_len` and is pure waste.

**Fix:** set `max_batch_size = 1` in the training config (one line), or allocate lazily.

### M3 — `get_peak_flops` has no consumer-GPU entries

`tools/utils.py:28`. No RTX 4090 / 5090 / A6000 / A40 branch, so they fall through to the A100's
312 TFLOPS and reported MFU is ~4x too low. Add: `"RTX 4090" -> 165e12` (or `82.6e12` if you want
MFU measured against the fp32-accumulate rate PyTorch actually achieves — pick one and note it),
`"RTX 5090" -> 209e12`, `"A6000"/"A40" -> 149.7e12`. Cosmetic, but you will be reading MFU all run.

### M4 — checkpoint retention will fill the disk

`keep_latest_k = 10`. A checkpoint of the §2 config is fp32 params (5.5 GB) + Adam `exp_avg` and
`exp_avg_sq` (11 GB) ≈ **16.6 GB**. Ten of those is 166 GB.

**Fix:** `keep_latest_k = 3`, and put `dump_folder` on a network volume.

### M5 — router uses softmax; DeepSeek-V3 uses sigmoid

`default_configs.py:19` sets `score_func="softmax"` while `MoEArgs` defaults to `"sigmoid"` and
the design (§D4, §3.1) specifies sigmoid top-k. With softmax and `route_norm=False`, the kept
top-k gate values sum to less than 1 and the sum varies per token, which changes the effective
residual scale. Either set `route_norm=True` or switch to `sigmoid`. Recommend `sigmoid` +
`route_norm=True`, matching V3.

### M6 — `ScaledDotProductAttentionWrapper` is a dead module

14-line file, a class with `__init__` and a docstring promising CP compatibility, no `forward`,
constructed as `self.inner_attention` on every `Attention` and never called. Either implement it
(see H2) or delete it — as it stands it advertises a capability the model does not have.

### L1 — `.gitignore` hides new files under `model/`

`.gitignore` contains `*model*`. Existing files are tracked and unaffected, but any **new** file
added under `distributed_framework/torchfeather/model/` is silently ignored — confirmed:
`model/moe/test.ipynb` shows as `!!` in `git status --ignored`. Also, `__pycache__/*.pyc` files
under `model/` are *tracked* (they show up in every `git status`).

**Fix:** narrow the pattern (`/models/`, `*.safetensors`, `*.bin`), and
`git rm -r --cached '*/__pycache__'`.

### L2 — `torch.compile` is on by default over a data-dependent MoE

`config.compile.enable = True`, `components = ["model", "loss"]`. MoE blocks are compiled with
`fullgraph=False`, but the token counts driving `_grouped_mm`'s `offs` are data-dependent, and the
line that makes that tolerable is commented out (`parallelize.py:172`,
`torch._dynamo.config.capture_scalar_outputs = True`). Expect long compile times and heavy
recompilation. **Start with `compile.enable = False`; turn it on as a measured experiment in
Stage 2.**

### L3 — dataloader runs tokenization in the training process

`ParallelAwareDataloader.__init__` never passes `num_workers`, so it is 0 and every batch is
tokenized inline in the training loop. At the throughputs in §5 this is probably not the
bottleneck, but `metrics` reports `time_metrics/data_loading(%)` — watch it, and add
`num_workers=2` if it exceeds ~10%.

### Still open from the prior analysis

`B4` (`cast(nn.ModuleDict, model.layers)` on a `ModuleList` in `activation_checkpoint.py:155` and
`parallelize.py:173`) and `B5` (`_apply_full_ac` ignores `ac_config`; `model_compile_enabled` is
accepted and unused) are both still present. Both harmless.

**`B1` is fixed** — `pipeline_parallel.py:298` now reads `pp_degree = pp_mesh.size()`. The
analysis doc is stale on this point.

---

## 2. The model to train

Total parameters drive memory; active parameters drive FLOPs. Measured on the meta device with
`vocab_size=102400, kv_lora_rank=256, qk_nope=64, qk_rope=32, v_head=64, seq_len=2048`:

| cfg | dim | layers | heads | moe_inter | E | top_k | total | active | GFLOP/tok |
|-----|-----|--------|-------|-----------|---|-------|-------|--------|-----------|
| A | 1024 | 12 | 8 | 512 | 32 | 4 | 0.855B | 0.327B | 1.52 |
| B | 1024 | 12 | 8 | 704 | 32 | 4 | 1.089B | 0.362B | 1.73 |
| **C** | **1024** | **16** | **8** | **704** | **32** | **4** | **1.382B** | **0.413B** | **2.10** |
| D | 1280 | 16 | 10 | 896 | 32 | 4 | 2.123B | 0.582B | 3.02 |
| E | 1536 | 18 | 12 | 896 | 32 | 4 | 2.836B | 0.755B | 4.01 |
| F | 1024 | 16 | 8 | 352 | 64 | 6 | 1.383B | 0.379B | 1.90 |

**Take C.** Reasons: 1.4B total sits in the middle of the 1–3B target; 32 experts divide cleanly
by `ep ∈ {2,4,8}`; `dim=1024` and `moe_inter_dim=704` are both multiples of 8, which
`torch._grouped_mm` requires for 16-byte-aligned strides; 8 heads divide by `tp=2`; and at
0.41B active it is cheap enough that the budget buys a defensible number of tokens.

Config F (64 experts, top-k 6) is the closer analogue of real DeepSeek-V3 routing and is the one
to pick if the *MoE* is more interesting to you than the *model*. It costs nothing extra.

Memory at `ep=8, dp_shard=8` on eight 24 GB cards: expert params 4 of 32 per rank, unsharded
(`efsdp = 1`) = 138M params x 16 B = 2.2 GB; everything else FSDP-sharded 8 ways = 0.55 GB.
**~2.8 GB of state per GPU**, leaving ~20 GB for activations. Comfortable.

```python
def get_torchfeather_1b_model_args() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        vocab_size=102400, dim=1024, inter_dim=2816, moe_inter_dim=704,
        n_layers=16, n_dense_layers=1, n_heads=8,
        moe_args=MoEArgs(num_experts=32, num_shared_experts=1, top_k=4,
                         score_func="sigmoid", route_norm=True,
                         score_before_experts=False, load_balance_coeff=1e-3),
        q_lora_rank=0, kv_lora_rank=256,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        max_seq_len=2048, original_seq_len=4096, mscale=1.0,
        max_batch_size=1,            # M2: KV cache is unused in training
        attn_impl="naive",           # M1: 3-4x cheaper attention; requires tp=1
        moe_enabled=True,
    )
```

### Vocabulary note

102400 tokens x 1024 dims x 2 (embedding + output head) = **210M parameters, 15% of the model**,
on a tokenizer built for Chinese and English while the corpus is English-only. The output
projection alone contributes 0.63 of the 2.10 GFLOP/token. A 32k–50k English tokenizer would buy
roughly 12% throughput and 100M parameters back. The loader reads any HF `tokenizer.json`
directory, so this is a config change — but the `tokenizer_config.json` must expose `bos_token`
and `eos_token` as dicts with a `content` key, which many repos do not. **Not recommended for
the first run**; note it as the obvious second-run improvement.

---

## 3. Data

The current source **works with one change**. `_load_fineweb_dataset` hardcodes `name="default"`,
which is the full FineWeb — 15T tokens, ~93 TB, 114 configs' worth of shards. Resolving that
file list over the network before step 1 is slow and pointless at this scale.

Confirmed available: `HuggingFaceFW/fineweb` and `HuggingFaceFW/fineweb-edu` both publish
`sample-10BT`, `sample-100BT`, `sample-350BT` alongside `default`.

**Use `HuggingFaceFW/fineweb-edu`, config `sample-10BT`** (~10B tokens, ~28 GB). It is the
educational-quality filter of FineWeb and gives a visibly better loss curve for a small model at
a fixed token budget — which matters when the budget only buys ~1.3B tokens.

```python
# datasets/hf_datasets.py
def _load_fineweb_dataset(dataset_path: str, split: str, name: str = "sample-10BT"):
    ...
    return load_dataset(dataset_path, name=name, split=split, streaming=True, ...)

DATASETS = {
    "fineweb": DatasetConfig(path="HuggingFaceFW/fineweb", ...),
    "fineweb_edu": DatasetConfig(
        path="HuggingFaceFW/fineweb-edu",
        loader=partial(_load_fineweb_dataset, split="train", name="sample-10BT"),
        sample_processor=_process_pretrain_record,
    ),
}
```

Stream it — do not download. 1.3B tokens is ~6 GB of raw text, a fraction of the sample, and
streaming state (`_data.state_dict()`) survives checkpoint resume once F1 is fixed.

**Alternatives if FineWeb misbehaves:** `HuggingFaceTB/smollm-corpus` (`fineweb-edu-dedup` +
`cosmopedia-v2`, purpose-built for sub-2B models, strongest option for quality per token);
`allenai/c4` config `en` (older, larger, noisier). Both are single-text-column and work with the
existing `_process_pretrain_record`.

**Tokenizer, download once:**

```bash
huggingface-cli download deepseek-ai/deepseek-moe-16b-base \
  tokenizer.json tokenizer_config.json \
  --local-dir distributed_framework/assets/hf/deepseek-moe-16b-base
```

---

## 4. The plan

### Stage 0 — fix and verify locally. Cost: $0. Time: half a day.

No GPU needed. Order matters: each step is independently checkable.

1. `requirements.txt` + install (`torch`, `datasets`, `tokenizers`, `torchdata`, `loguru`,
   `huggingface_hub`, `wandb`).
2. **F1** dedent the four `HuggingFaceDataset` methods. Verify: iterate 3 batches, assert shapes
   `(B, seq_len)` and `label[i] == input[i+1]`.
3. **F4** `torch.bincount` at both count sites. Verify: `counts.dtype == torch.int64`.
4. **F5** download tokenizer assets; make the wandb logger fall back to `BaseLogger`.
5. **F2/F3** write `clip_grad_norm_` and `set_pg_timeouts`.
6. **H4** wire `n_dense_layers` into `DecoderLayer`.
7. **M1/M2/M5** the config changes from §2. **M3** the peak-FLOPS table. **M4** `keep_latest_k=3`.
8. Re-run the MoE regression gate: design §9 checks 1, 4, 6 (`atol=5e-2` bf16).
9. **Close L1 from the MoE design** — run the multi-rank CPU/gloo EP harness again, now with F4
   fixed in the *source* rather than patched in the harness:
   `torchrun --nproc_per_node=4 <ep_harness>` on the gloo backend. All six assertions must pass
   without local patches. This is the last MoE item and it costs nothing.
10. Full-trainer CPU dry run, 5 steps, `world_size=1`, gloo, `steps=5`,
    `compile.enable=False`, tiny config. This is the step that finds whatever F6 turns out to be.

**Gate:** loss is finite and decreasing over 5 steps; a checkpoint writes and reloads.

### Stage 1 — single GPU smoke. 1x RTX 4090. ~2 h. **~$0.70**

The point is to convert every unknown into a measured number before spending on eight cards.

```bash
torchrun --standalone --nproc_per_node=1 -m torchfeather.train
# TORCHFEATHER_CONFIG=tf1b_smoke  steps=100  local_batch_size=2  seq_len=2048
```

Measure and record:

- **Does `torch._grouped_mm` run on this GPU?** The 4090 is sm89; the fast grouped-GEMM path is
  Hopper-only. `libtorch_cuda.so` in torch 2.13 exports `_grouped_mm_fallback`, so it should
  degrade rather than raise — but confirm, because if it raises, the reference loop is still in
  `moe.py:122-139` and needs uncommenting. **Check this in the first five minutes.**
- **Is FlashAttention actually selected?** Run
  `python -m torchfeather.minimal_examples.verify_flash_attention` before the trainer. It
  forces `sdpa_kernel([SDPBackend.FLASH_ATTENTION])`, so a rejected constraint raises rather
  than degrading to math in silence, and prints flash-vs-math peak memory.
- `tokens/s`, `tflops`, `mfu(%)`, `memory/max_reserved(GiB)` from the metrics line.
- `time_metrics/data_loading(%)` — the L3 check.
- Checkpoint save wall time and on-disk size.
- Kill the process at step 60 and restart. It must resume at 51 (interval 50), same loss.

**Gate:** 100 steps, loss down, resume works, tok/s recorded. Everything after this is sized off
that number.

### Stage 1 results — RUN 2026-09-04, gates passed, one bug found

Hardware: 1x RTX 4090 (sm89, 24 GiB, driver 570.169), RunPod secure, `$0.74/h`.
Config `tf1b_smoke`: 4 layers, 440M total / 258M active, seq 2048, local batch 2, 100 steps.
Total RunPod spend for the stage: **~$0.27** (16 min live plus two dead pods).

**Both gates pass.**

| gate | result |
|---|---|
| `torch._grouped_mm` on sm89 | **OK**, and bit-matches the reference loop. The Hopper-only worry was unfounded; the commented loop in `moe.py` stays commented |
| FlashAttention actually selected | **PASS** under forced `sdpa_kernel([FLASH_ATTENTION])` |
| flash vs math peak memory | 15.42 MiB vs 27.25 MiB on the probe shapes |
| absorb path rejected by flash | confirmed, with the kernel's own reason: `requires q,k,v ... less than or equal to 256. Got Query.size(-1): 288` |
| bf16 numerics on CUDA | maxdiff 7.81e-03, inside the 5e-2 bf16 budget |

**Measured throughput and memory.**

| metric | value |
|---|---|
| throughput | 25,000-28,500 tok/s steady state (1 GPU) |
| achieved | ~27 TFLOPS |
| MFU | **30-34%**, against a correct 82.6 TFLOPS 4090 basis (the M3 fix) |
| peak memory | 12.60 GiB of 23.5 (53.6%) |
| OOMs / alloc retries | 0 / 0 |
| loss over 100 steps | 11.5366 -> 7.3791 |
| checkpoint size | 5.0 GB for the 440M smoke model |
| checkpoint save | 3.51 s |
| checkpoint load | 4.56 s |

MFU of 30-34% is far above the 12-15% this plan projected. Note it is a *single-GPU*
number with no collectives; the 8-GPU figure is what Stage 2 config 1 exists to measure,
and on 4090s without NVLink it will be lower.

**Sizing implication for Stage 3.** At 2.09 GFLOP/token and ~27 TFLOPS/GPU the
compute-bound ceiling is ~12,900 tok/s/GPU, so ~103k tok/s across eight cards --
about 4x this plan's 26k assumption. If even half of that survives the collectives,
Stage 3 buys 2B+ tokens rather than 1.3B in the same 12 hours. **Do not raise `steps`
off this number** -- take it from Stage 2 config 1, which has the real communication cost.
`local_batch_size=8` at 16 layers is still untested and may need the documented fallback.

### F7 — a killed async checkpoint silently restarts training from step 1

Found by the Stage 1 resume test, and it is the exact Stage 3 failure mode.

Resume from a *complete* checkpoint works:

```
Loading the checkpoint from ./outputs/tf1b_smoke/checkpoint/step-50.
Finished loading the checkpoint in 4.56 seconds.
Training starts at step 51
step: 51  loss:  8.0491        <- continuing, not a fresh 11.5
```

But when the process is killed while an async save is still writing (5.0 GB takes
~3.5 s; the kill landed ~2 s after the save began), the `step-N` folder is left with no
`.metadata`. `_find_load_step` only accepts folders that have one, so it skips the torn
checkpoint -- correct -- and returns -1. `load()` then returns `False` **with no message**,
and the next line in the log is `Training starts at step 1`. Observed: a run killed at
step 64 restarted at step 1 with loss back at 11.5369.

On a 12-hour run this reads as a mysteriously reset loss curve hours later.

**Fix applied:** `CheckpointManager.load` now distinguishes "nothing saved yet" from
"everything saved is unusable" and logs a loud warning naming the orphaned folders.

**Operational consequence:** `keep_latest_k >= 2` is not optional, it is the thing that
makes a torn latest checkpoint survivable -- the loader falls back to the previous
complete one. The Stage 3 config already sets 3. Keep it.

### Stage 2 — the parallelism matrix. 8x RTX 4090. ~3 h. **~$8.20**

This is the part that teaches the mechanisms, and it is the cheapest part. `world_size = 8`,
same seed, same global batch, `steps=50`, `checkpoint.enable=False` except where noted. Compare
the loss curve of each against config 1.

| # | dp_shard | tp | ep | etp | pp | exercises |
|---|----------|----|----|-----|----|-----------|
| 1 | 8 | 1 | 1 | 1 | 1 | FSDP baseline, no expert plan |
| 2 | 8 | 1 | 2 | 1 | 1 | `ExpertParallel`, EFSDP wrapping |
| 3 | 8 | 1 | 8 | 1 | 1 | ep == world; the `Shard(1)` fallback in `apply_fsdp` |
| 4 | 4 | 2 | 1 | 1 | 1 | `TensorParallel` expert plan (needs `attn_impl="absorb"`) |
| 5 | 4 | 2 | 4 | 1 | 1 | EP borrows TP -> `ReordererSequenceParallel` |
| 6 | 4 | 2 | 2 | 2 | 1 | `ExpertTensorParallel` |
| 7 | 2 | 2 | 2 | 1 | 2 | PP + EP (`checkpoint.enable=False`, see H3) |

All seven satisfy `ParallelDims._validate`. Config 3 is the interesting one: `efsdp * ep = 8 > 4`
local experts, which is the only path that reaches the `Shard(1)` branch of `apply_fsdp`.

Also run here, because they are cheap and only possible with GPUs:

- **Checkpoint resharding.** Save at `ep=2`, restore at `ep=4`, confirm identical next-step loss.
  This is what silently breaks if a shard dim is declared wrong.
- **Expert balance.** `tokens_per_expert` variance should fall. Monotone growth in one expert
  means the bias update or its all-reduce is broken.
- **`compile.enable=True`** on config 1 and 2 — measure, then decide (L2).
- **Skip `cp`.** It is wrong (H2), not slow.

**Note on 4090 collectives:** consumer Ada blocks peer-to-peer over PCIe, so NCCL stages through
host memory. TP and EP all-to-all will look bad. That is a real and instructive lesson, not a
bug — but if you want to see what NVLink does, run configs 4–6 once on 4x A100 SXM ($5.56/h,
20 minutes, ~$2).

Set on the pod: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` if NCCL hangs during init on 4090s.

### Stage 2b — two nodes, optional. 2x 1x RTX 3090. ~2 h. **~$0.90**

Purely to internalize the launch mechanics your notes call out. Two separate pods, connected over
their public IPs. Throughput will be terrible; that is not the point.

```bash
# node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
         --master_addr=<node0_public_ip> --master_port=29500 -m torchfeather.train
# node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
         --master_addr=<node0_public_ip> --master_port=29500 -m torchfeather.train
```

`NCCL_SOCKET_IFNAME=eth0`, `NCCL_DEBUG=INFO`, and expose the port on node 0. Run 50 steps of
config 1. RunPod's Instant Clusters give you real multi-node with fast interconnect; they are also
where the money goes, so read a SLURM script instead.

### Stage 3 — the real run. 8x RTX 4090. ~12 h. **~$32.60**

```
parallelism:      dp_shard=8, ep=8, etp=1, tp=1, pp=1, cp=1
training:         seq_len=2048, local_batch_size=8, global_batch_size=64
                  -> 131,072 tokens per optimizer step, grad_accum=1
                  dtype=float32, mixed_precision_param=bfloat16
optimizer:        AdamW lr=4e-4, betas=(0.9, 0.95), wd=0.1, eps=1e-8
lr_scheduler:     warmup=300, cosine, decay_ratio=None, min_lr_factor=0.1
steps:            10000        -> 1.31B tokens
activation_ckpt:  selective / op
compile:          per the Stage 2 measurement
checkpoint:       interval=150, keep_latest_k=3, async, folder on the network volume
```

If it OOMs: `local_batch_size=4, global_batch_size=64` (grad_accum 2, identical token math).

**Set `steps` from the measured number, not from this document:**

```
steps = budget_hours * 3600 * measured_aggregate_tok_s / 131072
```

At an aggregate 26k tok/s across the eight cards, 12 h is 1.12B tokens ≈ 8,600 steps. The
plausible band is 20k–35k tok/s, so 0.9B–1.5B tokens. Set `steps` to the low end and let it
finish early rather than getting cut off mid-decay — the cosine schedule only reaches `min_lr`
if the run completes, and a truncated cosine leaves the model at a high LR.

### Stage 4 — show it works. 1x RTX 4090. ~1 h. **~$0.35**

Load the final checkpoint, run `DeepSeekV3Model.generate` (which calls `absorb_weights()` — the
inference path this whole MLA implementation exists for), and sample 20 completions. Record
final loss, the loss curve, per-expert utilization, and MFU. That is the deliverable.

---

## 5. RunPod specifics

Live pricing, community cloud, 2026-09-03:

| GPU | VRAM | $/h community | $/h secure | availability | max/pod |
|-----|------|---------------|------------|--------------|---------|
| **RTX 4090** | 24 GB | **0.34** | 0.74 | **HIGH** | 8 |
| RTX A5000 | 24 GB | 0.16 | 0.27 | LOW | 10 |
| RTX 3090 | 24 GB | 0.22 | 0.50 | LOW | 6 |
| A6000 | 48 GB | 0.33 | 0.53 | LOW | 3 |
| A40 | 48 GB | 0.35 | 0.44 | LOW | 1 / 10 |
| L40S | 48 GB | 0.79 | 0.99 | MEDIUM | 8 |
| A100 SXM 80 | 80 GB | 1.39 | 1.59 | MEDIUM | 8 |
| H100 SXM | 80 GB | 2.69 | 3.29 | LOW | 8 |

**Pick the RTX 4090.** It is the only card at HIGH availability, it is the cheapest TFLOPS/$ on
the board, and 8 of them fit in one pod. The A100 SXM is the alternative if the collectives'
behaviour is what you want to study — NVLink instead of host-staged PCIe — at 4x the price.

**Pod setup**

- Template: an official RunPod PyTorch image (CUDA 12.8 or 13.0 — both are available for the
  4090). Do not build a container; `pip install -r requirements.txt` on boot is faster than a
  build/push cycle.
- **Network volume, 150 GB, in the same data center as the pod.** ~$0.07/GB/month ≈ $10.50/mo,
  so ~$1 for three days. This is the single most important operational decision: it is the only
  storage that survives a pod dying.
- Mount it and set `config.job.dump_folder` under the mount. Three checkpoints at 16.6 GB each is
  50 GB; the volume also holds the HF cache.
- Container disk 40 GB is enough for the image and pip.

**Handling interruption** — the honest answer is that the checkpointer already does the hard part.

1. `CheckpointManager.load(step=-1)` scans the folder, takes the highest `step-N` that has a
   `.metadata` file, and restores model, optimizer, LR schedule, dataloader position and
   `train_state`. Resume is automatic; you re-launch the same command.
2. At `interval=150` and ~5 s/step, the maximum work lost to a kill is **~12 minutes**.
3. Wrap the launcher so a crash restarts itself:
   ```bash
   until torchrun --standalone --nproc_per_node=8 --max-restarts=3 -m torchfeather.train; do
     echo "trainer exited $?; restarting in 30s"; sleep 30
   done
   ```
   `--max-restarts` handles a single rank dying; the `until` loop handles the whole job dying.
4. **You cannot change GPU count on resume.** `ParallelAwareDataloader.load_state_dict` asserts
   `self.dp_world_size == state_dict["world_size"]`. If the 8-GPU pod is gone and only 4 are
   available, you must either wait, or resume with `checkpoint.load_step=0` (model-only, which
   the code supports) and accept losing the optimizer state and data position.
5. Prefer **on-demand** community pods over interruptible/spot for Stage 3. The premium is small
   next to a night of lost training, and community on-demand is not preemptible by another bidder.
6. Log to `WANDB_MODE=online` so the curve survives the pod. If you would rather not, `offline`
   plus the network volume works — just make sure `metrics.save_folder` is under the mount.

**Cost ledger**

| stage | hardware | rate | hours | cost |
|-------|----------|------|-------|------|
| 0 | local CPU | — | — | $0.00 |
| 1 | 1x RTX 4090 | $0.34/h | 2 | $0.68 |
| 2 | 8x RTX 4090 | $2.72/h | 3 | $8.16 |
| 2b | 2x 1x RTX 3090 | $0.44/h | 2 | $0.88 |
| 3 | 8x RTX 4090 | $2.72/h | 12 | $32.64 |
| 4 | 1x RTX 4090 | $0.34/h | 1 | $0.34 |
| — | 150 GB network volume | ~$0.015/h | ~72 | $1.05 |
| | | | **total** | **$43.75** |

Trim Stage 2b and one hour of Stage 3 to land under $40. Add ~$2 to run Stage 2 configs 4–6 on
A100 SXM for the NVLink comparison.

---

## 6. Go / no-go gates

| after | gate | if it fails |
|-------|------|-------------|
| Stage 0 | 5 CPU steps, finite decreasing loss, checkpoint round-trips | do not rent anything |
| Stage 0 | gloo EP harness: 6/6 assertions, no harness patches | F4 is not actually fixed |
| Stage 1 min 5 | `torch._grouped_mm` runs on sm89 | uncomment the loop in `moe.py:122-139` |
| Stage 1 min 5 | `verify_flash_attention` passes under forced `FLASH_ATTENTION` | read the raised constraint; MFU and max seq len both drop if it stays on math |
| Stage 1 | tok/s and GiB recorded; resume verified | size Stage 3 off the real number |
| Stage 2 #1 | FSDP baseline loss curve | nothing else is comparable without it |
| Stage 2 #2,5,6 | curves track #1 within noise | a wrong `Partial()`/`Replicate()` on the gate; see MoE design §15 |
| Stage 2 | `tokens_per_expert` variance falling | bias update or its all-reduce is broken |
| Stage 3 h1 | loss < 7.0, no NaN, MFU stable | stop the pod, do not burn 11 more hours |

---

## 7. What this deliberately does not do

- **No context parallelism.** Attention now routes through SDPA, which was the prerequisite, but
  CP is still blocked: `train.py` passes a `freqs_cis` buffer that does not exist in this repo,
  and `RotaryEmbedding._rotate` indexes by absolute position, which CP's load-balanced
  round-robin sharding breaks. See H2. About half a day, all in `rope.py`.
- **No multi-thousand-GPU anything.** Per your notes: read the Megatron/DeepSpeed configs.
- **No Chinchilla-optimal run.** 1.3B tokens for a 0.41B-active model is ~3 tokens/param against
  a ~20 target. Reaching 8B tokens costs roughly $200 at these rates. The deliverable here is a
  *working distributed training system with a real loss curve*, not a competitive model.

### FlashAttention — corrected

An earlier revision of this section said FlashAttention was out of scope. That was the
audit-time state; H2 changed it. Current state:

- **The training path gets flash.** `naive`, no KV cache, so `is_causal=True` with no
  `attn_mask`, and V zero-padded up to `qk_head_dim` giving `Dq == Dk == Dv == 96` — inside the
  256 head-dim cap. Activations are bf16 on both paths that matter (`fully_shard`'s
  `MixedPrecisionPolicy`, and `torch.autocast` for single-device/DDP), so the dtype constraint
  holds too.
- **Inference decode does not, by design.** `absorb` contracts over
  `kv_lora_rank + qk_rope_head_dim = 288`, past the cap, and passes an explicit `attn_mask` for
  the cached prefix, which flash does not accept in any form. Absorb exists to shrink the KV
  cache, not to maximize per-token throughput; it falls back to the math backend and that is
  the correct trade.
- **TP or PP without FSDP does not.** `maybe_enable_amp` disables mixed precision in that
  combination, leaving fp32, which flash refuses. No Stage 2 matrix config hits it — all seven
  have `dp_shard > 1`, so FSDP supplies bf16. Worth knowing before adding a config that doesn't.
- **Selection is asserted, not assumed.** Calling SDPA is not the same as getting flash; the
  dispatcher degrades to the math backend silently, which would quietly restore the
  537 MB/layer score tensor this was meant to remove.
  `torchfeather/minimal_examples/verify_flash_attention.py` forces
  `sdpa_kernel([SDPBackend.FLASH_ATTENTION])` so a failed constraint raises instead, and prints
  flash-vs-math peak memory. Numerics and gradients pass on CPU (`maxdiff 4.77e-07`);
  **the backend assertion is a Stage 1 gate** — it needs a GPU.
