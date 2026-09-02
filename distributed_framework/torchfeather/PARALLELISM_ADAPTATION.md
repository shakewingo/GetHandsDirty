# Parallelism Adaptation — Status & TODO

Tracks adapting the torchtitan-derived parallelism code (`model/parallelize.py`,
`distributed/`, `train.py`) to the hand-written model in `model/model.py`.

**Verification status: none of this has been executed.** No `torch` is installed on
the current machine, so every change below is reviewed and syntax-checked only. The
smoke tests in "How to verify" are the gate before trusting any of it.

---

## Phase 1 — Import and run single-GPU ✅ done

| # | Issue | Fix | Where |
|---|---|---|---|
| 1 | `class Attenti` typo made the model unimportable (`NameError` at `DecoderLayer.__init__`) | renamed to `Attention` | `model/model.py` |
| 2 | `apply_moe_ep_tp` called `.values()` on an `nn.ModuleList`; broke **every** TP run, MoE or not | iterate the list directly | `distributed/expert_parallel.py` |
| 3 | No `init_weights(buffer_device)`; meta-build + `to_empty()` left params *and* buffers uninitialized | added `DeepSeekV3Model.init_weights` | `model/model.py` |
| 4 | `get_params_and_flops` vs `get_nparams_and_flops` name mismatch | call site renamed | `train.py` |
| 5 | Model owned the loss and returned `CausalLMOutputWithPast` | `forward` returns a logits tensor; trainer owns the loss | `model/model.py` |

### Notes on #3 (init_weights)

The old `self.apply(self._init_weights)` ran inside `__init__`, which under
`torch.device("meta")` wrote nothing, and `to_empty()` then handed back uninitialized
memory. Three things now happen in `init_weights` instead:

- **`nn.RMSNorm` weights are initialized to ones.** These were previously missed
  entirely — `_init_weights` only handled `Linear` and `Embedding`. On a real device
  `nn.RMSNorm.__init__` sets them, but after `to_empty()` they were garbage. This was
  a silent correctness bug in every meta-device (i.e. every distributed) run.
- **RoPE cos/sin tables are rebuilt** on `buffer_device` via
  `RotaryEmbedding.init_weights`. They are now `persistent=False`, since they are a
  deterministic function of position and do not belong in a checkpoint.
- **KV cache buffers are zeroed** via `Attention.init_kv_cache`.

`__init__` still calls `init_weights()` when not on the meta device, so standalone use
(`python -m torchfeather.model.model`, `verify_absorb.py`) is unchanged.

### Notes on #5 (loss contract)

`forward(x, use_kv_cache=False, start_pos=0, last_token_only=False) -> Tensor`.

This one change unblocks three things at once:
- `train.py:381` already calls the model with no labels and passes the result to
  `components/loss.py::cross_entropy_loss`.
- PP requires stage outputs to be **tensors**; a dataclass cannot cross a stage boundary.
- `loss_parallel` needs the vocab-sharded logits DTensor to reach an `F.cross_entropy`
  running inside `train_context`, which is the trainer's scope, not the model's.

`last_token_only` replaces the old "if labels is None, project only the last position"
inference shortcut, which is now an explicit flag set by `generate` rather than an
inference inferred from the absence of labels.

Note `components/loss.py` uses `IGNORE_INDEX = -100` where the old in-model loss used
`ignore_index=0`. The packed dataloader (`datasets/hf_datasets.py`) emits no padding at
all, so neither value currently fires — but -100 is now the one in effect.

---

## Phase 2 — TP correctness ✅ done

| # | Issue | Fix | Where |
|---|---|---|---|
| 6 | `"attention.wk v_a"` plan key had a stray space, so `NoParallel` attached to nothing | `"attention.wkv_a"` | `model/parallelize.py` |
| 7 | `PrepareModuleInput(input_layouts=(Shard(1), Replicate()))` crashed on the `mask=None` decode path | `(Shard(1), None)` / `(Replicate(), None)` | `model/parallelize.py` |
| 8 | `_split_heads` compared against `Shard(-1)`, which never matches | derive local head count from the local shard | `model/model.py` |
| 9 | `apply_rotary_emb` multiplied a head-sharded DTensor by plain cos/sin tables | rotate the local shard, re-wrap | `model/rope.py` |
| 10 | `wkv_b` local head count derived from `device_mesh.size()` | derive from the weight shard's own shape | `model/model.py` |

### Notes on #8 — the subtle one

`DTensor.redistribute` **normalizes a negative shard dim to its positive index**. A
`ColwiseParallel` output on a 3-D activation therefore arrives as `Shard(2)`, not
`Shard(-1)`. The old comparison `x.placements[-1] == Shard(-1)` was always `False`, so
`local_heads` fell back to the full `n_heads` and the following `view` would have
raised a shape error on a tensor holding only `n_heads/tp` heads.

`_split_heads` now reads `local.shape[-1] // head_dim` off the shard it actually holds.
That is correct for a replicated input too, and does not assume the mesh is exactly the
TP dim. `_is_last_dim_shard` accepts either spelling of the trailing shard, because
`from_local` preserves whatever it was handed while `redistribute` normalizes.

### Notes on #9 — found while fixing, not in the original audit

DTensor refuses to mix a `DTensor` with a plain `torch.Tensor` in a binary op unless the
plain side is a 0-d scalar. `q_pe * cos` with head-sharded `q_pe` and a plain
`(1, S, 1, D)` table would have raised
`got mixed torch.Tensor and DTensor` on the first TP forward. RoPE is per-position and
rank-invariant, so it now runs on the local shard and re-wraps with the incoming
placements. This also sidesteps needing DTensor to support the strided slicing in
`rotate_adjacent`.

### Also cleaned up

- Removed the unused `DeepSeekV3Model.rotary_emb`. It was dead (every `Attention` builds
  its own) and, being an `nn.Module` child absent from every PP stage's FQN list,
  `_build_stage_from_modules` would have set it to `None` on every stage anyway.

---

## Phase 3 — Fence off what is out of scope ⬜ todo

Goal: fail loudly at setup instead of silently wrong or deep in the trainer.

- [ ] **Assert `parallel_dims.cp == 1`** in `parallelize_deepseekv3`. CP is structurally
      impossible today — see "Known limitations" below.
- [ ] **Guard `absorb_weights()` against DTensors.** `model.py` does
      `self.wkv_b.weight.view(...)` and `self.wo.weight.view(self.dim, n_heads, v_head_dim)`,
      which eager DTensor cannot do across a sharded dim. Worse, it builds fresh plain
      `nn.Linear` layers (`wq_abs`, `wo_abs`) that are **not in the TP plan**, so `wo_abs`
      would skip `RowwiseParallel`'s all-reduce and produce **silently wrong** output.
      `generate()` calls it unconditionally. Cheapest fix: no-op when any param is a
      `DTensor`, falling back to the per-call einsum path (already TP-aware).
- [ ] **Make `generate` refuse TP**, or route it through the non-absorbed path.
      `last_token_only` slices `x[:, [-1], :]` on a seq-sharded DTensor, which is
      untested.
- [ ] **Assert `attn_impl == 'absorb'` when `tp > 1`.** The `naive` path does
      `torch.cat([k_nope, k_pe.expand(...)], dim=-1)` mixing a `Shard(2)` DTensor with a
      `Replicate()` one; DTensor would redistribute (all-gathering the heads), which is
      wasteful at best and wrong at worst. `absorb` is the default, so this only fences
      the reference implementation.

## Phase 4 — Inference under parallelism ⬜ todo

- [ ] **KV cache sharding on the `naive` path.** `k_cache`/`v_cache` carry an `n_heads`
      dim but are plain unsharded buffers; writing a head-sharded DTensor into them will
      fail. The `absorb` path is safe — `kv_cache`/`pe_cache` hold the head-shared latent.
- [ ] **TP-aware weight absorption**: absorb on local shards and register the fused
      Linears into the TP plan. This is the performant answer to the Phase 3 guard.
- [ ] **Lazy KV cache allocation.** Both caches are allocated unconditionally at
      `max_batch_size x max_seq_len` even during training, where they are never used.

---

## Known limitations

**Context parallel cannot work without restructuring.** Three independent blockers:

1. RoPE lives inside `Attention` and derives position from `start_pos` plus the *local*
   sequence length. Under CP the sequence is split across ranks, so every rank above 0
   applies the wrong absolute positions.
2. `train.py:333` lists `m.freqs_cis` in `cp_buffers`; the model has no such attribute.
   Torchtitan keeps `freqs_cis` as a model-level buffer precisely so CP shards it in
   lockstep with the input.
3. `ScaledDotProductAttentionWrapper` (`model/attention.py`) is a stub — an `__init__`
   with no `forward`. It is instantiated as `Attention.inner_attention` and never called.
   CP works by intercepting SDPA or flex_attention; the hand-rolled matmul score path
   cannot be intercepted at all.

Fixing (1) and (2) together — hoisting RoPE to a single model-level buffer — would also
remove the current duplication of `n_layers` separate cos/sin table copies, one per
`Attention`. (3) is a larger change: it means routing the score math through SDPA.

**MoE is not implemented.** `DecoderLayer.moe_enabled` is hard-coded `False`, so
`apply_moe_ep_tp` iterates and skips every layer. When MoE lands, note that
`expert_parallel.py` expects `moe.shared_experts.w1/w2/w3`, whereas `FeedForward` here
uses `gate_proj`/`up_proj`/`down_proj` — the naming will need aligning.

**Missing modules copied from the reference.** Known and expected; `train.py` and
`parallelize.py` import these but they do not exist yet:
`torchfeather.distributed.model_parallel` (`apply_ddp`, `apply_fsdp`),
`torchfeather.components.{checkpoint,lr_scheduler,metrics,optimizer}`,
and the whole `torchfeather.tools` package.

---

## How to verify

Nothing below has been run. In order:

```bash
# 1. Single-GPU forward + the custom init scheme (Phase 1)
python -m torchfeather.model.model

# 2. Absorption still matches the unabsorbed path (unchanged by this work)
python -m torchfeather.model.verify_absorb

# 3. Meta-build round-trip, the path train.py actually takes (Phase 1, #3)
#    Build under meta -> to_empty -> init_weights, then assert the RoPE tables and
#    every RMSNorm weight are finite and non-garbage.

# 4. TP numerical equivalence (Phase 2) -- needs 2 GPUs
#    torchrun --nproc_per_node=2 with tensor_parallel_degree=2, compare the forward
#    against the single-GPU logits from step 1 at the same seed.
```

Step 3 is the one that matters most for Phase 1 and step 4 for Phase 2; steps 1 and 2
only prove nothing regressed for the non-distributed path.
