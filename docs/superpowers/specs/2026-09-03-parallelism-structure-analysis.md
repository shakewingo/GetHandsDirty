# Parallelism Execution Structure: Analysis

Date: 2026-09-03
Scope: `distributed_framework/torchfeather`
Status: **Analysis complete (§§1–5). Design proposed (§7), not approved.** No code changed.
Covers: structure analysis, the three unwritten functions blocking the PP path, the chosen
restructure, and a side discussion on what transfers to inference (§8).

## 0. What this is

A read of every file that touches parallelism, written to answer one question: *why is it hard
to see how PP / TP / EP / ETP / FSDP / CP / DP fit together, and what would make it clear?*

Files read in full: `train.py`, `model/parallelize.py`, `distributed/parallel_dims.py`,
`distributed/pipeline_parallel.py`, `distributed/model_parallel.py`,
`distributed/expert_parallel.py`, `distributed/activation_checkpoint.py`,
`distributed/utils.py`, `distributed/__init__.py`, `config/job_config.py`,
`config/default_configs.py`, `model/model.py`, `model/moe/moe.py`, `model/moe/utils.py`,
`components/optimizer.py`. Skimmed: `components/metrics.py`, `components/dataloader.py`,
`datasets/hf_datasets.py`.

---

## 1. The actual execution flow

There is exactly one path. Traced end to end:

```
train.py::__main__
 └─ get_config(TORCHFEATHER_CONFIG)          → JobConfig; every degree lives in .parallelism
 └─ Trainer.__init__(config)
     ├─ init_process_group("nccl")                                        [A] world exists
     ├─ _create_parallel_dims()  → ParallelDims(dp_replicate, dp_shard,
     │                                          cp, tp, pp, ep, etp, world_size)
     │      └─ __post_init__ → _validate()      degrees must multiply to world_size
     ├─ parallel_dims.world_mesh (property)                               [B] meshes exist
     │      └─ build_mesh()  → 1 flat world mesh + 3 unflatten views + 1 flatten
     ├─ dp_degree, dp_rank = get_mesh("batch").size(), .get_local_rank()
     ├─ build_hf_dataloader(dp_world_size=dp_degree, dp_rank=dp_rank)     [C] DP applied
     │      └─ split_dataset_by_node(ds, dp_rank, dp_world_size)
     ├─ with torch.device("meta"): model = DeepSeekV3Model(args)          [D] skeleton
     ├─ gradient_accumulation_steps = global_bs // (local_bs * dp_degree)
     ├─ if pp_enabled:                                                    [E] model transforms
     │      pipeline_llm(model, ..., parallelize_fn=parallelize_deepseekv3, loss_fn)
     │        ├─ pipeline_module_split()  → deepcopy + prune per stage → [PipelineStage], [nn.Module]
     │        ├─ for each part: parallelize_deepseekv3(part)      ← callback, per stage
     │        └─ build_pipeline_schedule(stages, loss_fn)
     │   else:
     │      model = parallelize_deepseekv3(model, ...)
     │        ├─ apply_non_moe_tp()          (parallelize.py, inline)
     │        ├─ apply_moe_ep_tp()           (distributed/expert_parallel.py)
     │        ├─ apply_ac()                  (distributed/activation_checkpoint.py)
     │        ├─ apply_compile()             (parallelize.py, inline)
     │        └─ apply_fsdp() | apply_ddp()  (distributed/model_parallel.py)
     ├─ m.to_empty(device) ; m.init_weights() ; m.train()                 [F] materialize
     ├─ build_optimizers_with_moe_load_balancing()   ← registers expert-bias step pre-hook
     ├─ build_lr_schedulers() ; CheckpointManager()
     ├─ train_context = get_train_context(loss_parallel_enabled, compiled_autograd)
     └─ maybe_enable_amp()                                                [G] runtime ctx built
 └─ Trainer.train()
     └─ loop: train_step(data_iterator)
         ├─ pull `gradient_accumulation_steps` microbatches
         ├─ dist_sum(local_valid_tokens, mesh("loss"))    → global_valid_tokens
         ├─ for each microbatch: forward_backward_step()
         │    ├─ create_context_parallel_ctx(mesh("cp"), buffers=[inputs, labels, freqs_cis])
         │    ├─ with train_context(cp_ctx):        ← loss_parallel + compiled_autograd + CP
         │    │    if pp:  pp_schedule.step(inputs, target=labels, losses=losses)
         │    │    else:   pred = model(inputs); loss = loss_fn(pred, labels)/gvt; loss.backward()
         ├─ if pp: p.grad.div_(global_valid_tokens)  for every param   ← PP loss-scale fixup
         ├─ clip_grad_norm_(..., pp_mesh, ep_enabled)
         └─ optimizers.step()   ← fires the MoE expert-bias pre-hook (all-reduce on "loss" mesh)
```

**The one-sentence version:** degrees → `ParallelDims` → meshes; then the model is transformed
once at construction (PP splits it, TP/EP/FSDP shard it), and two more parallelisms (CP, loss
parallel) are applied at *runtime* as context managers around every forward/backward, while DP
was already applied back at the dataloader by giving each rank different data.

---

## 2. The mesh model (this is the actual key)

`ParallelDims.build_mesh()` creates **one flat 1-D world mesh** and then re-interprets it three
different ways. Every parallelism reads from one of these views.

Invariant that makes all three consistent:
```
pp * dp_replicate * dp_shard * cp * tp == world_size
```

| View | Dim names | Sizes | Who reads it |
|---|---|---|---|
| `dataloading` | `pp, batch, cp, tp` | `pp, dp_r*dp_s, cp, tp` | dataloader (`batch`), CP ctx (`cp`), TP plans (`tp`), PP (`pp`) |
| `loss` | *(flattened)* | `batch × cp` | loss normalization, metrics, MoE expert-bias all-reduce |
| `dense` | `pp, dp_replicate, fsdp, tp` | `pp, dp_r, dp_s*cp, tp` | `apply_fsdp` for all non-expert params |
| `sparse` | `pp, dp_replicate, efsdp, ep, etp` | `pp, dp_r, dp_s*cp*tp/(etp*ep), ep, etp` | `apply_moe_ep_tp`, `apply_fsdp` for expert params |

**The legend nobody wrote down:**

- `batch` = `dp_replicate × dp_shard` — the DP world the dataloader shards over. CP is *not*
  in it, because every CP rank loads the *full* sequence and CP splits it later. (Hence the
  `ntokens //= cp` corrections at `train.py:301` and `:422`.)
- `fsdp` = `dp_shard × cp` — **CP ranks are FSDP shard ranks.** CP has no mesh of its own for
  weights; it folds into FSDP. This is why `fsdp_enabled` is `dp_shard_enabled or cp_enabled`.
- `loss` = `batch × cp` = `dp_replicate × dp_shard × cp` — every rank that holds a distinct
  slice of *tokens*. This is the reduction group for anything token-normalized.
- `efsdp` = `dp_shard × cp × tp / (etp × ep)` — **EP borrows from dp_shard and cp, plus tp when
  `etp == 1`.** That is exactly what the `_validate` assertions encode:
  - `etp == tp`: `ep % cp == 0` and `(dp_shard*cp) % ep == 0`
  - `etp == 1`: `ep % (cp*tp) == 0` and `(dp_shard*cp*tp) % ep == 0`

Two subtleties worth flagging:

- `_mesh_exist()` special-cases `"fsdp"` (always real backend, even at degree 1, so
  `fully_shard` can still apply `MixedPrecisionPolicy`) and `"efsdp"` (real iff `ep > 1`).
  Every *other* dim gets a `"fake"` backend at degree 1. Consequence:
  `get_optional_mesh("fsdp")` returns a real mesh at degree 1, but `get_optional_mesh("tp")`
  returns `None`. That asymmetry is load-bearing and easy to trip over.
- `build_mesh()` is lazy — triggered by the first `get_mesh` / `world_mesh` access.
  `train.py:103` (`_ = parallel_dims.world_mesh`) exists solely to force it.

---

## 3. Where each parallelism actually lives (the scatter map)

| Parallelism | Decided at | Applied at | Mechanism | Phase |
|---|---|---|---|---|
| DP (data) | `train.py:104` | `datasets/hf_datasets.py` | `split_dataset_by_node` | **data** |
| PP | `train.py:177` | `distributed/pipeline_parallel.py` | module surgery + `PipelineStage` + schedule | **build** |
| TP (dense) | `parallelize.py:71` | `parallelize.py::apply_non_moe_tp` | `parallelize_module` plan | **build** |
| TP (MoE) / EP / ETP | `parallelize.py:85` | `distributed/expert_parallel.py::apply_moe_ep_tp` | `parallelize_module` + custom `ParallelStyle`s | **build** |
| Activation ckpt | `parallelize.py:110` | `distributed/activation_checkpoint.py` | `ptd_checkpoint_wrapper` | **build** |
| `torch.compile` | `parallelize.py:114` | `parallelize.py::apply_compile` | per-block compile | **build** |
| FSDP / HSDP / EFSDP | `parallelize.py:116` | `distributed/model_parallel.py::apply_fsdp` | `fully_shard` | **build** |
| DDP | `parallelize.py:153` | `distributed/model_parallel.py::apply_ddp` | `replicate` | **build** |
| CP | `train.py:333` | `distributed/utils.py::create_context_parallel_ctx` | context manager | **runtime** |
| Loss parallel | `train.py:249` → `:347` | `dist_utils.get_train_context` | context manager | **runtime** |
| Grad normalization | `train.py:439` + `model_parallel.py` | `disable_fsdp_gradient_division` + manual `div_` | manual | **runtime** |
| MoE load balancing | `components/optimizer.py:190` | optimizer step pre-hook | hook + all-reduce on `loss` | **runtime** |

Note the split: **build-time** transforms mutate the module tree; **runtime** ones wrap the
forward/backward; **data-level** DP touches neither. `parallelize_deepseekv3` covers only the
build column. Nothing in the codebase names these three categories.

---

## 4. Why it's hard to follow — seven root causes

**R1. Three application phases with no vocabulary.**
Build-time / runtime / data-level (§3). A reader looking for "where is CP applied" finds nothing
in `parallelize.py` and has to discover it inside `forward_backward_step`. A reader looking for
"where is DP applied" finds nothing in either, because it's a dataloader argument.

**R2. Mandatory ordering encoded as bare sequential `if`s.**
`apply_non_moe_tp → apply_moe_ep_tp → apply_ac → apply_compile → apply_fsdp` is a *required*
order: TP before FSDP so the 1-D DTensor placements compose into 2-D; AC before compile so the
compiler sees the wrapper; FSDP last so `fully_shard` wraps the final tree. The code is five
`if` blocks in a row with no statement that the order is load-bearing or why.

**R3. PP inverts the call graph.**
`train.py` hands `parallelize_deepseekv3` to `pipeline_llm` as a callback. Reading top-down, the
"apply parallelism" step vanishes into `distributed/pipeline_parallel.py` and re-emerges as
`parallelize_fn(m, ...)` at line 105 of a different file. `train.py:177-205` then duplicates the
materialize/init/train block across both branches.

**R4. `parallelize.py` is the orchestrator but lives in `model/`.**
It imports from and drives every module in `distributed/`, yet sits in the model package under a
model-specific name. Worse, TP is split across two files by an axis that isn't visible from the
filenames: dense TP inline in `model/parallelize.py`, MoE TP in
`distributed/expert_parallel.py`. Someone asking "how does TP work here" must find both.

**R5. Mesh names are deliberately decoupled from parallelism names — with no legend.**
`fsdp` isn't the FSDP degree, it's `dp_shard × cp`. `loss` isn't a parallelism at all. `efsdp`
is a five-term expression. This naming is *good design* (each mesh is named for its consumer,
not its provenance), but §2's table exists nowhere in the repo, so every reader re-derives it.

**R6. Cross-cutting gradient-scaling logic is split three ways.**
These three only work as a set, and live in three files:
1. `disable_fsdp_gradient_division()` — FSDP stops dividing by world size (`model_parallel.py`)
2. loss is divided by `global_valid_tokens` before `backward()` (`train.py::forward_backward_step`)
3. under PP the schedule already ran `backward()`, so gradients are divided *after the fact*
   (`train.py::train_step:439`)

The comments explain each locally; nothing explains the invariant they jointly maintain.

**R7. `ParallelDims` has three jobs.**
Config validator (`_validate`), mesh factory (`build_mesh`), and lazy-init cache
(`get_mesh` triggers `build_mesh`). Plus the `_mesh_exist` special cases from §2. It's 269
lines doing the work of three concepts.

---

## 5. Bugs and gaps found while reading

Independent of any restructuring. **B1–B3 mean the PP path cannot currently run.**

| # | Location | Issue |
|---|---|---|
| B1 | `distributed/pipeline_parallel.py:306,307,313,318` | `pp_degree` is **never defined** in `pipeline_module_split`. Only `pp_rank` is. `NameError` the moment PP is enabled. Fix: `pp_degree = pp_mesh.size()`. |
| B2 | `train.py:456` → `distributed/utils.py` | `dist_utils.clip_grad_norm_` **does not exist**. Called every step. |
| B3 | `train.py:546` → `distributed/utils.py` | `dist_utils.set_pg_timeouts` **does not exist**. Called after step 1. |
| B4 | `distributed/activation_checkpoint.py:150` | `cast(nn.ModuleDict, model.layers)` — `layers` is an `nn.ModuleList`. Harmless (`named_children()` exists on both) but the annotation is wrong and misleads. Same pattern at `parallelize.py:170`. |
| B5 | `distributed/activation_checkpoint.py` | `_apply_full_ac(module, ac_config)` ignores `ac_config`; `_apply_ac_to_transformer_block(..., model_compile_enabled=...)` ignores that argument entirely. |

**Git history says these were never written.** `git log --all -S "def clip_grad_norm_"`,
`-S "def set_pg_timeouts"` and `-S "pp_degree = "` all return zero commits. Combined with this
being a CPU-only box, the conclusion is that **the PP path has never executed**, and no path has
ever run `train_step` past the clip call. B1–B3 are unwritten scaffolding, not regressions.

## 6. Decisions taken

| # | Question | Answer | Consequence |
|---|---|---|---|
| Q1 | Are B1–B3 regressions or unwritten? | Believed working — but git proves never written | Fix them *first*; the restructure then moves code that runs |
| Q2 | Does upstream diffability matter? | **No — this is my codebase now** | Free to move and rename files. Option A in full. |
| Q3 | How do we verify without GPUs? | Fake process group | Confirmed working on torch 2.13.0 (below) |

**Verification mechanism (confirmed on this box):**

```python
from torch.testing._internal.distributed.fake_pg import FakeStore
dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=8)
m = init_device_mesh("cpu", (8,), mesh_dim_names=("world",))
sub = m._unflatten(0, (2, 2, 2), ("pp", "batch", "tp"))   # works
sub["batch", "tp"]._flatten("loss")                        # works
```

Mesh shapes for any `world_size` can be validated on CPU in seconds. This is the safety net the
restructure runs against.

**Chosen approach: Option A** (orchestrator + mesh legend, no behavior change), with Option B's
phase vocabulary expressed as comments and log lines rather than a plan class.

---

## 7. Proposed design

Status: **presented, not yet approved.**

### 7.1 The organizing idea — name the four phases

The codebase applies parallelism in four distinct phases and has no word for any of them.
Everything else follows from naming them:

| Phase | What it does | Home |
|---|---|---|
| **0 · Topology** | degrees → meshes | `distributed/parallel_dims.py` |
| **1 · Data** | give each rank different tokens | dataloader (`dp_rank` / `dp_degree`) |
| **2 · Build** | transform the module tree — PP split, TP, EP/ETP, AC, compile, FSDP | `distributed/parallelize.py` |
| **3 · Runtime** | wrap fwd/bwd — CP ctx, loss-parallel, grad-scale reconciliation | `distributed/utils.py` + `train_step` |

Today only Phase 2 has a home, which is exactly why "where is CP applied?" and "where is DP
applied?" have no findable answer. The README and the in-code section headers both key off
this table.

### 7.2 File layout

```
distributed/
  parallel_dims.py         Phase 0 — topology only
  parallelize.py           ← MOVED from model/. The orchestrator and entry point.
  tensor_parallel.py       ← apply_non_moe_tp extracted out of parallelize.py
  expert_parallel.py       (unchanged) MoE TP / EP / ETP
  model_parallel.py        (unchanged) FSDP / HSDP / EFSDP / DDP
  activation_checkpoint.py (unchanged)
  pipeline_parallel.py     stage planning + schedule (loses the callback — see 7.3)
  utils.py                 Phase 3 contexts + collectives
  README.md                ← NEW. Mesh legend, phase table, ordering rules.
```

Moving `parallelize.py` out of `model/` appears to break a "`distributed/` is model-agnostic"
layering. **That layering does not exist today:** `expert_parallel.py` hardcodes
`moe.router.gate` and `moe.shared_experts.gate_proj`; `model_parallel.py` hardcodes
`token_embeddings` / `layers` / `layernorm` / `output`; `activation_checkpoint.py` hardcodes
`layers`. `distributed/` is already thoroughly DeepSeekV3-aware. Making that consistent is
clearer than half-enforcing a rule the code does not follow. The function keeps the name
`parallelize_deepseekv3` for the same reason — it is honest about hardcoding FQNs.

### 7.3 Collapse the PP inversion (the main code change)

Today `train.py` passes `parallelize_deepseekv3` *into* `pipeline_llm` as a callback (R3), then
duplicates the materialize/init/train block across both branches. Replace both with one function:

```python
# distributed/parallelize.py
def build_model_parts(model, parallel_dims, job_config, device, loss_fn) -> ModelParts:
    """Split, shard, and materialize. One road whether or not PP is on."""
    # Phase 2a — split across pipeline stages (PP)
    stages, parts = (pipeline_module_split(...) if parallel_dims.pp_enabled
                     else (None, [model]))

    # Phase 2b — shard within each stage (TP → EP → AC → compile → FSDP)
    for i, part in enumerate(parts):
        parts[i] = apply_sharding(part, parallel_dims, job_config)
        if stages:
            stages[i].submod = parts[i]

    # Phase 2c — materialize off the meta device
    for part in parts:
        part.to_empty(device=device_type)
        with torch.no_grad():
            part.init_weights()
        part.train()

    schedule = build_pipeline_schedule(job_config, stages, loss_fn) if stages else None
    return ModelParts(parts, schedule, has_first_stage, has_last_stage)
```

`train.py` loses its `if pp_enabled / else` block entirely — ~30 lines become one call.
`pipeline_llm` shrinks to stage planning and stops taking `parallelize_fn`, so the call graph
reads top-down with no bounce through another file. `apply_sharding` is today's
`parallelize_deepseekv3` body plus the mandatory-order section headers (R2).

### 7.4 `distributed/README.md`

Three things that exist nowhere in the repo and that every reader currently re-derives:

1. **The mesh legend** — §2 of this document.
2. **The phase table** — §7.1.
3. **The gradient-scaling invariant** — R6: `disable_fsdp_gradient_division` + dividing loss by
   `global_valid_tokens` + the post-hoc `p.grad.div_()` under PP are *one mechanism split across
   three files*, correct only as a set.

### 7.5 Prerequisite — make it runnable first

Refactoring code that has never executed is guesswork, so this comes first:

- **B1** — `pp_degree = pp_mesh.size()` in `pipeline_module_split`. One line.
- **B2** — write `dist_utils.clip_grad_norm_`. **The one genuinely uncertain item.** Needs
  DTensor grads reduced to full tensors, the norm combined across `pp_mesh` (model parts hold
  different params), and EP params handled separately since they live on the `sparse` mesh.
  ~60–80 lines of real logic, not scaffolding. May want its own design round.
- **B3** — write `dist_utils.set_pg_timeouts`. Small.
- **B4 / B5** — wrong `ModuleDict` casts, unused arguments.
- **Harness** — `scripts/check_meshes.py`: for every config in `default_configs.py`, spin a fake
  PG at that config's implied world_size, build the meshes, assert every size, and run the PP
  stage-split planning. Catches B1 and pins §7.2/§7.3 against regression. CPU, seconds.

### 7.6 Order of work

1. Verification harness (so steps 2–3 have a check)
2. B1–B5 fixes
3. File moves — `parallelize.py` → `distributed/`, extract `tensor_parallel.py`
4. `build_model_parts` collapse + phase section headers
5. `distributed/README.md`

Steps 3–5 are pure moves, renames and comments — no semantic change, harness green throughout.

**Risk note:** §7.3 is the only step with behavioral risk, since it changes when materialization
happens relative to schedule construction. Believed equivalent; the harness cannot prove it
without GPUs.

---

## 8. What transfers to inference

Asked mid-review: how much of this serves inference, and how does it differ from vLLM /
TensorRT-LLM? Recorded here because it shapes which seams are worth preserving.

### 8.1 Component-by-component

| Component | Transfers? | Why |
|---|---|---|
| **TP** (`apply_non_moe_tp`) | **Yes, nearly as-is** | Colwise `wq`/`wkv_b`, rowwise `wo` is exactly what vLLM does for MLA. Only change: drop `loss_parallel`, so `output` becomes `Replicate()` rather than `Shard(-1)`. |
| **EP** (`ExpertParallel`) | **Yes** | DeepSeek's own inference EP is this same all-to-all dispatch → grouped GEMM → all-to-all combine. |
| **CP** | Prefill only | Ring attention over a long prompt is real. At decode `S=1` there is no sequence to split. |
| **PP** | Split yes, schedule no | `pipeline_module_split` is reusable; `1F1B`/`GPipe`/`ZBV` are forward+backward-over-microbatch schedules. Inference needs forward-only pipelining over *requests*. |
| **DP** | Trivially | Replicas behind a load balancer. |
| **FSDP** | **No** | `fully_shard` reshards after forward and re-all-gathers per layer. In training that buys optimizer-state memory; in inference it is per-token overhead. Use TP/EP for memory instead. |
| AC, compile, optimizer, LR sched, grad scaling, checkpointing | No | Training-only by construction. |

Roughly: distributed layer ~40% reusable, training loop 0%.

### 8.2 Inference work already present in the model

`generate()`, the `kv_cache` / `pe_cache` buffers, and `absorb_weights()` (fusing W^UK into `wq`
and W^UV into `wo`) are genuine MLA inference optimizations, already written. But `generate()`
runs single-process and touches none of the parallelism machinery. Two notes if they are ever
wired together:

- **The MLA latent cache is correct under TP by architecture.** `kv_cache` holds `kv_lora_rank`
  (head-shared), so every rank holding a full copy is right and cheap. That is the point of MLA.
- **The `attn_impl='naive'` path is not.** `k_cache` is
  `(max_batch, max_seq, n_heads, qk_head_dim)` — full `n_heads`, unsharded — but under TP
  `k_new` is head-sharded, so the assignment shape-mismatches. Inference-only; does not affect
  the training path.

### 8.3 Why vLLM / TensorRT-LLM look nothing like this

The root difference is the shape of the workload.

**Training is a known computation.** Fixed batch, fixed seq len, every step identical to the
last. The whole execution is plannable at startup — which is exactly why
`parallelize_deepseekv3` runs once in `__init__` and never again. Bottleneck is FLOPs and
gradient communication; memory pressure is params + grads + optimizer states + activations;
latency is not a concept.

**Inference is an unknown workload.** Requests arrive at random times with random prompt lengths
and stop after a random number of tokens. Nothing is plannable at startup. And it is two
machines in one:

- **Prefill** — large `S`, compute-bound, resembles a training forward pass.
- **Decode** — `S=1`, arithmetic intensity ≈ 1, purely memory-bandwidth-bound. One decode step
  reads the *entire model* out of HBM to produce one token.

Memory pressure is the KV cache, which grows per request and frees unpredictably. Latency is a
first-class SLO (TTFT, TPOT).

**What engines have that this code has no analogue for:**

1. **PagedAttention / a KV block allocator** — the big one. `Attention` does
   `torch.zeros(max_batch_size, max_seq_len, kv_lora_rank)` upfront: contiguous, static, and a
   100-token request still reserves `max_seq_len`. vLLM allocates KV in fixed-size blocks from a
   global pool with a per-sequence block table, so memory tracks actual usage and prefix sharing
   falls out for free.
2. **Continuous batching** — requests join and leave the batch *between decode steps*. Training
   has no equivalent; a step's batch is frozen before it starts.
3. **Prefix caching** — built here at the API layer (commit `026f4a9`); in an engine it is a
   property of the block allocator, shared prefixes pointing at the same physical blocks.
4. **Chunked prefill / prefill-decode disaggregation** — one long prefill stalls every decode in
   the batch, so engines chunk it or move prefill to separate hardware.
5. **Serving quantization** — FP8/INT4 weight-only, AWQ/GPTQ. Gradients need numerics inference
   does not.
6. **Decode-shaped kernels** — `S=1` GEMMs are skinny matrix-vector products. `torch._grouped_mm`
   in `GroupedExperts` is a training-shaped kernel.

**And the frameworks are built differently:**

- **This codebase (torchtitan-style)** — a *library of composable transforms* over an eager
  `nn.Module`. Keep the model, apply `parallelize_module` / `fully_shard`, let DTensor emit the
  collectives. Maximally flexible and readable; the architecture stays editable.
- **vLLM** — a *runtime*. It owns the scheduler, the allocator and the execution loop. Models are
  re-implemented against its interfaces (`vllm/model_executor/models/deepseek_v3.py`, built from
  its own `ColumnParallelLinear` / `MergedColumnParallelLinear`). You port to it; you do not hand
  it your module.
- **TensorRT-LLM** — historically an *ahead-of-time compiler*: build a serialized engine per
  (model, GPU, TP degree, max batch, max seq len), fused kernels and algorithm choices baked in.
  Fastest, least flexible — rebuilds take minutes and artifacts do not port across GPU types. It
  has since added a more flexible PyTorch backend, so the AOT framing is no longer absolute.

**The frame worth keeping:** *training frameworks parallelize a known computation; inference
engines schedule an unknown one.* One is a set of transforms applied once at startup; the other
is a control loop that runs forever.

**Where they converge:** the sharding math is identical. vLLM shards MLA exactly the way
`apply_non_moe_tp` does, and DeepSeek's inference EP does the same all-to-all as
`ExpertParallel`. What is learned here transfers directly — it is the layer above (memory
management and scheduling) that is a different discipline.

### 8.4 Effect on the design

None, and mildly confirming: §7.2 and §7.3 keep the sharding plans separable from the training
orchestration, which is exactly the seam to cut along if they are ever reused for serving.

---

## 9. Open items

- §7 design not yet approved — section-by-section review pending.
- B2 (`clip_grad_norm_` under PP + EP) may warrant its own design round before implementation.
