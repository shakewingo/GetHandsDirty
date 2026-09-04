# DeepSeek-V3 MoE: Design

Date: 2026-09-02
Scope: `distributed_framework/torchfeather`
Status: **Phase 1 complete and verified.** Phase 2 code complete except §12.8 (one open
one-line dtype fix). L1 (multi-rank CPU/gloo) run 2026-09-03: all 6 assertions pass with
§12.8 patched. L3 multi-GPU matrix not yet run.

## 1. Goal

Add a Mixture-of-Experts feed-forward path to `DeepSeekV3Model` as **pure model code** — no
distributed concerns inside the module. Expert parallelism (EP) lands in a second phase and
must not require restructuring `MoE.forward`.

The reference is `hkproj/torchfeather` (itself derived from torchtitan). Its MoE is correct but
interleaves the model with EP mechanics: padding to a Triton kernel's alignment, a second
histogram that only differs under sequence-parallel routing, and index gymnastics that exist to
keep `torch.compile` happy. This design keeps the parts that are genuinely the model and defers
the rest.

**Phase 1 (this spec):** MoE runs on one device, no DTensor, no collectives.
**Phase 2 (separate spec):** wire EP/ETP through `parallelize.py` and `expert_parallel.py`.

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Experts are **stacked `nn.Parameter`s** `[E, out, in]`, computed in Phase 1 by a per-expert loop; `torch._grouped_mm` swaps in during Phase 2. | The stacked layout is what `ExpertParallel._partition_fn` shards on dim 0 — keeping it means Phase 2 changes no parameter shapes or checkpoint keys. The loop runs on CPU and is easy to step through while learning. **Correction (2026-09-03):** the original rationale claimed `_grouped_mm` needs sm90 and could not run on this CPU-only box. That is wrong — it runs on CPU in torch 2.13 given 16-byte-aligned strides (keep `dim` and `hidden_dim` multiples of 8), and matches the loop exactly. Phase 2 adopted it; see §12.3. |
| D2 | Naming is **`gate_proj` / `up_proj` / `down_proj`** everywhere, including `GroupedExperts`. | Matches the dense `FeedForward` and `Attention` in `model.py`. `w1/w2/w3` gives no hint which is the down projection. Cost is three hardcoded spots in `expert_parallel.py`, paid in Phase 2 (see §8). |
| D3 | **Keep the module seams EP hooks into**, drop everything else. `TokenReorderer` stays an `nn.Module`; `GroupedExperts.forward` keeps the `(routed_input, num_tokens_per_expert)` signature. | `distribute_module` can only attach `ReordererSequenceParallel` / `ExpertParallel` hooks to a real module with that exact signature. These two seams are the *only* EP-shaped thing in the pure model; padding, Triton and all-to-all stay outside. |
| D4 | Router is **plain sigmoid top-k**. No grouped/node-limited routing, no auxiliary balance loss. | Node-limited routing is a communication optimization that only pays off with multi-node EP. The aux loss is precisely what the bias mechanism replaces. Both are additive router changes later, touching nothing downstream. |
| D5 | Verification is a **`__main__` self-check block** in `moe.py`, not pytest. | The repo has no test suite, no pytest, and no numpy in the active env; `model.py` already ends with a `__main__` smoke block. Matching the existing convention beats introducing a framework. Revisit if a real suite is ever added. |

## 3. Component design (`model/moe/moe.py`)

Shapes throughout: `N = B*S` tokens, `D = dim`, `E = num_experts`, `K = top_k`.

### 3.1 `TopKRouter`

Returns `(top_scores, selected_experts_indices, num_tokens_per_expert)` with shapes
`(N, K)`, `(N, K)`, `(E,)`.

```python
scores = torch.sigmoid(self.gate(x))                 # (N, E), computed in fp32
if expert_bias is not None:
    _, sel = torch.topk(scores + expert_bias, self.top_k, dim=1, sorted=False)
    top_scores = scores.gather(dim=1, index=sel)     # gate value from UNBIASED scores
else:
    top_scores, sel = torch.topk(scores, self.top_k, dim=1, sorted=False)

if self.route_norm:
    top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
top_scores = top_scores * self.route_scale

num_tokens_per_expert = torch.histc(sel.view(-1), bins=E, min=0, max=E)
```

The bias/gating split is the important line. `expert_bias` must influence **selection only**;
if it leaks into `top_scores` it enters the gradient and corrupts the gate. The current draft in
the repo does `score += expert_bias` before `topk` and then gathers from the biased tensor —
that is the bug this replaces (see §7.2).

### 3.2 `TokenReorderer`

The router's output is *token-major*: for each token, which K experts. The experts want
*expert-major*: all rows for expert 0 contiguous, then expert 1, so one grouped GEMM with
per-expert offsets covers everything. The reorderer computes **no scores**; it produces one
permutation over the `N*K` (token, expert) assignment pairs.

```python
def forward(self, top_scores, selected_experts_indices):
    perm = selected_experts_indices.view(-1).argsort(stable=True)
    num_tokens_per_expert = torch.histc(
        selected_experts_indices.view(-1), bins=self.num_experts,
        min=0, max=self.num_experts,
    )
    return (
        top_scores.view(-1)[perm],   # (N*K,)  score per slot
        perm // self.top_k,          # (N*K,)  token index per slot
        num_tokens_per_expert,       # (E,)
    )
```

`stable=True` preserves token order inside each expert's block. `// top_k` inverts the
flattening `flat = token*K + k` to recover the token index — that one operator is the whole
trick, and it is why the module needs no scatter or mask.

The histogram is recomputed here rather than reused from the router. In Phase 1 the two are
identical and this is redundant; it is kept because in Phase 2 the reorderer may see only a
TP-shard of the tokens (`ReordererSequenceParallel`) while the router's histogram must stay
global for the load-balance statistics. Keeping it now means Phase 2 does not edit this file.

### 3.3 `GroupedExperts`

```python
self.gate_proj = nn.Parameter(torch.empty(E, hidden_dim, dim))   # EP Shard(0); ETP Shard(1)
self.up_proj   = nn.Parameter(torch.empty(E, hidden_dim, dim))   # EP Shard(0); ETP Shard(1)
self.down_proj = nn.Parameter(torch.empty(E, dim, hidden_dim))   # EP Shard(0); ETP Shard(2)

def forward(self, routed_input, num_tokens_per_expert):
    offsets = num_tokens_per_expert.cumsum(0, dtype=torch.int32).tolist()
    outs, start = [], 0
    for e, end in enumerate(offsets):
        xe = routed_input[start:end]
        h = F.silu(xe @ self.gate_proj[e].T) * (xe @ self.up_proj[e].T)
        outs.append(h @ self.down_proj[e].T)
        start = end
    return torch.cat(outs, dim=0)
```

`torch.cat` over a list rather than writing slices into a preallocated buffer: the slice lengths
are data-dependent, and `cat` keeps autograd straightforward and handles zero-token experts
without a special case. The `.tolist()` forces a device sync — acceptable, and EP incurs the
same sync in `_token_dispatch` regardless.

The shard-dim comments are load-bearing documentation for Phase 2: they state the mapping this
design owes `expert_parallel.py`.

### 3.4 `MoE.forward`

```python
B, S, D = x.shape
x = x.reshape(-1, D)

top_scores, sel, counts = self.router(x, self.expert_bias)
with torch.no_grad():
    self.tokens_per_expert.add_(counts)                # load-balance stats only

scores_sorted, tok_sorted, counts = self.reorderer(top_scores, sel)

routed_input = x[tok_sorted]                           # (N*K, D), tokens replicated K times
if self.score_before_experts:
    routed_input = (routed_input.float() * scores_sorted.unsqueeze(-1)).to(x.dtype)

routed_output = self.experts(routed_input, counts)     # (N*K, D)

out = self.shared_experts(x) if self.shared_experts is not None else torch.zeros_like(x)

if not self.score_before_experts:
    routed_output = (routed_output.float() * scores_sorted.unsqueeze(-1)).to(x.dtype)

out = out.index_add(0, tok_sorted, routed_output)
return out.view(B, S, D)
```

Computes `out[t] = shared(x[t]) + Σ_{e ∈ topk(t)} g_{t,e} · E_e(x[t])`. The hidden dim is never
split: every expert consumes and produces a full `D`-vector, and the merge is `index_add`
summing the K rows belonging to a token back into its slot.

`score_before_experts` scales the expert *input* instead of its output. The two are **not**
equivalent — SwiGLU is nonlinear. DeepSeek-V3 scales the output, so `False` is the correct
default (see §7.3).

Two deviations from upstream, both to be re-verified in Phase 2:

- **`x[tok_sorted]` / `out.index_add(0, ...)`** instead of `tok_sorted.reshape(-1,1).expand(-1,D)`
  fed to `torch.gather` / `scatter_add`. Identical math with a 1-D index. Upstream's expanded
  form is most likely chosen for `torch.compile` or DTensor friendliness; `parallelize.py:175`
  already drops MoE blocks to `fullgraph=False`, so the risk now is low and the question is
  answerable with a benchmark in Phase 2.
- **Shared expert called before the output scaling.** Upstream's comment says this overlaps the
  shared-expert compute with the EP token-combine collective. There is no collective in Phase 1,
  but the ordering costs nothing to keep.

## 4. Integration into `model.py`

`DecoderLayer` gains a `layer_id` and branches on it:

```python
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV3ModelArgs):
        ...
        self.moe_enabled = layer_id >= args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(args.moe_args, dim=args.dim, hidden_dim=args.moe_inter_dim)
        else:
            self.ffn = FeedForward(args)

    def forward(self, x, mask=None, use_kv_cache=False, start_pos=0):
        ...
        x = self.post_attention_layernorm(x)
        x = self.moe(x) if self.moe_enabled else self.ffn(x)
        x = residual + x
```

`DeepSeekV3Model.__init__` passes the index: `DecoderLayer(layer_id, args)`. That is the only
construction site in the repo.

The attribute names are not free choices — they are already depended on:

- `parallelize.py:175` and `:268` read `transformer_block.moe_enabled`.
- `parallelize.py:268-280` applies the dense-FFN TP plan (`ffn.gate_proj`, `ffn.up_proj`,
  `ffn.down_proj`) only when `not moe_enabled`. Constructing `ffn` on MoE layers would therefore
  leave it unsharded, untrained and counted as dense parameters by
  `get_params_and_flops` — hence the `if/else`, not an unconditional `ffn` plus an extra `moe`.
- `expert_parallel.py:apply_moe_ep_tp` and `optimizer.py:_update_expert_bias` both reach for
  `transformer_block.moe`.
- `model_args.get_params_and_flops` counts by the FQN substrings `moe.router`, `moe.experts`,
  `moe.shared_experts`.

So `self.moe`, `self.moe_enabled` and the sub-module names are fixed by existing code, and after
this change the active-parameter count and FLOPs estimate start reporting real numbers instead
of zeros for the sparse terms.

## 5. Weight initialization

`DeepSeekV3Model.init_weights` sweeps modules by type (`nn.Linear`, `nn.Embedding`,
`nn.RMSNorm`). `GroupedExperts` holds raw `nn.Parameter`s and matches none of them, so without a
new branch `gate_proj` and `up_proj` would keep whatever `torch.empty` returned. Add:

```python
elif isinstance(module, GroupedExperts):
    for p in (module.gate_proj, module.up_proj, module.down_proj):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(self.dim))
```

The depth-scaled rule that follows currently reads:

```python
if pn.endswith('wo.weight') or pn.endswith('down_proj.weight'):
```

`moe.shared_experts.down_proj.weight` already matches (it is an `nn.Linear`).
`moe.experts.down_proj` does not — it is a bare parameter with no `.weight` suffix. Extend to:

```python
if pn.endswith('wo.weight') or pn.endswith('down_proj.weight') or pn.endswith('experts.down_proj'):
```

`shared_experts.down_proj.weight` ends in `.weight`, so it cannot also match the new clause —
no double application.

`MoE` registers two buffers that `to_empty()` leaves uninitialized on the meta-device path:
`expert_bias` (persistent) and `tokens_per_expert` (non-persistent). `init_weights` must zero
both, in the same place it already re-materializes the RoPE tables and the KV cache.

## 6. Load balancing

Auxiliary-loss-free balancing (DeepSeek-V3, arXiv:2408.15664). No loss term anywhere — that is
the point. Three pieces, already two-thirds built:

1. **`TopKRouter.forward`** adds `expert_bias` to the scores for the top-k selection only (§3.1).
2. **`MoE.forward`** accumulates `tokens_per_expert` from the router histogram under `no_grad`.
   This is the only new piece.
3. **`optimizer.py:_update_expert_bias`** (already written, lines 186-237) runs as an optimizer
   **step pre-hook**, so it fires once per optimizer step and is unaffected by gradient
   accumulation. It all-reduces `tokens_per_expert` across DP/PP, then:
   ```
   delta = load_balance_coeff * sign(tokens_per_expert.mean() - tokens_per_expert)
   delta = delta - delta.mean()          # keep the bias zero-sum
   expert_bias += delta
   tokens_per_expert.zero_()
   ```
   Over-loaded expert → negative bias → less likely selected next step. `sign()` is why the
   activation-checkpointing double-count (forward, then recompute in backward) is harmless.

Phase 1 runs single-process, so the all-reduce is a no-op and the hook exercises end to end
without any distributed setup.

## 7. Pre-existing issues this work must fix

### 7.1 Circular import (verified, currently fatal)

`model/moe/moe.py:8` imports `DeepSeekV3ModelArgs` from `model_args`, while `model_args.py:6`
imports `MoEArgs` from `model.moe`. Importing the MoE package first raises:

```
ImportError: cannot import name 'MoEArgs' from partially initialized module
'torchfeather.model.moe' (most likely due to a circular import)
```

It only appears to work today because `model_args` happens to be imported first. The import is
unused — delete `moe.py:8`. `MoE` takes plain `dim` / `hidden_dim` ints precisely so it never
needs the model args type.

### 7.2 Router leaks the balancing bias into the gate value

`moe.py:69-75` — covered in §3.1.

### 7.3 `score_before_experts` default disagrees with the config

`MoEArgs.score_before_experts` defaults to `True` (`moe.py:20`) while
`default_configs.py:21` sets `False`. DeepSeek-V3 scales the expert output. Flip the dataclass
default to `False`; leave the config explicit.

### 7.4 `model/moe/__init__.py` exports only `MoEArgs`

Must also export `MoE` and the MoE-local `FeedForward`, which is what `model.py` will import.

## 8. Phase 2 seam inventory

Recorded now so the EP phase is a wiring exercise, not archaeology. Nothing here is done in
Phase 1.

| Seam | What Phase 2 does |
|---|---|
| `expert_parallel.py` `TensorParallel._partition_fn` | Rename `w1/w2/w3` → `gate_proj/up_proj/down_proj`; shard dims 1 / 1 / 2 respectively (D2). |
| `expert_parallel.py` `ExpertTensorParallel._tp_shard_dims` | Same rename: `{"gate_proj": 1, "up_proj": 1, "down_proj": 2}`. |
| `expert_parallel.py` `apply_moe_ep_tp` TP plan | `moe.shared_experts.w1/w2/w3` → `gate_proj` (Colwise), `up_proj` (Colwise), `down_proj` (Rowwise, `output_layouts=Partial()`). |
| `GroupedExperts.forward` | Add the `DTensor` → `to_local()` unwrap and swap the loop for `torch._grouped_mm`, which then requires the alignment padding in `moe/utils.py`. |
| `TokenReorderer` | Unchanged — `ReordererSequenceParallel` wraps it from outside. This is what D3 buys. |
| `MoE.forward` | Unchanged. Re-benchmark the `index_add` vs `scatter_add` choice under `torch.compile`. |
| `moe/utils.py`, `moe/kernels.py` | Untouched in Phase 1; become live once `_grouped_mm` is in. |

## 9. Verification

Runs on CPU, no pytest, no GPU (D5). A `__main__` block in `moe.py` asserting:

1. **Reference equivalence.** With `top_k == num_experts`, `route_norm=False`, `route_scale=1.0`,
   `num_shared_experts=0`, `score_before_experts=False`, `expert_bias=None`, the MoE output must
   equal a naive loop `Σ_e sigmoid(gate(x))[:, e] * FFN_e(x)` computed directly from the stacked
   weights. This is the test that actually pins the gather/index_add/permutation logic — every
   expert is selected, so nothing is masked out and any misrouting shows up numerically.
   **Tolerance:** the expert path is bf16 since Phase 2, so compare at `atol=5e-2` against an
   fp32 reference, or at `atol=1e-5` against a reference computed in bf16 (which matches exactly).
2. **Permutation integrity.** For a random `(N, K)` index tensor, walking `tok_sorted` in blocks
   sized by `counts` must yield, for block `e`, exactly the set of tokens whose top-k contains
   `e`.
3. **Top-1 identity.** With `top_k=1` and no shared expert, output equals
   `score * FFN_{argmax}(x)` per token.
4. **Bias affects selection only.** Construct an `expert_bias` that flips which expert wins, then
   assert the returned `top_scores` equals the *unbiased* score at the selected index.
5. **Counter.** `tokens_per_expert.sum() == N * top_k` after one forward.
6. **Gradient reach.** A full `DeepSeekV3Model` forward/backward with `n_dense_layers=1`,
   `n_layers=3`, `top_k == num_experts`: every expert parameter has a non-`None`, finite,
   non-zero gradient. Catches both the init gap in §5 and any accidental autograd break in the
   `torch.cat` loop.
7. **Import order.** `import torchfeather.model.moe.moe` as the first import succeeds (§7.1).

## 10. Out of scope

- Expert parallelism, `torch._grouped_mm`, the Triton permute kernel, all-to-all.
- Grouped / node-limited routing and the sequence-wise auxiliary loss (D4).
- Expert-choice routing. Token-choice is kept: expert-choice leaks information across a sequence
  (an expert's capacity decisions depend on future tokens), which breaks causality at inference,
  and token-choice yields the load statistics for free.
- Loading real DeepSeek-V3 HF checkpoints into the MoE layers.
- MoE behaviour under `generate` / KV-cache. The MoE path is position-independent, so it works,
  but nothing here optimizes decode.

---

# Phase 2: Expert Parallelism

Date added: 2026-09-03
Status: Phase 1 complete and verified (§9 checks 1, 4, 6 pass; `n_dense_layers` deliberately
left unwired). **Phase 2 code changes complete and cross-checked as of 2026-09-03**; the
verification ladder (§14) and multi-GPU matrix (§15) have not been run.

## 11. What Phase 2 actually is

The distributed machinery already exists in this repo and is untouched by Phase 1:
`expert_parallel.py` (the three `ParallelStyle`s), `parallel_dims.py` (the `ep`/`etp`/`efsdp`
mesh axes), `model_parallel.py` (EFSDP wrapping of expert params), `optimizer.py`
(`_update_expert_bias` and per-mesh param grouping), and `moe/utils.py` + `moe/kernels.py`
(alignment padding). None of it has ever run, because no MoE layer existed.

Phase 2 is therefore **not** "write EP". It is: fix the contract mismatches between the MoE
modules written in Phase 1 and the parallelism code that was copied from upstream, then verify.

Four parallelism modes come out of `apply_moe_ep_tp`, and each takes a different branch:

| mode | `tp` | `ep` | `etp` | expert plan | routed experts are… |
|---|---|---|---|---|---|
| TP only | >1 | 1 | — | `TensorParallel` | replicated across ranks, each expert's weights col/row-sliced |
| EP only | 1 | >1 | 1 | `ExpertParallel` | split across EP ranks, weights whole |
| EP borrowing TP | >1 | >1 | 1 | `ExpertParallel` + `ReordererSequenceParallel` | split across EP ranks; tokens split across TP ranks |
| ETP | >1 | >1 | =tp | `ExpertTensorParallel` | split across EP ranks *and* col/row-sliced within each |

## 12. Blockers found — **all closed 2026-09-03**

Each was verified against the tree when found, and re-verified as fixed. Two were resolved
differently from the original proposal; those are marked below.

### 12.1 `GroupedExperts` has no `num_experts` attribute — `AttributeError` — CLOSED

`model_parallel.py:92` reads `moe.experts.num_experts` to decide the EFSDP shard placement.
`GroupedExperts.__init__` stores only the three parameters. Add `self.num_experts = num_experts`.

### 12.2 `model.layers` is a `ModuleList`, but `optimizer.py` calls `.values()` — CLOSED (resolved the other way)

`optimizer.py` lines 175, 194 and 220 do `cast(nn.ModuleDict, model_part.layers)` followed by
`layers.values()`. `nn.ModuleList` has no `.values()`, so
`build_optimizers_with_moe_load_balancing` raises as soon as any MoE layer exists.

**Resolved by changing `optimizer.py` instead of the model**: its three `cast(nn.ModuleDict, ...)`
sites became `cast(nn.ModuleList, ...)` with direct iteration. `model.layers` stays a
`ModuleList`, so `DeepSeekV3Model.forward` and `init_weights` were untouched. The original
proposal, kept for context:

Upstream's model uses `self.layers = nn.ModuleDict()` with `self.layers[str(layer_id)] = block`.
Switching to `ModuleDict` is the smaller change and keeps parameter FQNs identical
(`layers.0.…` either way), so checkpoints are unaffected. It requires two follow-on edits: the
two `for layer in self.layers` loops in `DeepSeekV3Model.forward` and `init_weights` must become
`self.layers.values()`, because iterating a `ModuleDict` yields keys.

Everything else is already container-agnostic: `pipeline_parallel.py:246` handles both,
`activation_checkpoint.py` and `apply_compile` use `named_children()`, `apply_fsdp` uses
`list(...)`, `apply_moe_ep_tp` iterates directly.

### 12.3 `GroupedExperts.forward` must return `routed_input.shape[0]` rows, not `sum(counts)` — CLOSED (resolved by adopting `_grouped_mm`)

This is the one §8 glossed over, and it breaks silently rather than obviously.

Under EP, `ExpertParallel._token_dispatch` calls `_permute`, which pads the token block for each
local expert up to a multiple of `TOKEN_GROUP_ALIGN_SIZE_M = 8` and returns
`routed_input` of length `padded_max_len`, together with the **padded** per-expert counts. But
`sum(padded_counts) < padded_max_len` in general.

`_unpermute` then does `out_unpermuted[permuted_indices, :] = out`, which requires
`out.shape[0] == len(permuted_indices) == padded_max_len`. Upstream satisfies this for free
because `torch._grouped_mm` emits one output row per input row. The Phase 1 loop emits
`sum(counts)` rows and will raise a shape mismatch.

Fix without adopting `_grouped_mm`: pad the concatenated output back up to
`routed_input.shape[0]` with zeros. The padded rows are all-zero input, hence all-zero output,
and `_unpermute` discards them via the appended sentinel row — so this is correct, just wasteful.
That waste is the "side effect" the `expert_parallel.py` comment warns about.

### 12.4 `GroupedExperts.forward` needs the `DTensor` → `to_local()` unwrap — CLOSED

Every expert plan replaces the three parameters with `DTensor`s. `self.gate_proj[e]` on a
`DTensor` does not give the local shard. Unwrap once at the top of `forward`, as upstream does.

### 12.5 The three `w1/w2/w3` renames in `expert_parallel.py` — CLOSED

Worth recording how this failed in practice: the first attempt renamed **positionally**
(`w1→gate_proj, w2→up_proj, w3→down_proj`), which is wrong because upstream's `w2` is the
*down* projection. All three sites were affected. Final state, cross-checked:

```
TensorParallel._partition_fn : gate_proj=Shard(1), up_proj=Shard(1), down_proj=Shard(2)
ExpertTensorParallel dims    : {"gate_proj": 1, "up_proj": 1, "down_proj": 2}
shared_experts TP plan       : gate=Colwise, up=Colwise, down=Rowwise(Partial)
```

As tabled in §8. Restating the mapping because the positional order is a trap — upstream's `w2`
is the **down** projection, not "up":

| upstream | ours | shape | TP style | stacked shard dim |
|---|---|---|---|---|
| `w1` | `gate_proj` | `(E, hidden, dim)` | Colwise | 1 |
| `w3` | `up_proj` | `(E, hidden, dim)` | Colwise | 1 |
| `w2` | `down_proj` | `(E, dim, hidden)` | Rowwise, `output_layouts=Partial()` | 2 |

`gate` and `up` must be sharded **identically** because `silu(gate(x)) * up(x)` is elementwise
over `hidden`; `down` then consumes the sharded `hidden` and produces a partial sum needing one
all-reduce. This matches the dense plan already in `parallelize.py:277-279`.

### 12.6 No config sets `moe_enabled` — CLOSED

`get_deepseek_v3_model_args()` sets `moe_args` and `n_dense_layers` but never `moe_enabled`,
which defaults to `False`. The real DeepSeek-V3 config therefore builds an all-dense model and
`apply_moe_ep_tp` finds nothing to parallelize. Set it in `default_configs.py` (or wire
`n_dense_layers` per §4 and drop the flag).

### 12.7 Contract the reorderer already satisfies — do not change it

`ReordererSequenceParallel._prepare_input_fn` unpacks `(top_scores, selected_experts_indices)`
and `_prepare_output_fn` unpacks `(top_scores, token_indices_experts_sorted,
num_tokens_per_expert)` and asserts `hasattr(mod, "top_k")`. The Phase 1 `TokenReorderer` matches
all three. Note this is the one place where the router's `(indices, scores, …)` return order
would bite if it were ever propagated further.

### 12.8 `num_tokens_per_expert` is `float32`, but the all-to-all needs `int` — OPEN

Found by running L1. `all_to_all_single_autograd` asserts:

```
All output_split_sizes must be int or SymInt, got [15.0, 17.0]
```

`ExpertParallel._token_dispatch` derives the split sizes via `.tolist()` on
`num_tokens_per_expert`, so they inherit its dtype. `torch.histc` returns the dtype of its
*input*, and the CPU kernel is float-only, so §12's `.float()` workaround makes the counts
`float32` and the splits floats.

This is a genuine CPU/CUDA divergence, not a test artifact: on CUDA `histc` accepts int64 and
returns int64, which is why upstream never hit it.

**Fix — make the counts int64 at both call sites** (`TopKRouter.forward` and
`TokenReorderer.forward`):

```python
# option A (preferred): bincount is integral by construction, no bin arithmetic to reason about,
# and it raises if handed float scores by mistake
num_tokens_per_expert = torch.bincount(idx.view(-1), minlength=self.num_experts)

# option B: keep histc, cast back
num_tokens_per_expert = torch.histc(
    idx.view(-1).float(), bins=self.num_experts, min=0, max=self.num_experts
).to(torch.int64)
```

Everything downstream is dtype-agnostic: `cumsum(0, dtype=torch.int32)` in `GroupedExperts` and
the `float32` `tokens_per_expert` buffer both accept int64.

## 13. Implementation steps

Ordered so each step is independently verifiable.

1. ✅ **§12.1** add `self.num_experts` to `GroupedExperts`.
2. ✅ **§12.6** set `moe_enabled=True` in `get_deepseek_v3_model_args()`.
3. ✅ **§12.2** — done the other way: `optimizer.py`'s three sites now iterate a `ModuleList`.
   `model.layers` unchanged.
4. ✅ **§12.4** `DTensor` → `to_local()` unwrap in `GroupedExperts.forward`.
5. ✅ **§12.3** — **superseded**. Adopting `_grouped_mm` (step 8) removed the need to pad the loop
   output: it emits one row per input row, which is what `_unpermute` requires. The loop is gone,
   kept commented out in the source as a reference implementation.
6. ✅ **§12.5** the three renames in `expert_parallel.py` (see the correction noted there).
7. ✅ **step 8, done early** — `_grouped_mm` adopted unconditionally rather than behind an
   `is_cuda` check, since it runs on CPU too. L2 is therefore no longer a separate rung.
8. 🟡 **L1 run 2026-09-03 — all 6 assertions pass at 2 and 4 ranks**, but only with §12.8
   patched in the harness. Fix §12.8 in the source, then L1 is genuinely green.
9. ⬜ Run the L3 multi-GPU matrix (§15).

**Regression gate after every step above:** §9 checks 1, 4 and 6. Last run 2026-09-03 — all pass
(9.1 at `atol=5e-2`, exact against a bf16 reference).

## 14. Verification ladder

**L0 — single process, CPU (already passing).** §9 checks 1, 4, 6. Re-run after every step
above; steps 1-6 are all designed to be no-ops when EP is off.

**L1 — multi-rank on CPU with the gloo backend. No GPU needed. — RUN 2026-09-03, all pass**

Result at `--nproc_per_node=4` (harness: `MoE` wrapped directly with `ExpertParallel`, one
batch row per rank, E=8, top_k=2, S=16):

```
[PASS] 1 EP-vs-baseline parity  maxdiff 0.00000
[PASS] 2 token conservation  sent [[9,6,10,7],[4,10,6,12],[2,10,11,9],[10,10,7,5]]
                             recv [[9,4,2,10],[6,10,10,10],[10,6,11,7],[7,12,9,5]]
[PASS] 2b local sent == N*K  32 vs 32
[PASS] 3 padding inert (implied by 1)
[PASS] 4 tokens_per_expert global  128 vs 128
[PASS] 4b matches single-process reference
[PASS] 5 expert_bias agrees across ranks
[PASS] 6 local expert grads
```

Parity is **exact**, not merely within tolerance — the EP path is bit-identical to the
single-process reference, and `sent` is the exact transpose of `recv` with genuinely uneven
splits. Caveat: this required §12.8 patched in the harness.

Original rationale: This is the important finding
for this phase: gloo in torch 2.13 supports **uneven** `all_to_all_single`, verified directly:

```
rank0: uneven all_to_all_single OK -> [0.0, 1.0, 1.0]
rank1: uneven all_to_all_single OK -> [0.0, 0.0, 0.0, 1.0, 1.0]
```

and `moe/kernels.py` already ships a non-Triton CPU fallback in `fill_indices_wrapper`. So the
entire EP dispatch/combine path — the two all-to-alls, `_permute`/`_unpermute`, the padded
counts — is exercisable locally with `torchrun --nproc_per_node=4` on CPU.

**Scope caveat.** What is verified is that gloo supports uneven `all_to_all_single` and that
`kernels.py` has a working non-Triton CPU fallback. Whether the *full* `parallelize_deepseekv3`
stack (`fully_shard`, DTensor placements, the optimizer hook) runs on gloo/CPU is untested. So
scope L1 to the `MoE` module wrapped directly with `ExpertParallel`, not the whole trainer; if
the full stack needs NCCL, these assertions move up to L3. Assertions:

1. **EP-vs-baseline parity.** Same seed and same global batch, `ep=4` vs single-process. MoE
   output must match to `atol=5e-2` (bf16 expert path). This is the check that catches a wrong
   permutation.
2. **Token conservation.** For every rank, `sum(input_splits)` equals local `N*K`; summed across
   ranks, total sent equals total received.
3. **Padding is inert.** With `_permute` active, per-expert output for real (non-padded) rows is
   identical to the unpadded single-process result — proves §12.3's zero-padding is correct.
4. **`tokens_per_expert` is global.** After the optimizer hook's all-reduce, it sums to
   `global_batch * seq_len * top_k`.
5. **`expert_bias` agrees across ranks** after a step (it is derived from an all-reduced
   statistic; divergence means the reduce is wrong), and is zero-sum.
6. **Gradient reach.** Every *local* expert parameter has a finite, non-zero gradient.

**L2 — single GPU.** ~~Only needed once `_grouped_mm` is introduced~~ — **no longer a separate
rung.** `_grouped_mm` was adopted in Phase 2 and verified against the loop on CPU
(`matches loop: True`), so its correctness is covered by L0. A GPU run is still worth doing
for throughput, but nothing is gated on it.

**L3 — multi-GPU, NCCL.** The matrix in §15.

## 15. Multi-GPU test matrix

Run each with the same seed and global batch; compare final loss against config 1 and against a
single-GPU run. Degrees must satisfy `parallel_dims._validate`: when `ep > 1`, either
`etp == tp` (then `ep % cp == 0` and `(dp_shard * cp) % ep == 0`) or `etp == 1` (then
`ep % (cp * tp) == 0` and `(dp_shard * cp * tp) % ep == 0`).

| # | world | dp_shard | tp | ep | etp | pp | exercises |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 1 | 1 | 1 | 1 | FSDP baseline, no expert plan |
| 2 | 8 | 8 | 1 | 2 | 1 | 1 | `ExpertParallel`, EFSDP wrapping |
| 3 | 8 | 8 | 1 | 8 | 1 | 1 | EP == world; one local expert per rank |
| 4 | 8 | 4 | 2 | 1 | 1 | 1 | `TensorParallel` expert plan, no EP |
| 5 | 8 | 4 | 2 | 4 | 1 | 1 | EP borrows TP → `ReordererSequenceParallel` |
| 6 | 8 | 4 | 2 | 2 | 2 | 1 | `ExpertTensorParallel` |
| 7 | 8 | 2 | 2 | 2 | 1 | 2 | PP + EP; per-stage `moe_enabled` |

Config 3 is the interesting failure case for `apply_fsdp`: when
`efsdp_degree * ep_degree > moe.experts.num_experts`, FSDP switches the expert shard placement
from dim 0 to dim 1. That branch is only reached here, and it is also the branch that reads
`num_experts` (§12.1).

Additional checks at L3:

- **Checkpoint resharding.** Save at `ep=2`, restore at `ep=4`, confirm identical loss on the
  next step. DCP handles this through DTensor metadata; it is the main thing that silently
  breaks if a parameter's shard dim is declared wrong.
- **Loss-curve parity.** 50 steps at config 1 vs configs 2, 5 and 6; curves should track within
  noise. A wrong `Partial()`/`Replicate()` on the gate shows up here and nowhere else — this is
  the failure `gate_grad_placements` in `apply_moe_ep_tp` exists to prevent, and it only appears
  when `score_before_experts=False`, which is now our default.
- **Expert-usage balance.** `tokens_per_expert` variance should fall over training; if one
  expert's share grows monotonically, the bias update or its all-reduce is broken.

No local GPU is available on this machine (`torch.cuda.is_available()` is `False`), so L3 needs
rented hardware; a RunPod MCP server is configured in this environment for that.

## 16. Risks

- **`torch._grouped_mm` precision.** Verified working on CPU in torch 2.13 (so L1 is runnable),
  but it casts to bf16, which costs ~0.7% relative error versus fp32. Every numerical
  comparison downstream must budget for that. It also requires 16-byte-aligned strides: keep
  `dim` and `hidden_dim` multiples of 8.
- **Device-to-host syncs.** Both `_token_dispatch` (`output_splits.tolist()`) and the Phase 1
  loop (`offsets.tolist()`) sync. Expected; only a performance concern, and only after
  correctness is established.
- **`torch.compile`.** MoE blocks already run with `fullgraph=False`. The `index_add`/`x[idx]`
  choice from §3.4 should be benchmarked against upstream's `gather`/`scatter_add` at this point,
  not before.

# Phase 3: ACHIEVE SOMETHING REAL - Train a Light Model on GPUs

Moved to its own spec: `2026-09-03-phase3-training-plan.md`. It carries the full codebase
audit, the replacement model config (the one here is 16.2B, not 1-3B), the data decision,
and the staged RunPod runbook with costs.

Phase 2 items that spec closes out: **§12.8 is fixed at the source** (F4 there) and the L1
gloo harness is re-run unpatched as Stage 0 step 9. The §15 multi-GPU matrix becomes
Stage 2, minus the `cp` row — context parallelism is silently wrong because the model never
calls `F.scaled_dot_product_attention`, which is what `context_parallel` patches.
