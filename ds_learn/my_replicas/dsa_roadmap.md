# DSA implementation roadmap

Living document for implementing DeepSeek Sparse Attention (DSA) on top of `mla.ipynb`. Working notebook: `dsa.ipynb`. Paper: [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Reference code (target version, not V4-Pro): [V3.2 inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference).

## Status

| Phase | Title | Status |
|---|---|---|
| 1 | Setup & Config | ✅ Done |
| 2 | LightningIndexer module | ✅ Done |
| 3 | Wire indexer into MLA sparse attention | ✅ Done |
| 4 | Training pipeline (stages + KL + optimizer) | ✅ Done |
| 5 | Verify & smoke-test | ✅ Done |

## Prerequisites (already done in `mla.ipynb`)

- KV cache writes at `start_pos:end_pos` AND reads `[:end_pos]` for attention (was a silent no-op before — fixed).
- `RotaryEmbedding.forward` accepts `start_pos`.
- `LLM.forward` builds a `(S_q, T_k)` causal mask correctly for both prefill (S>1, start_pos=0) and decode (S=1, start_pos>0).
- `LLM.generate` prefills once then forwards a single token per step.
- Greedy-decode parity test: cache-on output == cache-off output (byte-identical).

`dsa.ipynb` is a byte-identical copy of `mla.ipynb` as the starting point.

---

## Phase 1 — Setup & Config ✅

**Goal**: expose DSA hyperparameters and a stage controller on the `Config`, without changing any behavior (`dsa_stage="off"` default keeps the model running plain MLA).

### Sub-steps

- [x] Add DSA hyperparams to `Config.__init__` signature
- [x] Bind them to `self` inside `__init__`
- [x] Re-instantiate `model = LLM(config)` so existing layers pick up the new config

### Fields added (cell `fdc48e97`)

```python
# dsa - lightning indexer
index_n_heads: int = 4
index_head_dim: int = 32
index_rope_head_dim: int = 16
index_topk: int = 32
share_q_lora_with_index: bool = True

# dsa - training stage
dsa_stage: Literal["off", "dense_warmup", "sparse"] = "off"
indexer_kl_weight: float = 1.0
```

### Constraints (verified on instantiation)

- `index_topk < max_seq_len` — can't select more positions than you have.
- `index_rope_head_dim <= index_head_dim`.
- `index_rope_head_dim` is even — RoPE rotates 2-D pairs.
- `dsa_stage` defaults to `"off"` so DSA stays dormant until we explicitly switch on.

---

## Phase 2 — LightningIndexer module ✅

**Goal**: a standalone `nn.Module` that, given the layer's input hidden states `h ∈ R^{B×S×d}`, produces index scores `I ∈ R^{B×S×T}` (Eq. 1 of the paper), and supports KV-caching the per-token `k^I` so decode is cheap.

Insert as a **new code cell**, immediately after the `MLA` cell (`35d8dd46`), before `FeedForward` (`26b0d80d`).

### Eq. 1 (target formula)

```
I_{t,s} = Σ_j  w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )
```

Three actors: per-head query `q^I_{t,j} ∈ R^{d^I}`, single-head key `k^I_s ∈ R^{d^I}`, per-head scalar gate `w^I_{t,j} ∈ R`.

### Sub-steps

- [x] 2.1 `__init__`: linear projections + RoPE module + `index_k_cache` buffer
- [x] 2.2 `forward`: implement Eq. 1 with cache write/read
- [x] 2.3 Decoupled RoPE on the last `index_rope_head_dim` dims of `q^I` and `k^I`
- [x] 2.4 Shape sanity check: `I.shape == (B, S, end_pos)` — passes for training/prefill/decode
- [x] 2.5 Bug-6 verification: cache row 1 untouched when B=1 < max_batch=2

### Shape contract

| Tensor | Shape | Notes |
|---|---|---|
| `h` (input) | `(B, S, hidden_size)` | layer input |
| `qr` (optional) | `(B, S, q_lora_rank)` | passed when `share_q_lora_with_index=True` |
| `q^I` | `(B, S, H_I, d_I)` | per-head |
| `k^I` (new) | `(B, S, d_I)` | single-head |
| `k^I` (after cache read) | `(B, end_pos, d_I)` | full prefix incl. new tokens |
| `w^I` | `(B, S, H_I)` | per-head scalar gate |
| `I` (output) | `(B, S, end_pos)` | index scores per (query, key) pair |

### Skeleton (fill in the last 3 lines yourself)

```python
class LightningIndexer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.H_I = config.index_n_heads
        self.d_I = config.index_head_dim
        self.rd  = config.index_rope_head_dim
        self.share_q_lora = config.share_q_lora_with_index and config.q_lora_rank > 0

        if self.share_q_lora:
            self.wq_b = nn.Linear(config.q_lora_rank, self.H_I * self.d_I)
        else:
            self.wq = nn.Linear(config.hidden_size, self.H_I * self.d_I)

        self.wk = nn.Linear(config.hidden_size, self.d_I)              # single-head
        self.w_proj = nn.Linear(config.hidden_size, self.H_I)          # per-head gate
        self.rotary_emb = RotaryEmbedding(self.rd, max_seq_len=config.max_seq_len)

        self.register_buffer('index_k_cache',
            torch.zeros(config.max_batch_size, config.max_seq_len, self.d_I),
            persistent=False)

    def forward(self, h, qr=None, use_kv_cache=False, start_pos=0):
        B, S, _ = h.shape
        end_pos = start_pos + S

        # 1. q^I (per-head)
        if self.share_q_lora:
            assert qr is not None, "share_q_lora_with_index=True but qr not provided"
            q = self.wq_b(qr)
        else:
            q = self.wq(h)
        q = q.view(B, S, self.H_I, self.d_I)

        # 2. k^I (single-head)
        k_new = self.wk(h)

        # 3. decoupled RoPE on last `rd` dims
        q_nope, q_rope = q[..., :-self.rd], q[..., -self.rd:]
        k_nope, k_rope = k_new[..., :-self.rd], k_new[..., -self.rd:]
        q_rope, k_rope = self.rotary_emb(q_rope, k_rope.unsqueeze(2), start_pos=start_pos)
        k_rope = k_rope.squeeze(2)
        q     = torch.cat([q_nope, q_rope], dim=-1)
        k_new = torch.cat([k_nope, k_rope], dim=-1)

        # 4. cache write & read (mirror MLA's pattern exactly)
        if use_kv_cache:
            self.index_k_cache[:B, start_pos:end_pos] = k_new
            k = self.index_k_cache[:B, :end_pos]
        else:
            k = k_new

        # 5. Eq. 1 — three lines (YOU fill these in):
        # scores = einsum(...)              # (B, S, H_I, T_k)
        # gated  = F.relu(scores) * w^I.unsqueeze(...)
        # I      = gated.sum(dim=2)         # (B, S, T_k)

        return I
```

### Stop-and-think

1. **Why is `k^I` single-head while `q^I` is multi-head?**
   *(Same family of reason as MLA's `k^R`. Cache size.)*
2. **Why ReLU instead of softmax across the indexer heads?**
   *(Hint: scale + top-k consistency across the sequence dim.)*

---

## Phase 3 — Wire indexer into MLA sparse attention ✅

**Goal**: MLA consults the indexer, picks top-k positions, attends only to those.

### Sub-steps

- [x] 3.1 Add `self.indexer = Indexer(config)` to `MLA.__init__`, guarded by `dsa_stage != 'off'`
- [x] 3.2 Force MQA-mode (absorb path) when `config.dsa_stage != "off"` — via combined `attn_impl=='naive' and dsa_stage=='off'` condition
- [x] 3.3 Compute indexer scores in MLA forward; thread `qr` through (q-lora intermediate post-`q_norm`)
- [x] 3.4 Build top-k sparse mask: `(B, S, T)` with 0 on selected positions, `-inf` elsewhere, via `scatter_`
- [x] 3.5 Add sparse mask to MLA's score tensor *before* softmax, with `.unsqueeze(2)` for head broadcast
- [x] 3.6 Smoke-tested: DSA-off vs DSA-sparse produce different outputs (sparse mask actually firing); decode step works

### Known limitation (deliberate scope cut)

Current implementation is the **mask-based** path (option (a)): full dense scores computed, then masked to -inf at non-top-k positions. This is correct mathematically but doesn't deliver the O(L·k) compute win. Real DSA uses the **gather-based** path (option (b)): gather only selected k positions, compute O(S·k) scores. Left as a Phase 6 optimization once everything else trains correctly.

### Implementation notes

- **Mask-based path (option a)**: simpler, computes dense scores then masks. Still O(L²) but conceptually correct. Good for learning replica.
- **Gather-based path (option b)**: real DSA. Gather k selected latents, compute O(L·k) scores. Skip for now; revisit if we want production-realistic compute.
- Causal mask still applies to the indexer scores BEFORE topk (must not select future positions).
- `topk` indices need to be `.detach()`-ed if used as gather indices — they aren't differentiable.

### Sketch (in MLA's MQA branch, before `scores.softmax(-1)`)

```python
if config.dsa_stage != "off":
    # qr comes from the q_lora path: qr = self.q_norm(self.wq_a(hidden_states))
    I = self.indexer(hidden_states_for_indexer, qr=qr_for_indexer,
                     use_kv_cache=use_kv_cache, start_pos=start_pos)
    # causal mask on the indexer (so top-k can't pick future)
    I = I + mask if mask is not None else I
    topk_idx = I.topk(min(config.index_topk, I.size(-1)), dim=-1).indices  # (B, S, k)
    # build a (B, S, T_k) mask with 0 on selected, -inf elsewhere
    sparse_mask = torch.full_like(I, float('-inf'))
    sparse_mask.scatter_(-1, topk_idx, 0.0)
    scores = scores + sparse_mask.unsqueeze(2)   # broadcast over head dim
```

### Stop-and-think

- Where do we read `h` (the indexer input) and `qr` from inside MLA, given that MLA's forward currently consumes already-`input_layernorm`'d hidden states? Does the indexer want the post-norm or pre-norm `h`? (Look at the V3.2 reference impl to confirm.)

---

## Phase 4 — Training pipeline ✅

**Goal**: implement the two-stage training from §2.1 of the paper.

### Sub-steps

- [x] 4.1 Stage controller threaded via `self.config.dsa_stage` inside MLA + LLM
- [x] 4.2 Param-freeze helper `freeze_for_dsa_warmup(model)` (only `*.indexer.*` trainable)
- [x] 4.3 Indexer KL loss `compute_indexer_kl(...)`
  - [x] Eq. 3 (warmup): full distribution `p_{t,:}`
  - [x] Eq. 4 (sparse): gather `p` and `I` at `topk_idx`, then KL
- [x] 4.4 Per-layer KL accumulation in `LLM.forward`, averaged across layers, weighted by `indexer_kl_weight`
- [x] 4.5 Detach `hidden_states` and `qr` before indexer call in sparse stage only
- [x] 4.6 `DSATrainer` overrides `create_optimizer` with two AdamW groups (main_lr vs indexer_lr)

### Bugs encountered + fixed during implementation

| # | Bug | Fix |
|---|---|---|
| 1 | Wrong stage name (`'dense'` instead of `'dense_warmup'`) | match Config Literal |
| 2 | `last_attn_probs` stashed BEFORE softmax (raw scores, not probs) | move stash AFTER softmax |
| 3 | `F.kl_div(..., log_target=True)` + `torch.log(p)` produces NaN when `p=0` | revert to standard `log_target=False` |
| 4 | Vocab mismatch (Config vocab=6400 vs Qwen2 tokenizer 151,665) | `Config(vocab_size=len(tokenizer))` |
| 5 | KL NaN from `0 * (-inf)` at causal-masked positions (found in Phase 5) | `log_q = torch.where(p > 0, log_q, 0)` before kl_div |

### Key paper details

- Warmup: 1e-3 LR on indexer only, ~1000 steps. Dense main attention.
- Sparse: 7.3e-6 LR on main, separate LR on indexer. Top-k = 2048 in paper (we use 32).
- Detach is critical: indexer trained only by L^I, main trained only by LM loss.

### KL target construction (warmup, Eq. 3)

```python
# main attention's softmax: scores after softmax, shape (B, S, n_h, T)
p = main_softmax.detach().sum(dim=2)                  # aggregate heads -> (B, S, T)
p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)   # L1 normalize over T
log_q = F.log_softmax(I, dim=-1)                      # (B, S, T)
# F.kl_div expects (log_q, p) with input=log-probs, target=probs
kl = F.kl_div(log_q, p, reduction="batchmean")
```

For sparse stage (Eq. 4), gather both `p` and `I` along `topk_idx` before the KL.

### Optimizer setup

```python
indexer_params = [p for n, p in model.named_parameters() if "indexer" in n]
main_params    = [p for n, p in model.named_parameters() if "indexer" not in n]
optimizer = torch.optim.AdamW([
    {"params": main_params,    "lr": 7.3e-6},
    {"params": indexer_params, "lr": 1e-3},
])
```

Need to override `Trainer.create_optimizer` or pass a custom optimizer.

---

## Phase 5 — Verify & smoke-test ✅

**Goal**: prove correctness with three small tests before training for real.

### Sub-steps (all passed)

- [x] 5.1 **Shape & smoke**: forward in all three `dsa_stage` modes (off / dense_warmup / sparse) produces finite loss
- [x] 5.2 **Off-mode regression**: `hasattr(attn, 'indexer') == False` when `dsa_stage='off'`
- [x] 5.3 **Cache parity under DSA**: greedy decode with `use_kv_cache=True` and `False` produces byte-identical tokens
- [x] 5.4 **Warmup freeze**: 48/48 indexer params change after one optimizer step, 0/115 main params change
- [x] 5.5 **Loss sanity & grad routing**: total loss finite; KL grad reaches indexer weights, LM grad reaches main weights, no cross-contamination

### Why cache parity test matters

The indexer also has a KV cache (`index_k_cache`). If it's mis-indexed by `start_pos`, decode will silently produce different tokens than prefill would. Same gold-standard correctness test we used for the MLA cache fix. **Confirmed passing.**

---

## References

- **Paper**: [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- **Reference impl (V3.2)**: `huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference`
- **NOT this one**: `DeepSeek-V4-Pro/inference/model.py` — different architecture (NSA-style block compression). Useful for shape/pattern reference but not on-target for V3.2.

## Open questions / TODOs to revisit

- Mask-based vs gather-based sparse path (Phase 3): start with mask, decide later if we want the real perf win.
- Chunked prefill handling (`use_kv_cache=True, S>1, start_pos>0`): not currently supported; deliberate scope cut.
- FP8 indexer: paper uses FP8 for the indexer's compute. We skip — bf16/fp32 is fine for a replica.
- Token gating for the gate `w^I`: paper doesn't apply any norm/activation on `w^I` before multiplying. We follow.
