# CSA / HCA implementation roadmap

Living document for implementing DeepSeek-V4's hybrid attention — **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)** — on top of `dsa.ipynb`. Working notebook: `csa_hca.ipynb` (start as a byte-identical copy of `dsa.ipynb`, exactly like `dsa.ipynb` started from `mla.ipynb`). Paper: DeepSeek-V4 (§2.3).

Reference implementations (the real upstream algorithms these are assembled from):
- **NSA** (the compression + selection + sliding-window pattern CSA generalizes): [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) — see `native_sparse_attention.py`. Original paper: [Native Sparse Attention (2502.11089)](https://huggingface.co/papers/2502.11089).
- **Lightning indexer** (CSA's top-k selector): you already built this in `dsa.ipynb` (`Indexer`, cell `d48e89b0`). CSA reuses it almost verbatim — just over compressed blocks.
- **Muon**: [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) — this is the exact variant the paper cites (weight decay + Newton-Schulz + RMS rescale).
- The V4-Pro HF `inference/model.py` is now available locally at `ds_learn/official_code/DeepSeek-V4-Pro/inference/model.py` — **this is the primary reference.** Its single `Attention` class (parameterized by `compress_ratios[layer_id]`, with an `Indexer` only when `ratio==4`) + `Compressor` + `Indexer` are exactly what we mirror. We anchor on it plus the §2.3 equations.

---

## Direction update (2026-06-10 decision log)

Two forks were settled with the notebook owner; they **supersede** parts of the original phase plan below:

1. **Unify HCA + CSA into ONE class**, mirroring official `Attention` (switch on `compress_ratio`: `0` → pure sliding-window; `m` = `compress_rate` → CSA with overlap + indexer; `m'` = `hca_compress_rate` → HCA dense). This **replaces the sibling `HCA`/`CSA` design** in Phases 3–4. The existing `HCA` draft becomes the seed of the unified class (now named `Attention`). `MLA` stays a separate, selectable A/B path.
2. ~~**Prefill / training-only scope.**~~ **SUPERSEDED 2026-06-11 — see below.**

### Scope update (2026-06-11)

**Decode / KV cache is back IN scope.** The earlier prefill-only cut is reversed: we now build the full decode path (window cache + compressed/state cache + indexer-key cache, mirroring official `Compressor`) and **un-defer the cache-parity test**. New phase added (Phase 6 — Decode & KV cache); verify/Muon renumbered to 7/8.

Also from the 2026-06-11 re-audit of the unified `Attention` cell — **five issues to fix before CSA** (the HCA prefill path currently can't run; see Phase 3 "Re-audit findings").

The equation-by-equation primer and Phase 2 are unaffected; Phases 1, 3–8 below reflect both updates.

---

## ⚠️ Read this first: the conceptual primer

You said you "don't get CSA/HCA at all." That's because the paper introduces ~6 ideas at once. Here's the decomposition. **None of these are new to you except the compressor** — you've already built the indexer, RoPE, and KV-cache discipline in `dsa.ipynb`.

### The one-sentence mental model

> **HCA** = "shrink the KV sequence by a big factor ($m' \approx 128$), then do plain attention over the shrunk sequence."
> **CSA** = "shrink the KV sequence by a small factor ($m \approx 4$), then do **DSA top-k** over the shrunk sequence."

Both end with the *same* core: Multi-Query Attention over the shrunk KV + a grouped output projection. The only difference is **how hard you compress** and **whether you then sparsify with the indexer**.

### How this relates to what you know

| Concept | Where you've seen it | What's new in CSA/HCA |
|---|---|---|
| Low-rank Q projection ($c^Q_t = h_t W_{DQ}$) | MLA's `wq_a`/`q_lora` | Same idea; Q latent is *shared* between indexer and core attention |
| Lightning indexer + top-k | DSA (`Indexer`, cell `d48e89b0`) | Operates on **compressed blocks**, not raw tokens |
| Decoupled / partial RoPE | MLA's `qk_rope_head_dim` split | Applied to last 64 dims; **plus a `-i` trick** on outputs (see below) |
| KV cache write/read at `start_pos` | MLA, Indexer | Now caches *compressed* entries + a small *state cache* of uncompressed tail |
| MQA (single KV head) | MLA "absorb" path is MQA-like | Core attention is explicitly MQA: one shared K=V entry per block |
| **Token-level compressor** | **nothing yet** | **This is the genuinely new piece. Build it first (Phase 2).** |

### Do we still use MLA? (two orthogonal compression axes)

Common confusion: "V3.2 put DSA *inside* MLA; in V4 the token compressor seems to replace MLA's compression — so is MLA gone?" The trap is thinking MLA's compression and the token compressor are the same operation. They compress **different axes**:

| | What it shrinks | Tokens after | Cache stores |
|---|---|---|---|
| **MLA latent** (V3/V3.2) | feature/head dim per token ($d \to c$) | unchanged ($n$) | one latent **per token** |
| **CSA/HCA token compressor** (V4) | sequence dim ($m$ tokens $\to 1$) | shrunk ($n/m$) | one entry **per block** |
| **DSA/CSA top-k** | *which* entries you attend | — | (compute only) |

V4's verdict, split by MLA's two halves:
- **KV half — replaced.** No more per-token latent + per-head up-projection ($W_{UK}, W_{UV}$/absorb). CSA/HCA project $H \to$ head-dim-$c$ entries (Eq 9/20), pool $m$ tokens into one, and run **plain MQA** (one shared compressed entry = K *and* V for all heads). Per-head variety now comes only from the queries + grouped output projection.
- **Query half — kept.** The low-rank query latent $c^Q_t = h_t W_{DQ}$ then up-project (Eq 13–14, 18, 24–25) survives in both CSA and HCA, now *shared* with the indexer.

Mental bridge: **CSA's core ≈ your MLA "absorb"/MQA mode, but the cached latent is pooled across $m$ tokens (and sparsely selected) instead of stored per-token.** ⚠️ In V3.2 the indexer scored against MLA's per-token latent; in V4 the per-token latent is gone and the indexer scores against the *token-compressed block* entries — don't conflate the two.

**Implication for this replica:** build CSA/HCA as **new sibling modules, not MLA subclasses** (Phase 3/4). Reuse `Indexer`, `RotaryEmbedding`, `RMSNorm`, and lift MLA's `wq_a`/`wq_b` query path. Keep `MLA` selectable for the first layers / A-B testing.

### CSA, decoded equation-by-equation (§2.3.1)

Let $H \in \mathbb{R}^{n \times d}$ be the layer input ($n$ = seq len, $d$ = hidden). Let $c$ = core head dim (512 in V4-Pro), $m$ = compression rate ($=4$).

1. **Two KV series + two weight series** (Eq 9–10). Project $H$ four ways — two *value* series and two *logit* series, all in $\mathbb{R}^{n \times c}$:

$$C^a = H W_{aKV}, \quad C^b = H W_{bKV}, \qquad Z^a = H W_{aZ}, \quad Z^b = H W_{bZ}$$

2. **Overlapping softmax pooling → one entry per $m$ tokens** (Eq 11–12). For compressed entry $i$, take block $i$ from the $a$-series (rows $mi : m(i{+}1){-}1$) **and** block $i{-}1$ from the $b$-series (rows $m(i{-}1) : mi{-}1$), add the learnable intra-block positional biases $B^a, B^b \in \mathbb{R}^{m \times c}$, and softmax **down the token axis, per channel** over the concatenated $2m$ rows:

$$\big[\,S^a_{\,mi:m(i+1)-1}\,;\ S^b_{\,m(i-1):mi-1}\,\big] \;=\; \operatorname{softmax}_{\text{row}}\!\big(\big[\,Z^a + B^a\,;\ Z^b + B^b\,\big]\big)$$

$$C^{\mathrm{Comp}}_i \;=\; \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j \;+\; \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j$$

   - **Why two series?** Entry $i$ blends block $i$ with its predecessor $i{-}1$ → a $2m$-token receptive field, while consecutive entries still advance by $m$, so net compression is exactly $\tfrac{1}{m}$. ($B^a, B^b$ are NSA's "intrablock positions" — they tell the pool *where* in the block each token sits.)
   - Edge case $i=0$: pad $C^b$ with zeros and $Z^b$ with $-\infty$ (so the softmax ignores the nonexistent predecessor).

3. **Compress the indexer keys the same way** → $K^{\mathrm{IComp}} \in \mathbb{R}^{(n/m) \times c_I}$. The indexer now scores *query-token → compressed-block*, not token→token. Everything else in your `Indexer` (low-rank queries Eq 13–14, ReLU-gated per-head sum Eq 15–16) is unchanged.

4. **Top-k over blocks** (Eq 17): $\operatorname{Top\text{-}k}(I_{t,:})$ with the causal constraint $s < \lfloor t/m \rfloor$ — a query may only see blocks that fully precede it.

5. **Shared-KV MQA over selected blocks** (Eq 18–19): queries $q_{t,h} = c^Q_t W_{UQ}$ (latent $c^Q_t$ shared with the indexer), one KV head per block (the compressed entry is *both* key and value):

$$o_{t,h} \;=\; \operatorname{CoreAttn}\!\big(q_{t,h},\ C^{\mathrm{Comp}}_{\text{sel}},\ C^{\mathrm{Comp}}_{\text{sel}}\big)$$

6. **Grouped output projection**: $c \cdot n_h$ is huge, so don't project it to $d$ in one matmul. Split $n_h$ heads into $g$ groups, project each group $\tfrac{c\, n_h}{g} \to d_g$, concat the $g$ groups $(d_g \cdot g)$, then project $\to d$.

### HCA, decoded (§2.3.2)

Same skeleton, **three simplifications**:
1. **Single series, non-overlapping** (Eq 20–23): just $C = H W_{KV}$, $Z = H W_Z$, bias $B \in \mathbb{R}^{m' \times c}$, softmax over $m'$ tokens per channel, weighted sum. Compression $\tfrac{1}{m'}$ with $m' \approx 128$:

$$S_{\,m'i:m'(i+1)-1} = \operatorname{softmax}_{\text{row}}\!\big(Z_{\,m'i:m'(i+1)-1} + B\big), \qquad C^{\mathrm{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j$$

2. **No indexer, no top-k** — dense MQA over *all* compressed entries.
3. Otherwise identical: shared-KV MQA + grouped output projection.

So if you build a `TokenCompressor` that supports both "overlapping (CSA)" and "simple (HCA)" modes, **HCA is ~30 lines and CSA is HCA + your existing indexer.**

### The four shared "other details" (§2.3.3) — apply to BOTH

These are small but easy to forget. Add them once, reuse in both modules.

- **Q/KV-entry RMSNorm**: RMSNorm each query head and the single compressed-KV head *just before* core attention. (You have `RMSNorm`, cell `749d72cd`.) Prevents exploding logits — and is *why* V4 can skip QK-Clip in Muon.
- **Partial RoPE + the $-i$ output trick**: apply RoPE to the **last 64 dims** of $q$ and of each KV entry. Because the KV entry is *also the value*, the naive output $o_{t,i}$ carries *absolute* position; counter it by applying RoPE with position $-i$ to the last 64 dims of each head output. Net effect: relative position survives into the output.
- **Sliding-window branch**: produce $n_{\text{win}}$ ($=128$) *uncompressed* KV entries for the most recent $n_{\text{win}}$ tokens; attend to them *alongside* the compressed/selected entries in the same softmax. Reason: a query can't see tokens inside its own not-yet-compressed block, and recent tokens matter most.
- **Attention sink** (Eq 27): learnable per-head logits $z'_h$ added to the softmax denominator so a head can attend to *almost nothing*:

$$s_{h,i,j} \;=\; \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) \;+\; \exp(z'_h)}$$

### Build order (important)

```
TokenCompressor (Phase 2)  →  HCA (Phase 3, simpler)  →  CSA (Phase 4, = HCA + indexer top-k)
```

Resist the urge to start with CSA. HCA forces you to get the compressor, MQA core, grouped projection, sliding window, sink, and partial-RoPE right *without* the sparse-selection complexity. Then CSA is a small delta.

---

## Status

| Phase | Title | Status |
|---|---|---|
| 1 | Setup & Config (hyperparams + layer schedule) | ✅ Done (assert fixed → `self.compress_rate`; `Config()` instantiates) |
| 2 | `TokenCompressor` module (CSA overlap + HCA simple) | ✅ Done (smoke-test green) |
| 3 | Unified `Attention` — HCA path (prefill) | ✅ Done — re-audit issues A–E fixed; HCA smoke test green (2026-06-11) |
| 4 | CSA path in the same class (overlap compressor + indexer top-k **over blocks**) | ✅ Done — wired into `Attention`; CSA smoke test (off/dense_warmup/sparse + causality) green (2026-06-11) |
| 5 | Wire into `DecoderLayer` by schedule (`mla`/`swa`/`hca`/`csa`) | ✅ Done — `LLM` fwd+bwd green across stages; MLA reverted to pure; 4.6 + 5.6 edge bug fixed & re-verified (2026-06-11) |
| 6 | **Decode & KV cache** (window + compressed/state + indexer-key; mirror official `Compressor`) | ✅ Done — cache-parity **bit-exact** (~3e-6) for HCA/CSA-dense/mix/MLA (2026-06-11) |
| 7 | Verify & smoke-test (shapes, causality, sink, sparse≠dense, **+ cache-parity**) | ✅ Done — `cache_parity` + Phase-7 verification cells added & green (2026-06-11) |
| 8 | Muon optimizer (separate, self-contained) | ⬜ Not started |
| — | mHC (manifold-constrained hyper-connections) | → own roadmap (official `Block.hc_*` + `hc_split_sinkhorn`) |

## Prerequisites (already done in `dsa.ipynb`)

- `Indexer` (lightning indexer) with KV-cached single-head keys + low-rank queries + ReLU-gated per-head sum.
- `RotaryEmbedding.forward(q, k, start_pos)` slicing cos/sin at `start_pos`.
- KV-cache write `[start_pos:end_pos]` / read `[:end_pos]` discipline, proven with a greedy-decode parity test.
- Causal `(S_q, T_k)` mask construction in `LLM.forward`.

`csa_hca.ipynb` starts as a byte-identical copy of `dsa.ipynb`.

---

## Phase 1 — Setup & Config

**Goal**: expose CSA/HCA hyperparameters and a per-layer attention schedule, without changing behavior (default schedule = all-MLA, so the model runs exactly as today).

### Sub-steps

- [x] 1.1 Copy `dsa.ipynb` → `csa_hca_muon.ipynb`; MLA smoke-test (cell `27f0c8c0`) kept as a parity guard.
- [x] 1.2 Add CSA/HCA fields to `Config.__init__` (cell `fdc48e97`) and bind to `self`.
- [x] 1.3 `default_attn_schedule` (moved into the Config cell); `Config.__init__` auto-fills `attn_schedule` when `None`.
- [x] 1.4 `LLM(config)` instantiates with the dormant fields (layer wiring deferred to Phase 5).
- [x] Constraint asserts added at the end of `__init__` (cheap mis-config guards).

> **As implemented — name map** (your notebook uses `compress_*`, not the `core_*` placeholders below):
> `compress_head_dim`=c, `compress_n_heads`=n_h, `compress_rope_dim`=RoPE dims, `compress_rate`=m (CSA),
> `hca_compress_rate`=m′ (HCA), `index_block_topk`=top-k over blocks, `sliding_window`=n_win, `use_attention_sink`.
> `out_proj_groups`/`out_proj_group_dim` are **computed inline** (commented derivation: `n_attn_heads // n_kv_heads` and `hidden // groups`), not stored as fields — the divisibility assert reuses that same expression.

### Fields to add

```python
# CSA/HCA — shared core attention
core_head_dim: int = 64          # `c` in the paper (V4-Pro uses 512; small here)
core_n_heads: int = 8            # `n_h`
out_proj_groups: int = 2         # `g`
out_proj_group_dim: int = 64     # `d_g`
core_rope_dim: int = 16          # paper uses 64; "last rd dims get RoPE"
sliding_window: int = 64         # `n_win` (paper 128)
use_attention_sink: bool = True

# CSA-specific
csa_compress_rate: int = 4       # `m`
csa_topk_blocks: int = 16        # top-k over COMPRESSED blocks (paper 512/1024)

# HCA-specific
hca_compress_rate: int = 32      # `m'` (paper 128; keep ≫ m)

# per-layer attention type; None => infer a default interleave
attn_schedule: list[str] | None = None
```

### Default schedule helper (paper §4.2.1)

V4-Flash: first 2 layers pure SWA, then CSA/HCA **interleaved**. V4-Pro: first 2 layers HCA, then interleaved. For your replica, a sane default for `n_layers=8`:

```python
def default_attn_schedule(n_layers):
    sched = []
    for i in range(n_layers):
        if i < 2:            sched.append("hca")     # heavy-compress early layers
        elif i % 2 == 0:     sched.append("csa")
        else:                sched.append("hca")
    return sched
```

Keep `"mla"` selectable too, so you can A/B a single layer against your known-good MLA.

### Constraints to assert on instantiation

- `hca_compress_rate % csa_compress_rate == 0` — so a KV-cache block can hold an integer number of both (the paper aligns cache blocks to $\operatorname{lcm}(m, m')$).
- `core_n_heads % out_proj_groups == 0`.
- `core_rope_dim <= core_head_dim` and even.
- `csa_topk_blocks < max_seq_len // csa_compress_rate`.

### Stop-and-think

1. Why does the paper align KV-cache blocks to $\operatorname{lcm}(m, m')$ rather than caching CSA and HCA layers independently? *(Hint: Figure 6 — one cache block must yield a whole number of both CSA and HCA compressed entries.)*

---

## Phase 2 — `TokenCompressor` module ⭐ (the new idea)

**Goal**: one module that turns $H \in \mathbb{R}^{B \times n \times d}$ into compressed KV entries $C^{\mathrm{Comp}} \in \mathbb{R}^{B \times (n/r) \times c}$, supporting both **CSA overlapping mode** (Eq 9–12) and **HCA simple mode** (Eq 20–23).

> **Status**: scaffolded in cell `d89c8c70` (between `Indexer` and `MLA`) + smoke-test cell `143c9e81`. `__init__` and both forward-method scaffolds are wired to your `compress_*` config names; the **pooling core is left for you** (the `# YOU FILL` blocks). Build `_simple` (HCA) first, then `_overlap` (CSA). All field names below use `compress_head_dim`(c) / `compress_rate`(r) / `hca_compress_rate`(m′).

### Shape contract

| Tensor | Shape | Notes |
|---|---|---|
| $H$ (input) | `(B, n, d)` | layer input (pre- or post-norm — see stop-and-think) |
| $C^a, C^b$ / $C$ | `(B, n, c)` | value series |
| $Z^a, Z^b$ / $Z$ | `(B, n, c)` | pooling logits |
| $B^a, B^b$ / $B$ | `(r, c)` | learnable intra-block positional bias |
| $C^{\mathrm{Comp}}$ (output) | `(B, n//r, c)` | one entry per $r$ tokens |

### HCA / simple mode (do this first — Eq 20–23)

```python
class TokenCompressor(nn.Module):
    def __init__(self, config, compress_rate, overlapping: bool):
        super().__init__()
        self.r = compress_rate
        self.c = config.core_head_dim
        self.overlapping = overlapping
        d = config.hidden_size
        # simple mode needs one (C, Z); overlapping needs (a, b) pairs
        self.w_kv = nn.Linear(d, self.c, bias=False)
        self.w_z  = nn.Linear(d, self.c, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.r, self.c))   # B in Eq 22
        if overlapping:
            self.w_kv_b = nn.Linear(d, self.c, bias=False)
            self.w_z_b  = nn.Linear(d, self.c, bias=False)
            self.bias_b = nn.Parameter(torch.zeros(self.r, self.c))

    def _simple(self, H):
        B, n, _ = H.shape
        r = self.r
        n_trim = (n // r) * r                       # drop trailing < r tokens (paper does this)
        C = self.w_kv(H[:, :n_trim]).view(B, n//r, r, self.c)   # (B, nb, r, c)
        Z = self.w_z (H[:, :n_trim]).view(B, n//r, r, self.c)
        # YOU fill in (3 lines):
        # S = softmax over the `r` axis of (Z + self.bias)        -> (B, nb, r, c)
        # Ccomp = (S * C).sum(over r axis)                        -> (B, nb, c)
        # return Ccomp
```

### CSA / overlapping mode (Eq 11–12) — the tricky one

The clean way to get the `a`(block i) + `b`(block i-1) overlap without a Python loop: build block `i` from the `a` series, build the *shifted* block `i-1` from the `b` series (shift the whole `b` sequence right by `r`, zero-pad the front, `-inf`-pad the `b` logits for entry 0), concat along the token axis to length `2r`, softmax over that `2r` axis per channel, weighted-sum.

```python
    def _overlap(self, H):
        B, n, _ = H.shape
        r = self.r
        n_trim = (n // r) * r
        Ca = self.w_kv  (H[:, :n_trim]).view(B, n//r, r, self.c)
        Za = self.w_z   (H[:, :n_trim]).view(B, n//r, r, self.c) + self.bias
        Cb = self.w_kv_b(H[:, :n_trim]).view(B, n//r, r, self.c)
        Zb = self.w_z_b (H[:, :n_trim]).view(B, n//r, r, self.c) + self.bias_b
        # shift the b-series blocks right by one block (block i uses b-block i-1):
        Cb = torch.cat([torch.zeros_like(Cb[:, :1]),  Cb[:, :-1]], dim=1)   # zeros for entry 0
        Zb = torch.cat([torch.full_like(Zb[:, :1], float('-inf')), Zb[:, :-1]], dim=1)
        # YOU fill in (4 lines):
        # C2 = concat([Ca, Cb], over the r axis)     -> (B, nb, 2r, c)
        # Z2 = concat([Za, Zb], over the r axis)     -> (B, nb, 2r, c)
        # S  = softmax over the 2r axis of Z2
        # Ccomp = (S * C2).sum(over 2r axis)         -> (B, nb, c)
        # return Ccomp
```

> Note: the `_overlap` block-shift gives the *same* receptive field as the paper's index arithmetic (`a` rows `[m·i:…]`, `b` rows `[m·(i-1):…]`) but is vectorized and cache-friendly. Verify equivalence on a tiny tensor before trusting it.

### Sub-steps

- [x] 2.1 `_simple` (HCA) + `_overlap` (CSA) forward paths.
- [x] 2.2 `forward(H)` dispatches on `self.overlapping`.
- [x] 2.3 Smoke test (cell `143c9e81`): shapes `(2,16,64)`/`(2,2,64)` + finite. **Green.**
- [x] 2.4 Tiny `r=2` overlap shape check.

> **Key gotcha (resolved):** the pooling softmax is over the **token axis (`dim=2`, size `r`/`2r`), per channel** — NOT `dim=-1`. Softmaxing over the channel axis makes entry-0's `-inf` `b`-half an all-`-inf` row → NaN. Same lesson recurs in Phase 3 (all-masked early queries).

### Stop-and-think

1. The softmax is "per channel over the token axis." Concretely: for a fixed block and fixed channel `c`, you get `r` (or `2r`) weights summing to 1, then a convex combination of that channel's values. **Why per-channel and not a single scalar weight per token?** *(Hint: it's a learned, content-dependent pooling — different channels can pool from different tokens in the block.)*
2. Does the compressor see pre-norm or post-norm `H`? Check what your `DecoderLayer` (cell `2fc7a50b`) feeds attention, and keep it consistent with how `Indexer` is fed.
3. The trailing `< r` tokens are dropped from compression. Where do those tokens get attended to instead? *(Answer: the sliding-window branch + the "state cache" in inference. For training you can ignore them or fold into SWA.)*

---

## Phase 3 — Unified `Attention` class — HCA path (prefill)

**Goal**: ONE attention class (mirroring official `Attention`) switched by `compression_rate`. Build the **HCA path first** (`rate = m'`, `overlapping=False`, no indexer): `compress → shared-KV MQA over all compressed entries + sliding window → grouped output projection`, with Q/KV norm, partial RoPE (+`-i` trick), and attention sink. Prefill first; decode added in Phase 6. CSA (Phase 4) is the *same class* with `overlapping=True` + indexer top-k — do **not** make a second class.

The class now exists (cell `1512b460`, renamed `HCA` → `Attention(config, compression_rate)`), but the rename/refactor introduced bugs that stop `forward` from running.

### Re-audit findings (2026-06-11) — ✅ all fixed, HCA smoke test green

| # | Bug | Fix |
|---|---|---|
| A | `_apply_rotary_emb`, `_get_window_topk_idxs`, `_get_compress_topk_idxs` are **class methods with no `self`**. Called as `self._fn(...)`, Python binds `self`→first param → args shift → crash. | Add `@staticmethod` to all three (none use instance state). |
| B | `forward` calls `self.group_proj` / `self.out_proj`, but `__init__` named them `self.wo_a` / `self.wo_b` → `AttributeError`. | Use `self.wo_a` / `self.wo_b`. |
| C | Final lines `a = self.out_proj(o); return o` return the **pre-projection** `o` `(B,S,g*dg)`, not `(B,S,d)`. | `return self.wo_b(o)`. |
| D | Smoke test still calls `HCA(config)` (old name + old signature). | `Attention(config, config.hca_compress_rate)`. |
| E (latent) | `TokenCompressor(config, config.hca_compress_rate, overlapping=False)` **hardcodes HCA**, ignoring the `compression_rate` arg, while `self.m = compression_rate` drives block-RoPE. Coincides for HCA, breaks for CSA. | `TokenCompressor(config, compression_rate, overlapping=(compression_rate == config.compress_rate))`. |

Everything else in the class (gather-based selection = official `sparse_attn` index lists, sink-as-column, `-i` inverse-RoPE, partial RoPE, window branch, grouped proj dims `nh*c/g → o_lora_rank → d`) already tracks official and carries over to CSA unchanged.

### Forward skeleton

```python
class HCA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nh = config.compress_n_heads
        self.c  = config.compress_head_dim
        self.rd = config.compress_rope_dim
        self.compressor = TokenCompressor(config, config.hca_compress_rate, overlapping=False)
        # low-rank shared query latent (reuse MLA's pattern / q_lora if present)
        self.w_dq = nn.Linear(config.hidden_size, config.q_lora_rank or config.hidden_size, bias=False)
        self.w_uq = nn.Linear(config.q_lora_rank or config.hidden_size, self.nh * self.c, bias=False)
        self.q_norm  = RMSNorm(self.c)
        self.kv_norm = RMSNorm(self.c)
        self.rotary  = RotaryEmbedding(self.rd, max_seq_len=config.max_seq_len)
        # grouped output projection (groups/d_g computed inline — not stored as config fields)
        g  = config.num_attention_heads // config.num_key_value_heads
        dg = config.hidden_size // g
        self.group_proj = nn.Linear((self.c * self.nh) // g, dg, bias=False)   # per-group
        self.out_proj   = nn.Linear(dg * g, config.hidden_size, bias=False)
        if config.use_attention_sink:
            self.sink = nn.Parameter(torch.zeros(self.nh))     # z'_h, Eq 27
        # + sliding-window K/V projections (uncompressed), n_win recent tokens
        # + KV caches: compressed-entry cache, and a small uncompressed "state" cache

    def forward(self, h, mask=None, use_kv_cache=False, start_pos=0):
        B, S, _ = h.shape
        # 1. compressed KV entries  -> Ccomp (B, Tc, c); cache them at block granularity
        # 2. queries: cq = w_dq(h); q = w_uq(cq).view(B, S, nh, c); q = q_norm(q)
        # 3. partial RoPE on last rd dims of q and of Ccomp (single KV head)
        #    Ccomp = kv_norm(Ccomp)
        # 4. MQA scores: einsum("bshc,btc->bsht", q, Ccomp) / sqrt(c)
        #    + causal/block mask so query t only sees blocks that fully precede it
        #    + sliding-window entries concatenated along the key axis
        # 5. attention sink: append exp(sink) to the softmax denominator
        #    (trick: concat a constant "sink logit" column, softmax, then drop that column)
        # 6. o = einsum("bsht,btc->bshc", probs, Ccomp_value)
        # 7. partial-RoPE the OUTPUT with position -t on last rd dims (the "-i" trick)
        # 8. grouped output projection -> out_proj -> (B, S, hidden)
        ...
```

### Sub-steps (incremental — keep it finite at every step)

- [ ] 3.1 Compressed-entry MQA core + block-causal mask **+ attention sink**. The sink is *not* a later add-on here: early queries (positions `< m′`) see **zero** visible blocks, so without the sink their softmax row is all `-inf` → NaN (the exact Phase-2 failure mode). The sink term in the denominator makes a zero-visible-block row produce a finite all-zero output. Target: right shapes + finite.
- [ ] 3.2 Grouped output projection (replace the placeholder `(c·nh)→d` Linear; same capacity, cheaper).
- [ ] 3.3 Q/KV RMSNorm + partial RoPE on inputs (last `compress_rope_dim` dims).
- [ ] 3.4 The `-i` output RoPE trick; sanity-check two identical KV entries at different positions now contribute differently.
- [ ] 3.5 Sliding-window branch (uncompressed recent `n_win` tokens) into the same softmax — this is what covers the query's own/recent block that the causal mask excludes.
- [ ] 3.6 KV-cache write/read for compressed entries (mirror `Indexer`/`MLA` discipline at *block* granularity).

### The block-causal mask (subtle)

Query at 0-indexed position $t$ may attend compressed block $s$ **only if $s < \lfloor t/m' \rfloor$** — strictly blocks *before the query's own block*. Why exclude its own block even when $t$ is that block's last token? The block's compressed entry is built from **all** $m'$ of its tokens; for an *earlier* query in the same block that entry would leak future tokens. Since the entry is shared by every query, the whole block is masked for all its members. The own/recent tokens are covered by the **sliding window** (3.5) instead. Build the `(S, Tc)` mask as `col < (row // m')` → `0.0` else `-inf` (use `start_pos + row` for the absolute query position once caching is added).

### Stop-and-think

1. Grouped output projection: a full $(c\, n_h) \to d$ linear has $c \, n_h \, d$ params. The grouped version has $g \cdot \tfrac{c\, n_h}{g} \cdot d_g + (d_g\, g) \cdot d$. Plug in V4-Pro numbers ($c=512,\ n_h=128,\ g=16,\ d_g=1024,\ d=7168$) — how much cheaper? *(This is the whole point of the trick.)*
2. The attention-sink "extra denominator term" is equivalent to concatenating one key whose logit is $z'_h$ and whose value is $0$. Convince yourself the two formulations give the same softmax output, then pick whichever is cleaner to cache.

---

## Phase 4 — CSA path inside the *same* `Attention` class (prefill)

**Goal**: enable the `compression_rate == compress_rate` branch of the Phase-3 class — `compress (m, overlapping) → compress indexer keys → lightning-indexer top-k over blocks → MQA over selected blocks + sliding window → grouped output projection`. **No new class.**

CSA shares ~everything with HCA. The deltas (gated on `ratio == compress_rate`):
1. `TokenCompressor(..., overlapping=True)` with rate $m$ (small).
2. **The real work — indexer over compressed blocks.** ⚠️ The current notebook `Indexer` scores **token→token** over a per-token `index_k_cache`. CSA needs it to score **query→compressed-block**: mirror official `Indexer`, which owns its *own* `Compressor(args, ratio, head_dim, rotate=True)` and compresses its keys the same way before scoring (Eq 13–17). So either give the `Indexer` an internal compressor or feed it compressed keys — this is the meatiest sub-task, not a copy-paste of `dsa.ipynb`.
3. $\operatorname{Top\text{-}k}$ over blocks with the causal constraint $s < \lfloor t/m \rfloor$; replace the HCA path's `get_compress_topk_idxs` with these top-k indices.
4. Scatter the top-k block mask onto the MQA scores before softmax (mirror MLA's Phase-3 sparse-mask scatter); stash `last_indexer_scores` for the KL.

### Reuse map

| CSA piece | Source |
|---|---|
| compressor | Phase 2 `TokenCompressor(overlapping=True)` |
| MQA core, grouped proj, sink, partial-RoPE, sliding window | Phase 3 `HCA` (factor these into a shared mixin/base class) |
| low-rank queries $q^I$, gate $w^I$, ReLU sum, top-k mask | `dsa.ipynb` `Indexer` (cell `d48e89b0`) + MLA's sparse-mask scatter |

### Sub-steps

- [x] 4.1 Generalize the compressor (Bug E fix): the class already handles both rates in one body; `overlapping = (compression_rate == config.compress_rate)`. **No base class / subclassing** — one `Attention` switched by `compression_rate`. ✅
- [x] 4.2 **Indexer block-key mode** (2026-06-11). `Indexer(config, compress_rate=int)` owns its **own** `TokenCompressor(..., overlapping=(rate==compress_rate), head_dim=index_head_dim)` → block keys `K^IComp ∈ (B, nb, index_head_dim)`. `compress_rate=None` keeps the V3.2 per-token path **byte-identical** (verified). Added `head_dim` arg to `TokenCompressor`. Smoke-tested both modes. ✅
- [x] 4.3a Low-rank indexer queries + ReLU-gated per-head sum → `I ∈ (B, S, nb)` (Eq 13–16); q RoPE'd at token positions, block keys at block-start positions. ✅ *(`I` returned raw; caller masks.)*
- [~] 4.3b **Share the query latent** (Eq 14): `qr` is threaded into `indexer(..., qr=qr)`, but with `q_lora_rank=0`/`share_q_lora_with_indexer=False` the indexer still projects from `hidden`. True sharing needs `q_lora_rank>0` + the flag — left as a config switch, not blocking.
- [x] 4.4 **Wired into `Attention`** (2026-06-11): `self.is_csa` branch builds `I`, masks to visible blocks (`comp_idx>=0`), and in `sparse` keeps `I.topk(block_topk)` (drops the rest to `-1`); `dense_warmup` keeps all visible blocks. Audit fixed 6 issues (missing `self.config`, `I + mask` shape, `int += -inf` index/logit mix, logits-vs-probs stash, block-granularity stash, bare-global `config`) + renamed `*_mask`→`*_idx`, guarded indexer creation on stage. ✅
- [x] 4.5 Stash `last_indexer_scores` (`I`, `(B,S,nb)`) and `last_attn_probs` (**post-softmax** block probs `probs[..., -nb:]`, `(B,S,nh,nb)`) — aligned for `compute_indexer_kl`. ✅
- [ ] 4.6 **(Phase-5 follow-up)** `LLM.forward` KL accumulation must pass `index_block_topk` (not `index_topk`) for CSA layers and guard on `last_indexer_scores is not None`.

### Stop-and-think

1. In DSA the indexer scored token→token; here it scores token→**block**. What changes in the KL target $p$? *(Hint: aggregate the dense MQA block-probabilities the same way you aggregated head-probabilities before.)*
2. CSA's queries are *shared* with the indexer queries (both from $c^Q_t$). In your `dsa.ipynb` you already thread `qr` into the indexer — does the same latent feed both here? Confirm against Eq 14 vs Eq 18 (they share $c^Q_t$).

---

## Phase 5 — Wire into `DecoderLayer` with the interleaved schedule

**Goal**: each layer instantiates the attention type from `config.attn_schedule[layer_idx]`; `LLM.forward` still aggregates the indexer KL (only CSA layers contribute).

### Sub-steps

- [x] 5.1 `DecoderLayer.__init__(config, attn_type)` picks `MLA(config)` vs `Attention(config, rate)` by schedule string (`hca`→`hca_compress_rate`, `csa`→`compress_rate`, else→MLA). ✅
- [x] 5.2 `LLM.__init__` builds layers from `attn_schedule`. ✅
- [x] 5.3 KL accumulation guards on `attn_type == 'csa'`, passes `index_block_topk` (4.6 fixed), averages over CSA layers. ✅
- [x] 5.4 First-two-layers handled by the default schedule (HCA). ✅
- [x] 5.5 **MLA reverted to pure** — DSA/indexer removed; MLA is now the plain A/B path (its per-token `Indexer(compress_rate=None)` mode is now dead code, removable). ✅

### Re-audit (2026-06-11) — `LLM` forward+backward green (default + MLA-in-schedule across stages). Status:

- [x] 5.6 **Edge bug fixed**: KL term now gated on `if valid_layers > 0: … else: self.loss = main_loss` (no `.mean()` on int). Re-verified: `no-csa/sparse` returns finite loss instead of crashing. ✅
- [~] 5.7 *(cleanup, intentionally deferred)* The `Indexer` `compress_rate=None` legacy per-token branch is unused now that MLA dropped DSA. **Owner chose to leave it** (kept for a possible V3.2-DSA A/B).

### Stop-and-think (resolved)

The KL loop must skip non-indexer layers (HCA/MLA have `last_indexer_scores = None`); feeding `None` to `compute_indexer_kl` crashes. Solved via the per-layer `attn_type == 'csa'` flag (a value-check `last_indexer_scores is not None` is equivalent and also catches a hypothetical MLA-DSA layer).

---

## Phase 6 — Decode & KV cache (mirror official `Compressor`)

**Goal**: make the unified `Attention` correct under incremental decode (`start_pos > 0`, `S == 1`), so greedy generation with caching matches a full prefill bit-for-bit. This is the hardest piece — three caches plus the incremental-compression state machine from official `Compressor`/`Attention`.

The three caches per layer (CSA has all three; HCA has #1–2; SWA has only #1):

1. **Window cache** — uncompressed recent KV. Simplest-correct: a `(B, max_seq_len, c)` buffer, write `win_kv` at `[start_pos:end_pos]`, read `[:end_pos]`, then the decode branch of `get_window_topk_idxs` selects the last `n_win`. *(Official uses a `window_size` ring buffer — optional optimization; do the flat buffer first.)*
2. **Compressed cache + state cache** — the crux. Tokens arrive one at a time but a compressed entry only emits every `m` (`m'`) tokens. Mirror official `Compressor`:
   - Buffers `kv_state` / `score_state` of size `(1+overlap)*m` hold the uncompressed tail + its pooling logits.
   - Each decode step writes the new token's (kv, score+bias) into the state at `start_pos % m` (and the overlap slot for CSA).
   - When `(start_pos + 1) % m == 0`: pool the state → one entry, RoPE it at the block-start position, write to the compressed cache `[start_pos // m]`. For CSA, carry the previous block's `a`-half (the overlap bookkeeping).
   - Prefill (`start_pos == 0`) seeds the state with the trailing `remainder = S % m` tokens.
3. **Indexer-key cache** (CSA only) — same incremental compression with the indexer's own compressor (`rotate=True`, `index_head_dim`); the indexer then scores the single query against `index_kv_cache[:, :end_pos // m]` and returns top-k block indices.

### Sub-steps

- [x] 6.1 **Window cache** (2026-06-11): flat `win_cache (B, max_seq_len, c)`, write `win_kv` at `[start_pos:end_pos]`, read `[:end_pos]`; `_get_window_topk_idxs` decode branch returns the recent `n_win` absolute positions. ✅
- [x] 6.2 **Compressed/state cache for HCA** (2026-06-11): `TokenCompressor` now stateful — `kv_state`/`weight_state` (in-progress block), `comp_cache` (emitted entries), `_simple_step` emit on `(start_pos+1)%m==0`, prefill `seed=True` writes entries + seeds the remainder. Entries kept pre-norm/pre-RoPE (`Attention` applies on read). ✅
- [x] 6.3 **CSA overlap state** (2026-06-11): added `kv_state_b`/`weight_state_b` (current b-series) + `prev_kv_b`/`prev_weight_b` (predecessor); `_overlap_step` pools `cat([cur_a, prev_b])`, then shifts cur_b→prev_b; prefill seeds `prev_b` with the last complete block's b-series. ✅
  - **Verified**: incremental decode is **byte-exact** vs bulk (`max|diff|=0.0`) for both overlap modes across full-prefill / mid-block / boundary / ragged-length splits.
- [x] 6.4 **Indexer-key cache** (2026-06-11): the `Indexer`'s `TokenCompressor` inherits the decode path; indexer `start_pos > 0` scoring (q at token pos, keys at block-start pos) works. ✅
- [x] 6.5 **`Attention` decode wiring** (2026-06-11): threads `start_pos`/`use_kv_cache` into `token_compressor`, window cache write/read, compressed `offset = win_kv.shape[1]`, `_get_compress_topk_idxs` decode branch (all complete blocks visible), single-query `-i` RoPE. **Audit fixed 5 wiring bugs** (compressor not stepped; missing/inverted decode branches; `offset=S`; `torch.arrange` typo). ✅

### Stop-and-think

1. Why must the state cache hold *both* the kv values and their pooling logits, not just a running weighted sum? *(Hint: softmax over the block is normalized across all `m` tokens — you can't normalize until the block is full.)*
2. The lcm(m, m′) cache-block layout (Fig 6) lets CSA and HCA layers share block boundaries. For this replica you can cache each layer independently — but note where that would break shared-prefix reuse.

---

## Phase 7 — Verify & smoke-test

Mirror the gold-standard tests from `dsa_roadmap.md` Phase 5.

### Sub-steps

All of 7.1–7.6 are implemented in two notebook cells — `cache_parity` (7.3) and the **Phase 7 verification** cell (7.1, 7.2, 7.4, 7.5, 7.6) — and all pass.

- [x] 7.1 **Shapes & smoke**: finite loss + finite grads for every DSA stage (off/dense_warmup/sparse). ✅
- [x] 7.2 **Capacity**: a mla+hca+csa model overfits a fixed tiny batch (loss 8.8 → ~0.0 in 25 Adam steps). ✅
- [x] 7.3 **Cache parity** (the killer test) — `cache_parity` cell added (after `LLM`). **Bit-exact (~3e-6)** for HCA / CSA-dense / mixed / MLA schedules, confirming the window + compressed/state + indexer-key + MLA caches. ✅
  - ⚠️ **CSA-`sparse` is *not* bit-exact (~8e-2)** — and this is **expected, not a cache bug**: the indexer's `ReLU` saturates several block scores to exactly `0.0` at random init, so the top-k boundary is a tie and `torch.topk` breaks it differently for prefill (n in-array) vs decode (n+1). DeepSeek avoids this in production with batch-invariant deterministic kernels (paper §3.3). *Optional fix for an exact replica: add a deterministic index tie-break before `topk`, e.g. `I_vis - eps * arange(nb)`.*
- [x] 7.4 **Causality** (black-box): perturbing tokens after position `P` leaves `logits[:, :P]` unchanged (off + sparse), max|diff| < 1e-4. ✅
- [x] 7.5 **CSA sparse vs dense** (same weights): flipping `dsa_stage` sparse↔dense_warmup on one model changes the output (max|diff| ≈ 0.67), confirming top-k actually masks. ✅
- [x] 7.6 **Warmup freeze**: `freeze_for_dsa_warmup` leaves only the ~10 indexer params trainable, and only they receive gradient after a `dense_warmup` backward. ✅

### Why cache parity is the killer test

You have *three* caches per CSA layer (compressed entries, indexer keys, sliding-window state) and the `-i` output RoPE depends on absolute position. Any `start_pos` mistake silently diverges decode from prefill. This single test catches all of them.

---

## Phase 8 — Muon optimizer (self-contained; do anytime)

**Goal**: replace AdamW (for matrix params) with Muon, matching the paper's Algorithm 1. The paper's Muon = [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight)'s variant: Nesterov momentum → Newton-Schulz orthogonalization → RMS rescale → decoupled weight decay.

### Algorithm 1 in equations

Per step $t$, for each 2-D weight $W \in \mathbb{R}^{n \times m}$ with gradient $G_t$, momentum $\mu$, learning rate $\eta$, weight decay $\lambda$, RMS-rescale factor $\gamma$:

$$M_t = \mu M_{t-1} + G_t \qquad\qquad O'_t = \operatorname{NS}\!\big(\mu M_t + G_t\big) \quad\text{(Nesterov + orthogonalize)}$$

$$O_t = O'_t \cdot \sqrt{\max(n,m)}\,\gamma \quad\text{(RMS rescale)} \qquad\qquad W_t = W_{t-1}\,(1 - \eta\lambda) - \eta\, O_t \quad\text{(decoupled decay + update)}$$

The orthogonalizer $\operatorname{NS}(\cdot)$ normalizes $X_0 = M/\lVert M\rVert_F$ then iterates the quintic (Eq 28):

$$X_k = a\, X_{k-1} + b\,(X_{k-1} X_{k-1}^\top) X_{k-1} + c\,(X_{k-1} X_{k-1}^\top)^2 X_{k-1}$$

Paper's **hybrid** schedule: 10 iterations total — first 8 with $(a,b,c) = (3.4445,\, -4.7750,\, 2.0315)$ (fast convergence), last 2 with $(a,b,c) = (2,\, -1.5,\, 0.5)$ (settle singular values at $1$).

### What stays on AdamW (paper §2.4 "Basic Configurations")

Embedding, prediction head (`output`), all RMSNorm weights, and (for V4) mHC static biases/gates. **Everything else (the 2-D matrices) uses Muon.** Muon only makes sense for `ndim == 2` params.

### Verified reference core (from Moonlight)

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)       # paper's first-stage coeffs
    X = G.bfloat16()
    if G.size(0) > G.size(1): X = X.T
    X = X / (X.norm() + 1e-7)                  # M0 = M / ||M||_F
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X                       # Eq 28
    if G.size(0) > G.size(1): X = X.T
    return X

# in the optimizer step, per 2-D param p with grad g:
buf.mul_(momentum).add_(g)                      # momentum buffer
g = g.add(buf, alpha=momentum)                 # Nesterov
u = zeropower_via_newtonschulz5(g, steps=5)    # orthogonalize
adjusted_lr = lr * 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))  # RMS rescale (== sqrt(max(n,m))·γ, γ=0.2)
p.data.mul_(1 - lr * wd)                        # decoupled weight decay
p.data.add_(u, alpha=-adjusted_lr)             # update
```

### Paper-specific delta (optional, advanced)

The paper uses a **hybrid Newton-Schulz**: 10 steps total — first 8 with `(3.4445, -4.7750, 2.0315)`, last 2 with `(2, -1.5, 0.5)` to settle singular values exactly at 1. The Moonlight reference uses 5 steps single-stage. For a replica, single-stage 5 steps is fine; implement the hybrid only if you want bit-fidelity to Algorithm 1.

### Sub-steps

- [ ] 8.1 Drop in the `Muon` class (Moonlight `toy_train.py` is the cleanest single-file source).
- [ ] 8.2 Param routing: 2-D matrices → Muon; embeddings/head/norms → AdamW (mirror your `DSATrainer.create_optimizer`, cell `edb1b2d5`, but split on `ndim`/name instead of `indexer`).
- [ ] 8.3 Confirm the RMS-rescale lets you reuse your AdamW LR (paper sets RMS target 0.18; Moonlight uses 0.2 — pick one and note it).
- [ ] 8.4 Train a few hundred steps; confirm loss matches/beats AdamW and is stable (the whole point: faster convergence + stability, no QK-Clip needed because of Q/KV norm).

### Stop-and-think

1. Why does Muon need the **full** gradient matrix (and thus fights ZeRO sharding)? *(Hint: Newton-Schulz is a matrix operation on the whole 2-D grad; you can't orthogonalize a row-shard in isolation.)* The paper's §3.5.1 hybrid-ZeRO is the production answer — skip for a single-GPU replica.
2. The paper drops QK-Clip "because the attention architecture allows direct RMSNorm on Q and KV entries." Which line in your HCA/CSA forward is doing that job? *(The Phase-3 Q/KV norm — §2.3.3.)*

---

## References

- **Paper**: DeepSeek-V4, §2.3 (Hybrid Attention: CSA + HCA), §2.4 (Muon), §3.6 (KV cache layout).
- **NSA** (compression/selection/sliding-window pattern): [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch), paper [2502.11089](https://huggingface.co/papers/2502.11089).
- **Muon**: [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) (the cited variant), original [Muon writeup / KellerJordan](https://github.com/KellerJordan/Muon).
- **Your own DSA**: `dsa.ipynb` + `dsa_roadmap.md` — the `Indexer`, KL warmup, and cache discipline transfer directly.

## Open questions / deliberate scope cuts

- **State cache & on-disk KV** (§3.6): inference-time optimizations. Skip; a single in-memory compressed cache is enough for a replica.
- **FP8/FP4** for indexer QK and KV storage (§2.3.4, §3.4): skip; bf16/fp32 is fine.
- **Mask-based vs gather-based sparse path**: same trade-off as DSA Phase 3 — start mask-based, optimize later.
- **Exact $\operatorname{lcm}(m, m')$ cache-block layout** (Figure 6): implement the simplest per-type cache first; align blocks only if you tackle shared-prefix reuse.
- **Hybrid Newton-Schulz (8+2 steps)**: optional fidelity detail; single-stage 5 steps trains fine.
- **mHC**: the other big V4 piece (manifold-constrained hyper-connections, §2.2). Deserves its own roadmap — it touches the residual stream and the optimizer (static biases/gates stay on AdamW), not the attention modules.


## My questions
1. how each layer's kv cache is saved and ensured only be loaded for that certain layer?