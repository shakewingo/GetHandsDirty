import math
from typing import Literal
from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F

from torchfeather.model.attention import (
    ScaledDotProductAttentionWrapper,
)
from torchfeather.model.model_args import DeepSeekV3ModelArgs
# from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import RotaryEmbedding
from torch.distributed.tensor import DTensor, Replicate, Shard


def _is_last_dim_shard(placement, ndim: int) -> bool:
    """Whether `placement` shards the trailing dim of an `ndim`-dimensional tensor.

    `redistribute` normalizes a negative shard dim to its positive index, while
    `from_local` keeps whatever it was given, so both spellings show up in practice.
    """
    return isinstance(placement, Shard) and placement.dim in (ndim - 1, -1)


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """Merge the (n_heads, head_dim) dims of a (B, S, n_h, d_h) tensor.

    Under TP, x is a DTensor sharded on dim 2 (heads). Eager DTensor disallows
    flatten/reshape across a sharded dim, but the merge is data-movement-free:
    each rank owns a contiguous block of heads, so its local (n_h/tp) x d_h
    chunk is exactly its slice of the merged dim, which lands on dim 2 as well.
    Reinterpret the placement directly instead of redistributing."""
    if isinstance(x, DTensor):
        local = x.to_local().flatten(2)
        placements = tuple(
            Shard(2) if isinstance(p, Shard) and p.dim == 2 else p for p in x.placements
        )
        return DTensor.from_local(local, x.device_mesh, placements, run_check=False)
    return x.flatten(2)


def _split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Split the last dim of a (B, S, n_h * d_h) tensor into (B, S, n_h, d_h).

    Under TP, x is a DTensor sharded on the last dim (the merged n_h*d_h),
    which eager DTensor cannot .view() across -- but the split is purely local
    (head boundaries never cross the last-dim shard), so do it on the local
    tensor and re-wrap with the trailing shard moved onto the new head dim.

    The local head count is read off the shard we actually hold rather than
    derived from the mesh size, so this stays correct for a replicated input and
    does not assume the mesh is exactly the TP dim."""
    if isinstance(x, DTensor):
        local = x.to_local()
        local_heads = local.shape[-1] // head_dim
        out = local.view(*local.shape[:-1], local_heads, head_dim)
        placements = tuple(
            Shard(2) if _is_last_dim_shard(p, x.ndim) else p for p in x.placements
        )
        return DTensor.from_local(out, x.device_mesh, placements, run_check=False)
    return x.view(*x.shape[:-1], n_heads, head_dim)


class DeepSeekV3Model(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(DecoderLayer(args))

        self.layernorm = nn.RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # A meta-device build has no storage to write into; the trainer calls
        # `init_weights` again once `to_empty()` has materialized the parameters.
        if not self.token_embeddings.weight.is_meta:
            self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize every parameter and re-materialize every buffer.

        Args:
            buffer_device (torch.device | None): Device for buffers rebuilt from
                scratch (the RoPE tables). Defaults to their current device.

        Must cover buffers as well as parameters: the trainer builds on the meta
        device and calls `to_empty()` first, which leaves every buffer holding
        uninitialized memory.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # weights variance may boost D times for linear projection
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.dim))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.dim))
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('down_proj.weight'):
                # weights variance may boost 2L times after L layers transformer blocks (each block contains both attention and FFN so it boots 2 times)
                # so we scale down the final output weights in transformer block
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                module.init_weights(buffer_device)
            elif isinstance(module, Attention):
                module.init_kv_cache()

    def forward(self, x, use_kv_cache=False, start_pos=0, last_token_only=False):
        # Non-first PP stages receive (B, S, D) hidden states; the first stage gets (B, S) token ids.
        B, S = x.shape[0], x.shape[1]
        end_pos = start_pos + S  # to track the entire seq length

        # With pipeline parallelism, non-first stages have token_embeddings=None
        # and receive hidden states from the previous stage instead.
        if self.token_embeddings is not None:
            x = self.token_embeddings(x)

        if S == 1 and start_pos > 0:
            # Decode step: the single new query at position start_pos can attend to
            # every cached key 0..start_pos — no positions to mask out.
            mask = None
        else:
            T_k = end_pos if use_kv_cache else S
            row = torch.arange(S, device=x.device).unsqueeze(1)        # (S, 1)
            col = torch.arange(T_k, device=x.device).unsqueeze(0)      # (1, T_k)
            mask = torch.where(col <= start_pos + row, 0.0, float('-inf'))
            if isinstance(x, DTensor):
                # Under TP, scores is head-sharded; broadcasting a sharded mask into it
                # is disallowed, but the causal mask is a pure function of position
                # (identical on every rank), so keep it replicated.
                mask = DTensor.from_local(
                    mask, x.device_mesh, (Replicate(),), run_check=False
                )

        for layer in self.layers:
            x = layer(x, mask=mask, use_kv_cache=use_kv_cache, start_pos=start_pos)

        if self.layernorm is not None:
            x = self.layernorm(x)

        if self.output is None:
            # Pipeline-parallel non-final stage: pass hidden states to the next stage.
            return x

        # `last_token_only` keeps a decode step from projecting the whole prefill
        # window through the vocab matrix; training always wants every position.
        if last_token_only:
            x = x[:, [-1], :]
        return self.output(x)

    def absorb_weights(self) -> None:
        """Fuse MLA's W^UK/W^UV into every layer's attention, once, before running `generate`."""
        if self.args.attn_impl == 'naive':
            return  # nothing to absorb: 'naive' never compresses K/V in the first place
        for layer in self.layers:
            layer.attention.absorb_weights()

    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                    use_kv_cache=True):
        self.absorb_weights()  # no-op after the first call (per-layer `_absorbed` guard) or if attn_impl == 'naive'

        input_ids = inputs['input_ids']
        s = input_ids.shape[1]
        # start_pos = how many tokens are already represented in the KV cache.
        # First iteration: 0  -> forward the entire prompt (the "prefill" pass).
        # Subsequent iterations: forward only the single new token at position start_pos.
        # Without cache: start_pos stays 0 and we forward the full growing sequence each step
        # (quadratic-cost behavior, kept as a comparison baseline).
        start_pos = 0
        while input_ids.shape[1] - s < max_new_tokens - 1:
            tokens_in = input_ids[:, start_pos:] if use_kv_cache else input_ids
            logits = self.forward(tokens_in, use_kv_cache=use_kv_cache,
                                     start_pos=start_pos, last_token_only=True)
            logits = logits[:, -1, :]

            # apply penalty for repetitive tokens, per batch row
            for b in range(input_ids.shape[0]):
                for token in set(input_ids[b].tolist()):
                    logits[b, token] /= repetition_penalty

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            if (idx_next == eos).all():
                break

            # IMPORTANT: advance start_pos BEFORE we cat the new token onto input_ids,
            # so the next iteration's `input_ids[:, start_pos:]` is just the new token.
            if use_kv_cache:
                start_pos = input_ids.shape[1]

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if stream:
                yield input_ids[:, s:]

        if not stream:
            yield input_ids[:, s:] 

class FeedForward(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.args = args
        self.inter_dim = args.inter_dim
        self.dim = args.dim
        self.gate_proj = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.dim, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Attention (nn.Module):
    """MLA attention"""
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        
        self.q_lora_rank = args.q_lora_rank  # i.e. d_c'
        self.kv_lora_rank = args.kv_lora_rank # i.e. d_c
        self.qk_nope_head_dim = args.qk_nope_head_dim  # i.e. d_h
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim # d_h + d^R_h
        self.v_head_dim = args.v_head_dim # d_h

        # down and up projection for mla
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False) # down prj for hidden size, d->d_c+d^R_h; merge W^DKV and W^KR in one linear projection for easier computation
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)  # up prj for hidden size, d_c->2*n_h*d_h; merge W^UK and W^UV in one linear projection for easier computation

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False) # down prj for hidden size, d->d_c'
            self.q_norm = nn.RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False) # up prj for hidden size, d_c'->n_h*(d_h+d^R_h)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False) # n_h*d_h->d
        self.rotary_emb = RotaryEmbedding(args)

        # YaRN handling
        self.softmax_scale = self.qk_head_dim**-0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = ScaledDotProductAttentionWrapper() # TODO: may need (B, H, S, D) as input shape

        if self.args.attn_impl == 'naive':
            self.register_buffer('k_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer('v_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)

        else:
            # what actually gets cached in MLA
            self.register_buffer('kv_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer('pe_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

        # True once `absorb_weights()` has fused W^UK/W^UV into wq*/wo for frozen-weight inference.
        self._absorbed = False

    def init_kv_cache(self) -> None:
        """Zero the KV cache buffers, which `to_empty()` leaves uninitialized."""
        for name in ('k_cache', 'v_cache', 'kv_cache', 'pe_cache'):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.zero_()

    @torch.no_grad()
    def absorb_weights(self) -> None:
        """Fuse W^UK into the q projection and W^UV into wo, once, for frozen-weight inference.

        Mathematically identical to the per-call `einsum` absorption already used in `forward`
        when `attn_impl != 'naive'`; this only moves that (weight x weight) fusion out of the hot
        loop so repeated decode steps no longer redo it on every call. Only valid for frozen
        weights (do not use mid-training) — re-run after any further fine-tuning.
        """
        if self.args.attn_impl == 'naive':
            raise ValueError("weight absorption only applies to the non-naive (compressed) attn_impl")
        if self._absorbed:
            return  # idempotent: re-running would just recompute the same fused weights from wq/wkv_b/wo

        wkv_b_wght = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)  # (2 * n_h * d_h, d_c) -> (n_h, 2 * d_h, d_c)
        w_uk_wght = wkv_b_wght[:, :self.qk_nope_head_dim]   # (n_h, d_h, d_c) == W^UK
        w_uv_wght = wkv_b_wght[:, -self.v_head_dim:]        # (n_h, v_h, d_c) == W^UV

        # --- fuse W^UK into whichever projection currently produces q_nope ---
        q_proj = self.wq if self.q_lora_rank == 0 else self.wq_b
        in_dim = q_proj.in_features # d_c'
        w_q_wght = q_proj.weight.view(self.n_heads, self.qk_head_dim, -1) # (n_h*(d_h+d^R_h), d_c') -> (n_h, d_h+d^R_h, d_c')
        w_q_nope_wght, w_q_rope_wght = torch.split(w_q_wght, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=1)
        w_q_nope_abs = torch.bmm(w_uk_wght.transpose(1, 2), w_q_nope_wght)  # (n_h, d_c, d_c')
        w_q_abs = torch.cat([w_q_nope_abs, w_q_rope_wght], dim=1).reshape(
            self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim), in_dim
        ) # (n_h*(d_c+d^R_h), d_c')

        b_abs = None
        if q_proj.bias is not None:
            b = q_proj.bias.view(self.n_heads, self.qk_head_dim)
            b_nope, b_rope = torch.split(b, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=1)
            b_nope_abs = torch.bmm(b_nope.unsqueeze(1), w_uk_wght).squeeze(1)  # (n_h, d_c)
            b_abs = torch.cat([b_nope_abs, b_rope], dim=1).reshape(-1)

        fused_q = nn.Linear(in_dim, self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim),
                             bias=b_abs is not None, device=w_q_wght.device, dtype=w_q_wght.dtype)
        fused_q.weight.copy_(w_q_abs)
        if b_abs is not None:
            fused_q.bias.copy_(b_abs)
        fused_q.requires_grad_(False)
        if self.q_lora_rank == 0:
            self.wq_abs = fused_q
        else:
            self.wq_b_abs = fused_q

        # --- fuse W^UV into wo ---
        wo_wght = self.wo.weight.view(self.dim, self.n_heads, self.v_head_dim).permute(1, 0, 2)  # (n_h, dim, d_h)
        wo_abs = torch.bmm(wo_wght, w_uv_wght).permute(1, 0, 2).reshape(self.dim, self.n_heads * self.kv_lora_rank)

        fused_o = nn.Linear(self.n_heads * self.kv_lora_rank, self.dim,
                             bias=self.wo.bias is not None, device=wo_wght.device, dtype=wo_wght.dtype)
        fused_o.weight.copy_(wo_abs)
        if self.wo.bias is not None:
            fused_o.bias.copy_(self.wo.bias)  # wo's bias is added after decompression either way, so it's unchanged
        fused_o.requires_grad_(False)
        self.wo_abs = fused_o

        self._absorbed = True

    def forward(self, x, mask=None, use_kv_cache=False, start_pos=0):
        # `start_pos` is the absolute position of the FIRST query token in this batch.
        # Prefill: start_pos=0, S = prompt_len. (S == end_pos)
        # Decode step: start_pos = past_len, S = 1 (just the newly generated token). (end_pos = start_pos + 1)
        # Training: use_kv_cache=False, start_pos=0 — math identical to before this refactor.
        # NOTE on q/k length: q's length is always S (current batch). k's length is S (no cache)
        # or end_pos (with cache, reads full prefix). The einsum letters `s` (for q) and `t` (for k)
        # below encode exactly this asymmetry.
        B, S, d = x.shape
        end_pos = start_pos + S

        kv = self.wkv_a(x) # (B, S, d_c + d^R_h)
        kv_nope, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        if self._absorbed:
            # W^UK already fused into wq*_abs, so q_nope comes out pre-absorbed (dim d_c, not d_h).
            if self.q_lora_rank == 0:
                q = self.wq_abs(x)
            else:
                q = self.wq_a(x) # (B, S, d_c')
                q = self.q_norm(q) # (B, S, d_c')
                q = self.wq_b_abs(q) # (B, S, n_h*(d_c+d^R_h))
            q = _split_heads(q, self.n_heads, self.kv_lora_rank + self.qk_rope_head_dim)
            q_nope, q_pe = torch.split(q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        else:
            if self.q_lora_rank == 0:
                q = self.wq(x)
            else:
                q = self.wq_a(x) # (B, S, d_c') 
                q = self.q_norm(q) # (B, S, d_c')
                q = self.wq_b(q) # (B, S, n_h*(d_h+d^R_h))
            q = _split_heads(q, self.n_heads, self.qk_head_dim) # (B, S, n_h, d_h+d^R_h)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        k_pe = k_pe.unsqueeze(2) # k_pe shape:(B, S, 1, d^R_h)
        q_pe = self.rotary_emb.apply_rotary_emb(q_pe, start_pos=start_pos)
        k_pe = self.rotary_emb.apply_rotary_emb(k_pe, start_pos=start_pos)
        if self.args.attn_impl == 'naive':
            q = torch.cat([q_nope, q_pe], dim=-1) # (B, S, n_h, d_h+d^R_h)
            kv_nope = self.kv_norm(kv_nope)
            kv_nope = self.wkv_b(kv_nope) # (B, S, 2*n_h*d_h))
            kv_nope = _split_heads(kv_nope, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v_new = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k_new = torch.cat([k_nope, k_pe.expand(-1,-1,self.n_heads,-1)], dim=-1) # (B, S, n_h, d_h+d^R_h)

            if use_kv_cache:
                self.k_cache[:B, start_pos:end_pos, :, :] = k_new
                self.v_cache[:B, start_pos:end_pos, :, :] = v_new
                k = self.k_cache[:B, :end_pos]  # (B, end_pos, n_h, qk_head_dim) -- full prefix incl. k_new
                v = self.v_cache[:B, :end_pos]  # (B, end_pos, n_h, v_head_dim)
            else:
                k, v = k_new, v_new              # (B, S, n_h, qk_head_dim) / (B, S, n_h, v_head_dim)

            # NOTE: einsum is avoided under TP -- aten.einsum's sharding rule flattens
            # batch dims including the head-sharded dim; matmul propagates cleanly.
            # (B,S,H,Dq) @ (B,H,Dk,T) -> (B,H,S,T) -> (B,S,H,T)
            scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)).permute(0, 2, 1, 3) / math.sqrt(self.qk_head_dim)
        else:
            # consider weights absortion to avoid calculating k distinctly, i.e. via changing multiply order i.e. A*(B*C) -> (A*B) * C to reduce computation cost
            if not self._absorbed:
                wkv_b = self.wkv_b.weight # if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) , (d_h*n_h, d_c)
                # wkv_b is head-sharded on dim 0 (merged n_h*(d_h+v_h)) under TP, and
                # every use below is a per-head contraction against a head-sharded
                # activation, so stay on the local shard throughout. The local head
                # count comes from the shard itself rather than the mesh size, which is
                # only the TP degree while the plan is applied on the 1-D `tp` mesh.
                if isinstance(wkv_b, DTensor):
                    wkv_b = wkv_b.to_local()
                local_heads = wkv_b.shape[0] // (self.qk_nope_head_dim + self.v_head_dim)
                wkv_b = wkv_b.view(local_heads, -1, self.kv_lora_rank) # (n_h, d_h + v_h, d_c)
                # q_{nope} = q_{nope} \times W^{UK}
                # Per-head contraction with a head-sharded weight; DTensor matmul can't
                # handle a sharded *batch* dim, so run it on the local tensors and
                # re-wrap with the head sharding it came in with.
                # (B,S,H,d_h) x (H,d_h,d_c) -> (B,S,H,d_c)
                if isinstance(q_nope, DTensor):
                    qn_local = torch.einsum("bshd,hdc->bshc", q_nope.to_local(), wkv_b[:, :self.qk_nope_head_dim])
                    q_nope = DTensor.from_local(qn_local, q_nope.device_mesh, q_nope.placements, run_check=False)
                else:
                    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # else: q_nope was already absorbed via wq_abs/wq_b_abs above, via absorb_weights().
            kv_nope_new = self.kv_norm(kv_nope)
            k_pe_2d_new = k_pe.squeeze(2)

            if use_kv_cache:
                self.kv_cache[:B, start_pos:end_pos] = kv_nope_new
                self.pe_cache[:B, start_pos:end_pos] = k_pe_2d_new
                kv_nope = self.kv_cache[:B, :end_pos]  # (B, end_pos, kv_lora_rank)
                k_pe_2d = self.pe_cache[:B, :end_pos]  # (B, end_pos, qk_rope_head_dim)
            else:
                kv_nope = kv_nope_new                  # (B, S, kv_lora_rank)
                k_pe_2d = k_pe_2d_new                  # (B, S, qk_rope_head_dim)

            # Under TP, run the attention core on local tensors: DTensor matmul can't
            # propagate through the head-sharded batch dims, and the core is per-rank
            # anyway (each rank owns a slice of heads). q_nope/q_pe are Shard(2) (heads);
            # kv_nope/k_pe_2d are head-shared, so their local tensors are already full.
            tp_sharded = isinstance(q_nope, DTensor)
            head_mesh = q_nope.device_mesh if tp_sharded else None
            if tp_sharded:
                q_nope_l = q_nope.to_local()
                q_pe_l = q_pe.to_local()
                kv_nope_l = kv_nope.to_local() if isinstance(kv_nope, DTensor) else kv_nope
                k_pe_2d_l = k_pe_2d.to_local() if isinstance(k_pe_2d, DTensor) else k_pe_2d
            else:
                q_nope_l, q_pe_l, kv_nope_l, k_pe_2d_l = q_nope, q_pe, kv_nope, k_pe_2d

            # (B,S,H,d) -> (B,H,S,d) @ (B,d,T) -> (B,H,S,T) -> (B,S,H,T); same for the rope part
            scores = (torch.matmul(q_nope_l.permute(0, 2, 1, 3), kv_nope_l.transpose(1, 2).unsqueeze(1)).permute(0, 2, 1, 3) +
                        torch.matmul(q_pe_l.permute(0, 2, 1, 3), k_pe_2d_l.transpose(1, 2).unsqueeze(1)).permute(0, 2, 1, 3)) / math.sqrt(self.qk_head_dim)
        if mask is not None:
            # scores is a local tensor on the absorb path (under TP); the naive path
            # keeps DTensor scores, so only wrap the mask there.
            if isinstance(scores, DTensor) and not isinstance(mask, DTensor):
                mask = DTensor.from_local(
                    mask, scores.device_mesh, (Replicate(),), run_check=False
                )
            scores += mask.unsqueeze(1) 
        scores = scores.softmax(dim=-1)

        if self.args.attn_impl == 'naive':
            # (B,S,H,T)->(B,H,S,T) @ (B,T,H,d_h)->(B,H,T,d_h) -> (B,H,S,d_h) -> (B,S,H,d_h)
            x = torch.matmul(scores.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            x = self.wo(_flatten_heads(x))
        elif self._absorbed:
            # `kv_nope` is the cache slice during decode and the live latent during training.
            # (B,S,H,T)->(B,H,S,T) @ (B,1,T,d_c) -> (B,H,S,d_c) -> (B,S,H,d_c)
            x = torch.matmul(scores.permute(0, 2, 1, 3), kv_nope_l.unsqueeze(1)).permute(0, 2, 1, 3)
            if tp_sharded:
                x = DTensor.from_local(x, head_mesh, (Shard(2),), run_check=False)
            x = self.wo_abs(_flatten_heads(x))  # W^UV already fused in here, no decompression step needed
        else:
            # `kv_nope` is the cache slice during decode and the live latent during training.
            # Either way it carries the right K-side rows for the weighted sum.
            x = torch.matmul(scores.permute(0, 2, 1, 3), kv_nope_l.unsqueeze(1)).permute(0, 2, 1, 3)
            # Per-head W^UV decompression: local einsum on the head slice.
            # (B,S,H,d_c) x (H,d_h,d_c) -> (B,S,H,d_h)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
            if tp_sharded:
                x = DTensor.from_local(x, head_mesh, (Shard(2),), run_check=False)
            x = self.wo(_flatten_heads(x))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        # Dense-only for now; MoE support lands later. Downstream parallelism
        # code (TP/FSDP/compile) checks this flag to pick plans.
        self.moe_enabled = False

        self.input_layernorm = nn.RMSNorm(self.dim)     
        self.attention = Attention(args)
        self.post_attention_layernorm = nn.RMSNorm(self.dim)
        self.ffn = FeedForward(args)

    def forward(self, x, mask=None, use_kv_cache=False, start_pos=0):
        residual = x
        x = self.input_layernorm(x)
        # NOTE: mask is passed positionally so the TP PrepareModuleInput hook on
        # `attention` (input_layouts=(Shard(1), None)) sees the declared arg count.
        x = self.attention(x, mask, use_kv_cache=use_kv_cache, start_pos=start_pos)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.ffn(x)
        x = residual + x
        return x



if __name__ == "__main__":
    args = DeepSeekV3ModelArgs()
    model = DeepSeekV3Model(args)
    B, S = 2, 100
    input_ids = torch.randint(0, args.vocab_size, (B, S))
    labels = torch.randint(0, args.vocab_size, (B, S))
    # for pn, p in model.named_parameters():
    #     print(pn, p.shape)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1))
    logger.info(f"Output shape: {logits.shape}, loss: {loss}")