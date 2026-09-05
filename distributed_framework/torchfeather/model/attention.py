import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

__all__ = [
    "ScaledDotProductAttentionWrapper",
    "local_head_count",
]


class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    Two things this buys over an explicit `matmul` / mask-add / `softmax` chain:

    1. **Memory and speed.** The flash and memory-efficient backends never materialize the
       ``(B, H, S, T)`` score matrix. That matrix is what caps sequence length here --
       537 MiB per layer at ``B=8, S=2048, H=8`` in bf16, and four times that at ``S=4096``.
    2. **Context parallelism.** `torch.distributed.tensor.experimental.context_parallel`
       works by swapping out the global `F.scaled_dot_product_attention` for a ring-attention
       implementation. Score math written as raw `matmul` is invisible to that swap, so CP
       silently trains on unsharded attention. The call below must therefore go through the
       module-level function, not a captured reference.

    The flash backend requires ``Dq == Dk == Dv``. MLA does not satisfy that -- the naive path
    is 96/96/64 -- so `forward` zero-pads V up to the query head dim and slices the result
    back. The padded columns see all-zero values, hence produce all-zero outputs, so the slice
    is exact rather than approximate.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        is_causal: bool = True,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over ``(B, H, S, D)`` tensors and return ``(B, H, S, Dv)``.

        Args:
            q (torch.Tensor): Queries, ``(B, H, S, Dqk)``.
            k (torch.Tensor): Keys, ``(B, H, T, Dqk)``.
            v (torch.Tensor): Values, ``(B, H, T, Dv)``. May differ from ``Dqk``.
            scale (float | None): Softmax scale. Defaults to ``Dqk ** -0.5``.
            is_causal (bool): Apply a causal mask. Mutually exclusive with `attn_mask`;
                context parallelism only shards correctly on this path.
            attn_mask (torch.Tensor | None): Explicit additive or boolean mask, for the
                KV-cache decode case where `is_causal` would mask out the cached prefix.

        Returns:
            torch.Tensor: Attention output, ``(B, H, S, Dv)``.
        """
        # RoPE is computed in fp32 by design, so q and k come back wider than v, which
        # carries the parameter dtype. SDPA requires all three to agree. Narrow q/k to v
        # rather than promoting v: flash only accepts fp16/bf16, so promoting would
        # silently drop us onto the math backend and re-materialize the score matrix.
        # Under torch.autocast this never showed up -- autocast casts SDPA's inputs for
        # us -- which is why it survived the single-GPU stage and only failed under FSDP,
        # where mixed precision comes from fully_shard and there is no autocast context.
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        head_dim_qk, head_dim_v = q.shape[-1], v.shape[-1]

        if head_dim_v < head_dim_qk:
            # Differentiable, and the pad columns carry no gradient back into v.
            v = F.pad(v, (0, head_dim_qk - head_dim_v))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
        )

        if head_dim_v < head_dim_qk:
            out = out[..., :head_dim_v]
        return out


def local_head_count(x: torch.Tensor, head_dim_index: int = 2) -> int:
    """Head count actually held on this rank, read off the shard rather than the mesh.

    Under TP the head dim is sharded, so `n_heads` from the model args is the global count and
    using it to `expand` a head-shared tensor produces a shape that no longer lines up with the
    sharded one it is about to be concatenated with.
    """
    if isinstance(x, DTensor):
        return x.to_local().shape[head_dim_index]
    return x.shape[head_dim_index]
