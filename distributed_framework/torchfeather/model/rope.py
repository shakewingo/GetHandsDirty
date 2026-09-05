import math
import random
import numpy as np
from loguru import logger
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torchfeather.model.model_args import DeepSeekV3ModelArgs

class RotaryEmbedding(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super(RotaryEmbedding, self).__init__()
        self.dim = args.qk_rope_head_dim
        self.seqlen = args.max_seq_len
        self.beta_fast = args.beta_fast
        self.beta_slow = args.beta_slow
        self.base = args.rope_theta
        self.factor = args.rope_factor
        self.original_seq_len = args.original_seq_len

        cos, sin = self._build_tables()
        # Non-persistent: the tables are a deterministic function of position, so they
        # are rebuilt by `init_weights` rather than restored from a checkpoint.
        self.register_buffer("cos_cached", cos, persistent=False) # (S, D)
        self.register_buffer("sin_cached", sin, persistent=False) # (S, D)

    def _build_tables(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the (max_seq_len, qk_rope_head_dim) cos/sin tables, YaRN-scaled."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)) # (D/2,)
        t = torch.arange(self.seqlen, dtype=torch.float32, device=device)  # (S,)
        freqs = torch.outer(t, inv_freq) # (S, D/2)

        # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
        if self.seqlen > self.original_seq_len:
            low, high = self.find_correction_range(
                self.beta_fast, self.beta_slow, self.dim, self.base, self.original_seq_len
            )
            smooth = 1 - self.linear_ramp_factor(low, high, self.dim // 2).to(freqs.device)
            freqs = freqs / self.factor * (1 - smooth) + freqs * smooth

        # Interleave to match adjacent pairs: [f0, f0, f1, f1, ...]
        freqs = torch.repeat_interleave(freqs, 2, dim=-1) # (S, D)
        return freqs.cos(), freqs.sin()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Rebuild the cos/sin tables on a real device after a meta-device build.

        `to_empty()` allocates the buffers without writing them, so the tables must be
        recomputed here rather than carried over from `__init__`.
        """
        device = buffer_device if buffer_device is not None else self.cos_cached.device
        self.cos_cached, self.sin_cached = self._build_tables(device=device)

    @staticmethod
    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )


    @staticmethod
    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        low = math.floor(RotaryEmbedding.find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(RotaryEmbedding.find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    @staticmethod
    def rotate_adjacent(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def apply_rotary_emb(self, x: torch.Tensor, start_pos=0) -> torch.Tensor:
        # Under TP `x` is head-sharded while the tables are plain tensors, and DTensor
        # refuses to mix the two in a binary op. The rotation is per-position and
        # identical on every rank, so run it on the local shard and re-wrap.
        if isinstance(x, DTensor):
            local = self._rotate(x.to_local(), start_pos)
            return DTensor.from_local(local, x.device_mesh, x.placements, run_check=False)
        return self._rotate(x, start_pos)

    def _rotate(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        seqlen = x.size(1)
        cos = self.cos_cached[start_pos:start_pos+seqlen, :].view(1, seqlen, 1, self.dim)  # (1, seqlen, 1, dim)
        sin = self.sin_cached[start_pos:start_pos+seqlen, :].view(1, seqlen, 1, self.dim)  # (1, seqlen, 1, dim)
        # The tables are fp32 -- the angles need that precision -- so this expression
        # promotes a bf16 `x`. Return in x's dtype: everything downstream (the SDPA call
        # on the naive path, the raw matmuls on the absorb path) requires q, k and v to
        # agree, and a silently widened q is what breaks them.
        return (x * cos + self.rotate_adjacent(x) * sin).to(x.dtype)

if __name__ == "__main__":
    torch.manual_seed(123)
    
    # Pass an instantiated args object
    args = DeepSeekV3ModelArgs()
    rotary_embed = RotaryEmbedding(args)

    k = torch.randn((2, args.max_seq_len, args.n_heads, args.qk_rope_head_dim))
    k_embed = rotary_embed.apply_rotary_emb(k, start_pos=0)
    logger.info(f"Output shape: {k_embed.shape}")
