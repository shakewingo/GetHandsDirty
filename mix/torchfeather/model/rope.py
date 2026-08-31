import math
import random
import numpy as np
from loguru import logger
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torchfeather.model.model_args import DeepSeekV3ModelArgs

class RotaryEmbedding(nn.Module):
    def __init__(self, args: DeepSeekV3ModelArgs):
        super(RotaryEmbedding, self).__init__()
        self.dim = args.qk_rope_head_dim
        self.seqlen = args.max_seq_len
        beta_fast = args.beta_fast
        beta_slow = args.beta_slow
        base = args.rope_theta
        factor = args.rope_factor

        self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)) # (D/2,)
        self.t = torch.arange(self.seqlen, dtype=torch.float32)  # (S,)
        self.freqs = torch.outer(self.t, self.inv_freq) # (S, D/2)
        logger.debug(f"freqs shape: {self.freqs.shape}")

        # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
        if self.seqlen > args.original_seq_len:
            low, high = self.find_correction_range(
                beta_fast, beta_slow, self.dim, base, args.original_seq_len
            )
            smooth = 1 - self.linear_ramp_factor(low, high, self.dim // 2)
            self.freqs = self.freqs / factor * (1 - smooth) + self.freqs * smooth
            logger.debug(f"freqs shape after YaRN: {self.freqs.shape}")
        # Interleave to match adjacent pairs: [f0, f0, f1, f1, ...]
        self.freqs = torch.repeat_interleave(self.freqs, 2, dim=-1) # (S, D)

        self.register_buffer("cos_cached", self.freqs.cos()) # (S, D)
        self.register_buffer("sin_cached", self.freqs.sin()) # (S, D)

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
        seqlen = x.size(1)
        cos = self.cos_cached[start_pos:start_pos+seqlen, :].view(1, seqlen, 1, self.dim)  # (1, seqlen, 1, dim)
        sin = self.sin_cached[start_pos:start_pos+seqlen, :].view(1, seqlen, 1, self.dim)  # (1, seqlen, 1, dim)
        return x * cos + self.rotate_adjacent(x) * sin

if __name__ == "__main__":
    torch.manual_seed(123)
    
    # Pass an instantiated args object
    args = DeepSeekV3ModelArgs()
    rotary_embed = RotaryEmbedding(args)

    k = torch.randn((2, args.max_seq_len, args.n_heads, args.qk_rope_head_dim))
    k_embed = rotary_embed.apply_rotary_emb(k, start_pos=0)
    logger.info(f"Output shape: {k_embed.shape}")
