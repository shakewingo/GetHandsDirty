from dataclasses import dataclass, field

from loguru import logger
from torch import nn
from typing import Literal
from torchfeather.model.moe import MoEArgs

@dataclass
class DeepSeekV3ModelArgs:
    max_seq_len: int = 4096 * 2 # 4096 * 4
    vocab_size: int = 4096 # 102400
    dim: int = 512 # 2048
    inter_dim: int = 2048 # 10944
    n_layers: int = 3 # 27
    n_dense_layers: int = 1
    n_heads: int = 8 # 16
    norm_eps: float = 1e-5  # eps used for RMSNorm

    # MoE
    moe_inter_dim: int = 1408
    moe_enabled: bool = True
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 256 # 512
    qk_nope_head_dim: int = 64 # 128
    qk_rope_head_dim: int = 32 # 64
    v_head_dim: int = 64# 128

    # yarn - to process the RoPE for context beyond original training length in inference
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # extra
    max_batch_size: int = 2
    attn_impl: Literal["naive", "absorb"] = "absorb"

    def get_params_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_experts = 0
        nparams_shared_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            elif "moe.shared_experts" in name:
                nparams_shared_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = nparams_moe_router + nparams_shared_experts + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
        nparams_active = nparams_dense + nparams_sparse_active
        logger.info(
            f"Total parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {nparams_active:,}"
        )

        n_layers = self.n_layers
        n_heads = self.n_heads
        head_dims = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim

        # Based on 6*P rule, P: params_counts
        # 1. embedding is excluded as it's just table lookup rather than operations
        # 2. attention is a separate operation to add rather than FNN-like operation
        # TODO: how attention FLOPS is calculated
        num_flops_per_token = 6 * (nparams_dense - nparams_embedding + nparams_sparse_active) + 6 * n_layers * n_heads * head_dims * seq_len
        return nparams, num_flops_per_token


