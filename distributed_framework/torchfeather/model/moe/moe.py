from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor 


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = False

    # token-choice
    top_k: int = 1
    load_balance_coeff: float | None = 1e-3


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.inter_dim = hidden_dim
        self.dim = dim
        self.gate_proj = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.dim, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class TopKRouter(nn.Module):
    """
    Generate the expert score across all experts per token and select the top-k experts per token.
    """
    def __init__(self, dim: int, num_experts: int, top_k: int, 
                 score_func: Literal["softmax", "sigmoid"], route_norm: bool, 
                 route_scale: float):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.gate = nn.Linear(self.dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.dim) # (B, S, D) -> (B*S, D)
        score = self.gate(x) # (B*S, D) -> (B*S, num_experts)
        if self.score_func == "softmax":
            score = F.softmax(score, dim=-1)
        elif self.score_func == "sigmoid":
            score = torch.sigmoid(score)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")
    
        if expert_bias is not None:
            _, topk_experts_indices = torch.topk(score + expert_bias, self.top_k, dim=1, sorted=False)
            topk_scores = score.gather(dim=1, index=topk_experts_indices)   # gather gate value from original scores to ensure gate grad is not impacted
        else:
            topk_scores, topk_experts_indices = torch.topk(score, self.top_k, dim=1, sorted=False)

        if self.route_norm:
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
        topk_scores = topk_scores * self.route_scale

        # bincount, not histc: histc's CPU kernel is float-only, and the float counts it
        # returns propagate into ExpertParallel's all-to-all split sizes, which must be int.
        num_tokens_per_expert = torch.bincount(
            topk_experts_indices.view(-1), minlength=self.num_experts
        )  # (E,) int64
        return topk_experts_indices, topk_scores, num_tokens_per_expert

class TokenReorderer(nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, topk_scores, topk_experts_indices):
        """
        Given example that B*S=3, top_k=2, D=4
        topk_experts_indices = [[0, 1], [2, 3], [3, 0]] represents each token's top2 selected experts
        """
        perm = topk_experts_indices.view(-1).argsort(stable=True) # (B*S, K) -> (B*S*K, ); flatten the input and get the index of sorted value, e.g. [0, 5, 1, 2, 3, 4]
        # The histogram is recomputed here rather than reused from the router. 
        # In Phase 1 the two are identical and this is redundant; it is kept because in later phase the reorderer may see only a TP-shard of the tokens (ReordererSequenceParallel) 
        # while the router's histogram must stay global for the load-balance statistics. Keeping it now means Phase 2 does not edit this file.
        num_tokens_per_expert = torch.bincount(
            topk_experts_indices.view(-1), minlength=self.num_experts
        )  # (E,) int64 -- see the note in TopKRouter.forward
        return (
            topk_scores.view(-1)[perm],   # (B*S*K,)  score per slot, i.e get [0.9, 0.4, 0.1, 0.7, 0.3, 0.6] score for the given example
            perm // self.top_k,          # (B*S*K,)  token index per slot, i.e. get [0, 2, 0, 1, 1, 2] for the given example
            num_tokens_per_expert,       # (E,), i.e. get [2, 1, 1, 2] for the given example
        )

class GroupedExperts(nn.Module):
    """
    TODO: Apply EP shard and ETP when training on this part. Since we want to do parallelism based on experts, so the routed_input here is flatten selected tokens across all experts?
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))   # EP Shard(0); ETP Shard(1)
        self.up_proj   = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))   # EP Shard(0); ETP Shard(1)
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))   # EP Shard(0); ETP Shard(2)

    # def forward(self, routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    #     offsets = num_tokens_per_expert.cumsum(0, dtype=torch.int32).tolist()
    #     outs, start = [], 0
    #     for e, end in enumerate(offsets):
    #         """
    #         For the give example: 
    #         routed_input -> (B*S*K, D)
    #             expert 0: routed_input[0:2]  -> (2, D)
    #             expert 1: routed_input[2:3]  -> (1, D)
    #             expert 2: routed_input[3:4]  -> (1, D)
    #             expert 3: routed_input[4:6]  -> (2, D)
        
    #         """
    #         xe = routed_input[start:end]
    #         h = F.silu(xe @ self.gate_proj[e].T) * (xe @ self.up_proj[e].T)
    #         outs.append(h @ self.down_proj[e].T)
    #         start = end
    #     return torch.cat(outs, dim=0)

    def forward(self, routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offsets = num_tokens_per_expert.cumsum(0, dtype=torch.int32)
        w_gate, w_up, w_down = self.gate_proj, self.up_proj, self.down_proj
        if isinstance(w_gate, DTensor):
            w_gate, w_up, w_down = w_gate.to_local(), w_up.to_local(), w_down.to_local()
        x = routed_input.bfloat16()
        h = F.silu(torch._grouped_mm(x, w_gate.bfloat16().transpose(-2, -1), offs=offsets))  # torch._grouped_mm allows matrix manipulation computed in parallel
        h = h * torch._grouped_mm(x, w_up.bfloat16().transpose(-2, -1), offs=offsets)
        return torch._grouped_mm(h, w_down.bfloat16().transpose(-2, -1), offs=offsets).type_as(routed_input)
    
class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()
        num_experts = moe_args.num_experts
        self.experts = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
        self.shared_experts = (
            FeedForward(dim, hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )

        self.router = TopKRouter(dim, num_experts, moe_args.top_k, moe_args.score_func, moe_args.route_norm, moe_args.route_scale)
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        self.score_before_experts = moe_args.score_before_experts

        #define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        # expert_bias is updated outside the model in an optimizer step pre hook to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x = x.reshape(-1, D)

        topk_experts_indices, topk_scores, num_tokens_per_expert = self.router(x, self.expert_bias)
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)  # load-balance stats only

        topk_scores_experts_sorted, token_indices_experts_sorted, num_tokens_per_expert = self.reorderer(topk_scores, topk_experts_indices)

        routed_input = x[token_indices_experts_sorted]     # it will select the row index based on 1st dim of x, so eventually get (B*S*K, D), i.e. tokens replicated K times
        if self.score_before_experts:
            routed_input = (routed_input.float() * topk_scores_experts_sorted.unsqueeze(-1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)     # (B*S*K, D)
        out = self.shared_experts(x) if self.shared_experts is not None else torch.zeros_like(x)

        if not self.score_before_experts:
            # Align to `out`, not to `x`. Under torch.autocast (single-device / DDP, i.e. any
            # run where FSDP is not handling mixed precision) the shared-expert Linear returns
            # bf16 while `x` stays fp32, and index_add requires both operands to match.
            routed_output = (
                routed_output.float() * topk_scores_experts_sorted.unsqueeze(-1)
            ).to(out.dtype)

        # add output from routed experts to shared expert
        out = out.index_add(0, token_indices_experts_sorted, routed_output)
        return out.view(B, S, D)


# The benifit of cho. osing token-based choice for experts versus expert-based choice for tokens:
# 1. expert-based choice for tokens may have data leaking when model learns expert 1 chooses token 1 because token 1's next token is token 2. This may impact inference performance.
# 2. Will get load info for free with choosing token-based choice, bceause each token indicates how much prob they want to choose each expert.
