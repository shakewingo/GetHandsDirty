import torch

from torchfeather.tools.utils import _round_up
from .kernels import generate_permute_indices

TOKEN_GROUP_ALIGN_SIZE_M = 8


def _permute(x, num_tokens_per_expert_group, ep_degree, num_local_experts):
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
    with torch.no_grad():
        (
            permuted_indices,
            num_tokens_per_expert,
            _offsets,
        ) = generate_permute_indices(
            num_tokens_per_expert_group,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    x = torch.vstack((x, x.new_zeros(x.shape[-1])))
    input_shape = x.shape

    x = x[permuted_indices, :]
    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out
