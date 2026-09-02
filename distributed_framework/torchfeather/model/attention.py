import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "ScaledDotProductAttentionWrapper",
]

class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.
    """
    def __init__(self) -> None:
        super().__init__()
