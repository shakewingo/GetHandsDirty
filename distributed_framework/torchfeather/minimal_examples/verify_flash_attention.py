"""Verify the attention path: numerics on any device, flash-backend selection on CUDA.

Calling `F.scaled_dot_product_attention` does not mean FlashAttention runs. The dispatcher
silently falls back to the math backend -- which materializes the (B, H, S, T) score matrix,
the exact cost the wrapper exists to avoid -- whenever a constraint fails. This script asserts
the fallback is not happening.

Run on CPU for the numerics, and on the Stage 1 GPU pod for the backend assertion:

    python -m torchfeather.minimal_examples.verify_flash_attention
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torchfeather.model.attention import ScaledDotProductAttentionWrapper

B, H, S, D_QK, D_V = 2, 8, 256, 96, 64


def _reference(q, k, v, scale):
    """The explicit matmul / mask-add / softmax chain the wrapper replaced."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal = torch.full((q.shape[-2], k.shape[-2]), float("-inf"), device=q.device)
    scores = scores + torch.triu(causal, diagonal=1)
    return torch.matmul(scores.softmax(dim=-1), v)


def check_numerics(device: str, dtype: torch.dtype) -> None:
    """The V-padding trick must be exact, not approximate."""
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D_QK, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D_QK, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D_V, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D_QK)

    got = ScaledDotProductAttentionWrapper()(q, k, v, scale=scale, is_causal=True)
    want = _reference(q.float(), k.float(), v.float(), scale).to(dtype)

    assert got.shape == (B, H, S, D_V), got.shape
    tol = 5e-2 if dtype is torch.bfloat16 else 1e-4
    diff = (got.float() - want.float()).abs().max().item()
    assert diff < tol, f"maxdiff {diff} >= {tol}"
    print(f"[PASS] numerics {device}/{dtype}  shape {tuple(got.shape)}  maxdiff {diff:.2e}")


def check_grad(device: str, dtype: torch.dtype) -> None:
    """Padded V columns must carry no gradient back into the real ones."""
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D_QK, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D_QK, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D_V, device=device, dtype=dtype, requires_grad=True)

    ScaledDotProductAttentionWrapper()(
        q, k, v, scale=1.0 / math.sqrt(D_QK), is_causal=True
    ).sum().backward()

    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None, f"{name} has no grad"
        assert torch.isfinite(t.grad).all(), f"{name} grad not finite"
        assert t.grad.abs().sum() > 0, f"{name} grad is all zero"
    assert v.grad.shape == v.shape, v.grad.shape
    print(f"[PASS] gradients reach q/k/v, finite and non-zero, v.grad {tuple(v.grad.shape)}")


def check_flash_selected() -> None:
    """The decisive test: force flash-only and confirm the call still succeeds.

    `sdpa_kernel([FLASH_ATTENTION])` disables every other backend, so a constraint failure
    raises here instead of silently degrading to math.
    """
    q = torch.randn(B, H, S, D_QK, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D_QK, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D_V, device="cuda", dtype=torch.bfloat16)
    wrapper = ScaledDotProductAttentionWrapper()

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = wrapper(q, k, v, scale=1.0 / math.sqrt(D_QK), is_causal=True)
    assert out.shape == (B, H, S, D_V), out.shape
    print(f"[PASS] FLASH_ATTENTION accepts the training shapes "
          f"(Dq={D_QK} Dk={D_QK} Dv={D_V}->{D_QK} padded, bf16, is_causal)")

    # Peak memory is the point: flash must not materialize (B, H, S, T).
    score_matrix_bytes = B * H * S * S * 2
    for label, backends in (
        ("flash", [SDPBackend.FLASH_ATTENTION]),
        ("math ", [SDPBackend.MATH]),
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(backends):
            wrapper(q, k, v, scale=1.0 / math.sqrt(D_QK), is_causal=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print(f"       {label} peak {peak / 2**20:8.2f} MiB")
    print(f"       an (B,H,S,T) score matrix would be {score_matrix_bytes / 2**20:.2f} MiB")


def check_absorb_rejected() -> None:
    """Document why the absorbed path cannot use flash: Dq=288 exceeds the 256 head-dim cap."""
    d_absorbed = 256 + 32
    q = torch.randn(B, H, 64, d_absorbed, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, 64, d_absorbed, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, 64, 256, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            ScaledDotProductAttentionWrapper()(q, k, v, is_causal=True)
    except RuntimeError as exc:
        print(f"[INFO] absorbed dims (Dq={d_absorbed}) rejected by flash, as expected: "
              f"{str(exc).splitlines()[0][:100]}")
        return
    print(f"[INFO] absorbed dims (Dq={d_absorbed}) unexpectedly accepted by flash")


if __name__ == "__main__":
    check_numerics("cpu", torch.float32)
    check_grad("cpu", torch.float32)

    if torch.cuda.is_available():
        print(f"\n--- CUDA: {torch.cuda.get_device_name(0)} ---")
        check_numerics("cuda", torch.bfloat16)
        check_grad("cuda", torch.bfloat16)
        check_flash_selected()
        check_absorb_rejected()
    else:
        print("\n[SKIP] no CUDA -- backend selection is unverified. "
              "Re-run this on the Stage 1 pod before trusting the flash claim.")
