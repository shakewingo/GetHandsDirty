"""Standalone check that Attention.absorb_weights() doesn't change forward() outputs.

Run with:
    cd /Users/yingyao/Desktop/Code.nosync/torchfeather
    uv run --with transformers python /Users/yingyao/Desktop/Code.nosync/GetHandsDirty.nosync/mix/torchfeather/model/verify_absorb.py

Notes:
- `torchfeather.model.rope` doesn't define a `RotaryEmbedding` class yet (only the functional
  `apply_rotary_emb`/`precompute_freqs_cis`), so it's stubbed out here with an identity rotation.
  That's fine for this check: rope is untouched by the fusion, we're only verifying the
  W^UK/W^UV absorption math, so isolating it from a real rope implementation is deliberate.
- `transformers` isn't a torchfeather dependency, hence `--with transformers` above instead of
  editing the project's pyproject.toml.
"""
import sys
import time
import types
from types import SimpleNamespace

import torch

rope_stub = types.ModuleType("torchfeather.model.rope")


class _IdentityRotaryEmbedding:
    def __init__(self, args):
        pass

    def apply_rotary_emb(self, x, start_pos=0):
        return x


rope_stub.RotaryEmbedding = _IdentityRotaryEmbedding
sys.modules["torchfeather.model.rope"] = rope_stub

import importlib.util

MODEL_PATH = "/Users/yingyao/Desktop/Code.nosync/GetHandsDirty.nosync/mix/torchfeather/model/model.py"
spec = importlib.util.spec_from_file_location("mix_model", MODEL_PATH)
mix_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mix_model)
Attention = mix_model.Attention


def make_args(q_lora_rank):
    return SimpleNamespace(
        dim=32,
        n_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        max_seq_len=64,
        original_seq_len=64,
        mscale=1.0,
        rope_factor=1,
        max_batch_size=2,
        attn_impl="absorb",
    )


def check_equivalence(q_lora_rank):
    torch.manual_seed(0)
    args = make_args(q_lora_rank)
    attn = Attention(args).eval()
    x = torch.randn(2, 6, args.dim)

    with torch.no_grad():
        out_ref = attn(x, use_kv_cache=False)
        attn.absorb_weights()
        out_fused = attn(x, use_kv_cache=False)

    max_diff = (out_ref - out_fused).abs().max().item()
    print(f"[q_lora_rank={q_lora_rank}] training-mode (no cache) max abs diff: {max_diff:.3e}")
    assert torch.allclose(out_ref, out_fused, atol=1e-5), "fused output diverges from reference"

    # Also check the incremental-decode path: prefill, then a couple of single-token steps.
    torch.manual_seed(0)
    args2 = make_args(q_lora_rank)
    attn_ref = Attention(args2).eval()
    attn_fused = Attention(args2).eval()
    attn_fused.load_state_dict(attn_ref.state_dict())
    attn_fused.absorb_weights()

    prompt = torch.randn(2, 5, args.dim)
    with torch.no_grad():
        attn_ref(prompt, use_kv_cache=True, start_pos=0)
        attn_fused(prompt, use_kv_cache=True, start_pos=0)
        start_pos = 5
        for _ in range(3):
            step = torch.randn(2, 1, args.dim)
            out_ref_step = attn_ref(step, use_kv_cache=True, start_pos=start_pos)
            out_fused_step = attn_fused(step, use_kv_cache=True, start_pos=start_pos)
            step_diff = (out_ref_step - out_fused_step).abs().max().item()
            print(f"[q_lora_rank={q_lora_rank}] decode step @ start_pos={start_pos} max abs diff: {step_diff:.3e}")
            assert torch.allclose(out_ref_step, out_fused_step, atol=1e-5)
            start_pos += 1
    print(f"[q_lora_rank={q_lora_rank}] OK: fused weights match unfused reference in both modes\n")


def bench(q_lora_rank, n_steps=200):
    torch.manual_seed(0)
    args = make_args(q_lora_rank)
    attn_ref = Attention(args).eval()
    attn_fused = Attention(args).eval()
    attn_fused.load_state_dict(attn_ref.state_dict())
    attn_fused.absorb_weights()

    prompt = torch.randn(args.max_batch_size, 64, args.dim)
    with torch.no_grad():
        attn_ref(prompt, use_kv_cache=True, start_pos=0)
        attn_fused(prompt, use_kv_cache=True, start_pos=0)

        step = torch.randn(args.max_batch_size, 1, args.dim)
        start = time.perf_counter()
        for i in range(n_steps):
            attn_ref(step, use_kv_cache=True, start_pos=64 + i)
        t_ref = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(n_steps):
            attn_fused(step, use_kv_cache=True, start_pos=64 + i)
        t_fused = time.perf_counter() - start

    print(f"[q_lora_rank={q_lora_rank}] {n_steps} decode steps: unfused={t_ref:.4f}s fused={t_fused:.4f}s "
          f"({t_ref / t_fused:.2f}x)")


if __name__ == "__main__":
    check_equivalence(q_lora_rank=0)
    check_equivalence(q_lora_rank=8)
    bench(q_lora_rank=0)
