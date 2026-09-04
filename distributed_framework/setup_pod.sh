#!/usr/bin/env bash
# One-shot setup for a fresh RunPod pod. Run from distributed_framework/.
set -euo pipefail

pip install --no-cache-dir -r requirements.txt

# Tokenizer assets. hf_assets_path in the configs points here.
hf download deepseek-ai/deepseek-moe-16b-base \
  tokenizer.json tokenizer_config.json \
  --local-dir assets/hf/deepseek-moe-16b-base

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name} sm{p.major}{p.minor} {p.total_memory/2**30:.1f}GiB x{torch.cuda.device_count()}")
    # Stage 1 gate: _grouped_mm is Hopper-native and falls back below sm90. Confirm which.
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2, 64, 64, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([8, 16], device="cuda", dtype=torch.int32)
    try:
        torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
        print("torch._grouped_mm: OK")
    except Exception as e:
        print(f"torch._grouped_mm: FAILED -> {type(e).__name__}: {e}")
        print(">>> uncomment the reference loop in model/moe/moe.py")
PY

# Stage 1 gate: SDPA silently falls back to the math backend when a constraint fails,
# which would restore the (B,H,S,T) score tensor the wrapper exists to avoid. Assert it.
python -m torchfeather.minimal_examples.verify_flash_attention
