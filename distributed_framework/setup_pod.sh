#!/usr/bin/env bash
# One-shot setup for a fresh RunPod pod. Run from distributed_framework/.
set -euo pipefail

# Verified on a RunPod RTX 4090 (sm89, driver 570.169) 2026-09-04:
#  - the image's python is PEP 668 externally-managed, so pip needs
#    --break-system-packages. Fine here: the container is disposable.
#  - torch 2.13.0 does NOT exist for CUDA 12.8; that index stops at 2.11.0, and
#    2.13 ships only as cu130, which needs driver >= 580. 2.11 carries both APIs
#    this repo actually needs (DeviceMesh._unflatten, torch._grouped_mm); 2.8
#    does NOT have _unflatten, so 2.9 is the real floor.
CUDA_TAG="${CUDA_TAG:-cu128}"
pip install --break-system-packages --no-cache-dir --upgrade \
  torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install --break-system-packages --no-cache-dir --upgrade-strategy only-if-needed \
  -r requirements.txt

# Fail fast: a mismatched torch build reports cuda False and every later gate
# silently degrades to CPU instead of erroring.
python -c "import torch,sys; ok=torch.cuda.is_available(); print(f'torch {torch.__version__} | cuda {ok}'); sys.exit(0 if ok else 1)"

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
