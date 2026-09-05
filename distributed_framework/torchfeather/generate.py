"""Generate text from a trained torchfeather checkpoint.

    python -m torchfeather.generate --prompt "The capital of France is"
    python -m torchfeather.generate --interactive --device cuda

Loads the bf16 export produced by Stage 4, not the sharded DCP checkpoint: a single
file, one process, no torchrun. Memory is the binding constraint on CPU -- an MoE
saves compute, not weights, so all 1.3B parameters must be resident even though only
0.41B are active per token. `--mmap` (on by default) keeps them file-backed so the
model runs on a box with less RAM than the checkpoint.
"""

import argparse
import time

import torch
from loguru import logger

from torchfeather.components.tokenizer import DeepSeekV3Tokenizer
from torchfeather.model.model import DeepSeekV3Model
from torchfeather.model.rope import RotaryEmbedding
from torchfeather.tools import device_utils

DEFAULT_WEIGHTS = "./artifacts/torchfeather-1b-16000.pt"
DEFAULT_TOKENIZER = "./assets/hf/deepseek-moe-16b-base"


def _materialize_buffers(model: torch.nn.Module, device: str) -> None:
    """Give real storage to the buffers that no checkpoint carries.

    The RoPE tables and the KV cache are registered non-persistent, so they are absent
    from the state dict and stay on the meta device after an `assign=True` load. They
    are small; the parameters are what must not be copied.
    """
    for mod in model.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_meta:
                mod._buffers[name] = torch.zeros(
                    buf.shape, dtype=buf.dtype, device=device
                )


def load_model(
    weights_path: str,
    device: str,
    dtype: torch.dtype,
    mmap: bool = True,
) -> tuple[DeepSeekV3Model, dict]:
    """Build the model from the args stored in the checkpoint and load its weights.

    Builds on the meta device in the target dtype and loads with `assign=True`, so the
    parameters become the checkpoint's tensors rather than copies of them. Combined with
    `mmap`, that means the weights stay file-backed and never occupy anonymous memory --
    the difference between running and being OOM-killed on a machine whose free RAM is
    smaller than the checkpoint.

    Args:
        weights_path (str): Path to the bf16 export (a dict with "model" and "model_args").
        device (str): Device to place the model on, e.g. "cpu" or "cuda".
        dtype (torch.dtype): Compute dtype. Must match the checkpoint's dtype when
            `mmap` is set, since assignment cannot convert in place.
        mmap (bool): Memory-map the tensors instead of reading them into RAM.

    Returns:
        tuple[DeepSeekV3Model, dict]: The loaded model in eval mode, and the checkpoint
            metadata (step, final_loss, model_args).
    """
    ckpt = torch.load(weights_path, map_location="cpu", mmap=mmap, weights_only=False)
    args = ckpt["model_args"]

    # `absorb` folds W^UK/W^UV into the q and o projections, which is what makes decode
    # cheap: the KV cache holds the latent, not per-head keys and values.
    args.attn_impl = "absorb"
    args.max_batch_size = 1

    with torch.device("meta"), device_utils.set_default_dtype(dtype):
        model = DeepSeekV3Model(args)

    missing, _ = model.load_state_dict(ckpt["model"], strict=False, assign=True)
    _materialize_buffers(model, device)
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.init_weights(torch.device(device))
    if missing:
        logger.warning(f"{len(missing)} tensors absent from the checkpoint, e.g. {missing[:3]}")

    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    return model, meta


def generate(
    model: DeepSeekV3Model,
    tokenizer: DeepSeekV3Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
) -> tuple[str, float]:
    """Complete `prompt` and return the continuation plus tokens/sec."""
    ids = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True, add_eos=False)], device=device
    )
    start = time.perf_counter()
    out = None
    for out in model.generate(
        {"input_ids": ids},
        eos=tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stream=False,
        use_kv_cache=True,
    ):
        pass
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(out[0].tolist())
    return text, len(out[0]) / elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--interactive", action="store_true", help="Read prompts from stdin in a loop.")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding.")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--no-mmap", action="store_true", help="Read weights into RAM instead of mapping them.")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    logger.info(f"loading {args.weights} onto {device} as {args.dtype}")
    model, meta = load_model(args.weights, device, dtype, mmap=not args.no_mmap)
    tokenizer = DeepSeekV3Tokenizer(args.tokenizer)
    logger.info(
        f"step {meta.get('step')} | final_loss {meta.get('final_loss')} | "
        f"{sum(p.numel() for p in model.parameters()):,} params"
    )

    prompts = iter(lambda: input("\nprompt> "), "") if args.interactive else [args.prompt]
    for prompt in prompts:
        if not prompt.strip():
            continue
        text, tps = generate(
            model, tokenizer, prompt, device,
            args.max_new_tokens, args.temperature,
            args.top_k if args.top_k > 0 else None,
            args.repetition_penalty,
        )
        print(f"\n{prompt}{text}")
        print(f"\n[{tps:.1f} tok/s]")


if __name__ == "__main__":
    main()
