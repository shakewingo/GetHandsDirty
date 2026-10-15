"""Train a LoRA adapter on exported SFT examples, with loss on the completion only.

Usage: python -m agent_from_scratch.evals train --examples DATASET_DIR --output DIR
           [--epochs 3] [--lr 1e-4] [--rank 16] [--alpha 32] [--dropout 0.05] [--accumulate 8]
           [--max-per-skeleton N] [--limit N] [--seed 0] [--save-last] [--pilot]

DATASET_DIR is an `export` output (examples.jsonl and metadata.json). Writes one adapter per
epoch (DIR/epoch-N, PEFT format), train_log.jsonl and metadata.json. Needs torch, transformers
and peft (requirements-train.txt); they are imported only when training starts.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
import time

from .tensorize import IGNORE, encode_example, select_examples
from ..run import save
from ...config import BENCH_MODEL, BENCH_N_CTX, CHAT_TEMPLATE_PATH
from ...utils import write_jsonl

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_dataset(directory: Path, *, pilot: bool = False) -> tuple[list[dict], dict]:
    """(examples, dataset metadata) from an `export` output directory.

    Refuses a pilot dataset unless `pilot`, and a dataset exported under another chat template.
    """
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["pilot"] and not pilot:
        raise ValueError(f"{directory.name} is a pilot dataset; --pilot marks a pipeline check.")
    template = sha256(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    if metadata["chat_template_sha256"] != template:
        raise ValueError(f"{directory.name} was exported under a different chat template than the one on disk.")
    # Split on "\n" only: str.splitlines() also breaks on U+2028 and similar characters, which
    # JSON written with ensure_ascii=False may hold inside a string.
    lines = (directory / "examples.jsonl").read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line], metadata


def fit(model, rows: list[dict], *, epochs: int, lr: float, accumulate: int = 8, seed: int = 0,
        warmup: float = 0.1, on_epoch=None) -> list[dict]:
    """Train `model`'s trainable parameters in place; one record per optimizer step.

    Each row is one forward pass (no padding). An optimizer step averages the loss of
    `accumulate` rows, and each row's loss is the mean over its own completion tokens.
    The optimizer is AdamW without weight decay, with gradients clipped at norm 1.0.

    Args:
        model: a causal LM whose only trainable parameters are the adapter's.
        rows: `encode_example` outputs.
        seed: fixes the order in which rows are visited, a new order every epoch.
        warmup: share of the steps with a linearly rising rate; cosine decay to zero after it.
        on_epoch: called with the 1-based epoch number after each epoch.

    Returns:
        list of {"step", "epoch", "loss", "lr", "rows"}.
    """
    import torch

    model.train()
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)

    total_steps = math.ceil(len(rows) / accumulate) * epochs
    warmup_steps = max(1, round(total_steps * warmup))

    def rate(step: int) -> float:
        """The learning rate of optimizer step `step`, counted from 0."""
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr / 2 * (1 + math.cos(math.pi * progress))

    order = random.Random(seed)
    records = []
    for epoch in range(1, epochs + 1):
        indices = list(range(len(rows)))
        order.shuffle(indices)
        for start in range(0, len(indices), accumulate):
            batch_indices = indices[start:start + accumulate]
            group_size = len(batch_indices)
            # Gradients add up over the group, so dividing each row's loss by the group
            # size makes the step the average of its rows.
            group_loss = 0.0
            for idx in batch_indices:
                row = rows[idx]
                input_ids = torch.tensor(row["input_ids"], device=device).unsqueeze(0)
                labels = torch.tensor(row["labels"], device=device).unsqueeze(0)
                loss = model(input_ids=input_ids, labels=labels).loss / group_size
                loss.backward()
                group_loss += loss.item()

            step_lr = rate(len(records))
            for group in optimizer.param_groups:
                group["lr"] = step_lr
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            records.append({"step": len(records) + 1, "epoch": epoch, "loss": group_loss,
                            "lr": step_lr, "rows": group_size})
        if on_epoch is not None:
            on_epoch(epoch)
    return records



def hf_tokenize(tokenizer):
    """text -> (ids, offsets) from a fast Hugging Face tokenizer, with no added special tokens."""
    def tokenize(text: str):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoded["input_ids"], [tuple(span) for span in encoded["offset_mapping"]]
    return tokenize


def add_lora(model, *, rank: int, alpha: int, dropout: float):
    """Freeze `model` and attach LoRA matrices to every attention and MLP projection."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                        target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, required=True, help="An export output directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the adapters")
    parser.add_argument("--base", default=BENCH_MODEL["id"], help="Hugging Face model ID or local path")
    parser.add_argument("--revision", default=BENCH_MODEL["revision"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--accumulate", type=int, default=8, help="Rows averaged per optimizer step")
    parser.add_argument("--max-per-skeleton", type=int, help="Cap examples per skeleton, whole trajectories")
    parser.add_argument("--limit", type=int, help="Train on the first N examples only (overfit smoke)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-last", action="store_true", help="Keep only the final epoch's adapter")
    parser.add_argument("--pilot", action="store_true", help="Accept a pilot dataset; the adapter is a pipeline check")
    args = parser.parse_args()
    out = args.output.resolve()
    package = Path(__file__).resolve().parents[2]
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    if out.is_relative_to(package):
        parser.error("Evidence must be outside the source package.")
    try:
        examples, dataset = load_dataset(args.examples, pilot=args.pilot)
    except (ValueError, OSError, KeyError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    examples = select_examples(examples, max_per_skeleton=args.max_per_skeleton, limit=args.limit)
    if not examples:
        parser.error("No example selected.")

    import peft
    import torch
    import transformers

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    try:
        rows = [encode_example(example, hf_tokenize(tokenizer), max_tokens=BENCH_N_CTX) for example in examples]
    except ValueError as error:
        parser.error(str(error))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    # Recompute activations during the backward pass instead of storing them all: without this
    # the longest rows do not fit in GPU memory.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Attach the adapter, then move the base and the adapter to the device together.
    model = add_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout).to(device)
    model.print_trainable_parameters()
    out.mkdir(parents=True)
    started = time.time()

    def save_epoch(epoch: int) -> None:
        if epoch == args.epochs or not args.save_last:
            model.save_pretrained(out / f"epoch-{epoch}")
    records = fit(model, rows, epochs=args.epochs, lr=args.lr, accumulate=args.accumulate, seed=args.seed,
                  on_epoch=save_epoch)
    write_jsonl(out / "train_log.jsonl", records)
    adapters = {path.parent.name: sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.glob("epoch-*/adapter_model.safetensors"))}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, text=True, capture_output=True,
                              check=True).stdout.strip()
    trained = sum(label != IGNORE for row in rows for label in row["labels"])
    save(out / "metadata.json", {
        "dataset": {"directory": args.examples.name, "track": dataset["track"], "pilot": dataset["pilot"],
                    "flawed_turns": dataset["flawed_turns"], "max_per_task": dataset["max_per_task"],
                    "examples_sha256": sha256((args.examples / "examples.jsonl").read_bytes()).hexdigest()},
        "selection": {"max_per_skeleton": args.max_per_skeleton, "limit": args.limit,
                      "examples": len(rows), "example_ids": [row["id"] for row in rows]},
        "tokens": {"total": sum(len(row["input_ids"]) for row in rows), "trained": trained},
        "base": args.base, "revision": args.revision, "dtype": str(next(model.parameters()).dtype),
        "gradient_checkpointing": True,
        "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "targets": list(LORA_TARGETS)},
        "optimizer": {"epochs": args.epochs, "lr": args.lr, "accumulate": args.accumulate, "warmup": 0.1,
                      "schedule": "linear warmup, cosine decay", "clip": 1.0, "weight_decay": 0.0},
        "adapters_sha256": adapters,
        "seed": args.seed, "steps": len(records), "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"], "elapsed_seconds": round(time.time() - started, 1),
        "device": device, "git_revision": revision,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__}})
    print(json.dumps({"examples": len(rows), "trained_tokens": trained, "steps": len(records),
                      "first_loss": round(records[0]["loss"], 4), "last_loss": round(records[-1]["loss"], 4)}))


if __name__ == "__main__":
    main()
