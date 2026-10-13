"""Ask a vLLM server to continue each example's prompt greedily and compare with its target.

Usage: python -m agent_from_scratch.evals replay --examples DATASET_DIR --output FILE
           [--adapter NAME] [--ids FILE] [--limit N] [--base-url URL] [--served-model NAME]

An adapter that was overfitted on a few examples must reproduce them exactly once it is loaded
by the server; the base model, asked the same way, shows what the adapter changed. On a whole
dataset the exact-match rate is a fit measure, not a capability result.
"""

import argparse
import json
from pathlib import Path

from ..run import save
from ...config import BENCH_DECODING, BENCH_MODEL, VLLM_BASE_URL
from ...llm import QWEN_END


def replay(examples: list[dict], http, model: str, *, max_tokens: int = BENCH_DECODING["max_tokens"]) -> dict:
    """Greedy continuations against targets.

    Args:
        examples: lines of examples.jsonl.
        http: an httpx.Client whose base URL is the server.
        model: the name to request, which is the adapter's name or the base model's.

    Returns:
        {"model", "examples", "exact", "mismatches": [{"id", "target", "generated", "finish_reason"}]}.
    """
    served = {entry["id"] for entry in http.get("/v1/models").raise_for_status().json()["data"]}
    if model not in served:
        raise ValueError(f"The server does not serve {model!r}; it serves {sorted(served)}.")
    mismatches = []
    for example in examples:
        body = {"model": model, "prompt": example["prompt"], "temperature": 0.0, "max_tokens": max_tokens,
                "stop": [QWEN_END]}
        choice = http.post("/v1/completions", json=body).raise_for_status().json()["choices"][0]
        if choice["text"] + QWEN_END != example["completion"]:
            mismatches.append({"id": example["id"], "target": example["completion"],
                               "generated": choice["text"], "finish_reason": choice.get("finish_reason")})
    return {"model": model, "examples": len(examples), "exact": len(examples) - len(mismatches),
            "mismatches": mismatches}


def main():
    import httpx

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, required=True, help="An export output directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the report")
    parser.add_argument("--adapter", help="LoRA module name on the server; omit for the base model")
    parser.add_argument("--ids", type=Path, help="A train metadata.json; replay exactly the examples it trained on")
    parser.add_argument("--limit", type=int, help="Replay the first N examples only")
    parser.add_argument("--base-url", default=VLLM_BASE_URL)
    parser.add_argument("--served-model", default=BENCH_MODEL["id"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; evidence is never overwritten.")
    examples = [json.loads(line) for line in
                (args.examples / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.ids is not None:
        wanted = set(json.loads(args.ids.read_text(encoding="utf-8"))["selection"]["example_ids"])
        examples = [example for example in examples if example["id"] in wanted]
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        parser.error("No example selected.")
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=600.0) as http:
        try:
            report = replay(examples, http, args.adapter or args.served_model)
        except ValueError as error:
            parser.error(str(error))
    save(args.output, report)
    print(json.dumps({key: report[key] for key in ("model", "examples", "exact")}))


if __name__ == "__main__":
    main()
