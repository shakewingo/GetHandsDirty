"""Combine our SFT examples with public ones into one dataset that `train` reads unchanged.

Usage: python -m agent_from_scratch.evals mix --own DIR --public DIR [DIR ...] --output DIR
           [--self DIR] [--own-share 0.333] [--seed 0]

Our examples are repeated a whole number of times k, chosen so that they come closest to
--own-share of the rows; the public datasets are used as `public` sampled them. --self adds a
student's own verified rollouts (an `rft` export) once, without changing k. The rows are then
shuffled with --seed. Design: docs/STAGE9_ITEM4_PLAN.md (D7) and docs/STAGE10_PLAN.md (E1).
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

from .train import load_dataset
from ..run import save
from ...utils import write_jsonl


def repeat_factor(own: int, public: int, share: float) -> int:
    """The whole number k >= 1 for which k * own is closest to `share` of all the rows.

    k * own / (k * own + public) = share  gives  k = share * public / ((1 - share) * own).
    """
    if own < 1 or not 0 < share < 1:
        raise ValueError("Need at least one own example and a share strictly between 0 and 1.")
    return max(1, round(share * public / ((1 - share) * own)))


def combine(own: list[dict], public: list[list[dict]], *, share: float, seed: int,
            extra: list[dict] = ()) -> tuple[list[dict], int]:
    """(shuffled rows, k): `own` repeated k times, then every public dataset and `extra` once.

    `extra` is left out when choosing k, so adding it leaves the other rows as they were.
    """
    k = repeat_factor(len(own), sum(map(len, public)), share)
    rows = [{**example, "provenance": {**example["provenance"], "copy": copy}}
            for copy in range(k) for example in own]
    rows += [example for dataset in public for example in dataset]
    rows += list(extra)
    random.Random(seed).shuffle(rows)
    return rows, k


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--own", type=Path, required=True, help="An export output (teacher-wide)")
    parser.add_argument("--public", type=Path, nargs="+", required=True, help="public outputs")
    parser.add_argument("--self", dest="self_set", type=Path,
                        help="An rft export (the student's own verified rollouts), added once")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--own-share", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Evidence must be outside the source package.")
    try:
        # load_dataset refuses a pilot dataset and one exported under another chat template.
        own, own_metadata = load_dataset(args.own)
        public = [load_dataset(directory) for directory in args.public]
        extra, extra_metadata = load_dataset(args.self_set) if args.self_set else ([], None)
        rows, k = combine(own, [examples for examples, _ in public], share=args.own_share, seed=args.seed,
                          extra=extra)
    except (ValueError, OSError, KeyError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    if own_metadata["track"] == "public" or any(metadata["track"] != "public" for _, metadata in public):
        parser.error("--own takes our own examples and --public only public ones.")
    if extra_metadata is not None and extra_metadata["track"] != "rft":
        parser.error("--self takes an rft export: the student's own verified rollouts.")

    def source(directory: Path, metadata: dict, examples: list[dict]) -> dict:
        return {"directory": directory.name, "track": metadata["track"], "examples": len(examples),
                "examples_sha256": sha256((directory / "examples.jsonl").read_bytes()).hexdigest(),
                **({"source": metadata["source"]} if "source" in metadata else {})}

    out.mkdir(parents=True)
    write_jsonl(out / "examples.jsonl", rows)
    own_rows = k * len(own)
    # `train` reads track, pilot, flawed_turns, max_per_task and chat_template_sha256.
    save(out / "metadata.json", {
        "track": "mix", "pilot": False, "flawed_turns": own_metadata["flawed_turns"],
        "max_per_task": own_metadata["max_per_task"],
        "chat_template_sha256": own_metadata["chat_template_sha256"], "target": "canonical",
        "own": source(args.own, own_metadata, own),
        "public": [source(directory, metadata, examples)
                   for directory, (examples, metadata) in zip(args.public, public)],
        "self": source(args.self_set, extra_metadata, extra) if args.self_set else None,
        "repeat_own": k, "own_share_target": args.own_share,
        "own_share": round(own_rows / len(rows), 4), "self_share": round(len(extra) / len(rows), 4),
        "examples": len(rows), "seed": args.seed})
    print(json.dumps({"examples": len(rows), "repeat_own": k, "own_share": round(own_rows / len(rows), 4)}))


if __name__ == "__main__":
    main()
