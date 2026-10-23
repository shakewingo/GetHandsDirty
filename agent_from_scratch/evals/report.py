"""Stage 11's result tables (the `report` command): pass@1 by arm and subset, paired
comparisons, bootstrap intervals. (Not `results.py`: the repository ignores `*result*` files.)

Usage: python -m agent_from_scratch.evals report --arm NAME=RUN,RUN,RUN [--arm ...]
           [--subset NAME=SKELETON,SKELETON ...] [--pair A:B ...] [--output FILE.md]

An arm is one bench run per seed (results.json directories). pass@1 is the share of passed runs
over every task and seed. A pair (A:B) is matched by task and seed: wins and losses for B against A,
McNemar's exact p over the discordant pairs (the `compare` test), and a 95% bootstrap interval of
B's pass@1 minus A's that resamples tasks, keeping all seeds of a task together. An arm with fewer
runs than another is reported with its seed count, never averaged silently. Design:
docs/STAGE11_PLAN.md (F6).
"""

import argparse
import json
from pathlib import Path
import random

from .trajectory import compare


def load(paths: list[Path]) -> list[list[dict]]:
    return [json.loads((path / "results.json").read_text(encoding="utf-8")) for path in paths]


def keep(runs: list[list[dict]], skeletons: set[str] | None) -> list[list[dict]]:
    return runs if skeletons is None else [[r for r in run if r["skeleton"] in skeletons] for run in runs]


def pass_at_1(runs: list[list[dict]]) -> float:
    """Passed runs over every task and seed."""
    records = [r for run in runs for r in run]
    if not records:
        raise ValueError("No records.")
    return sum(bool(r["passed"]) for r in records) / len(records)


def paired(a: list[list[dict]], b: list[list[dict]]) -> dict:
    """B against A, matched by task and seed (seed i of A with seed i of B)."""
    if len(a) != len(b):
        raise ValueError(f"Seed counts differ: {len(a)} and {len(b)}.")
    first = [{**r, "id": f"{r['id']}#{seed}"} for seed, run in enumerate(a) for r in run]
    second = [{**r, "id": f"{r['id']}#{seed}"} for seed, run in enumerate(b) for r in run]
    result = compare(first, second)
    return {"pairs": result["tasks"], "wins": result["only_b"], "losses": result["only_a"],
            "both": result["both"], "neither": result["neither"], "mcnemar_p": result["mcnemar_p"]}


def bootstrap_diff(a: list[list[dict]], b: list[list[dict]], *, resamples: int = 10000,
                   seed: int = 0) -> tuple[float, float, float]:
    """(B - A in pass@1, 2.5th percentile, 97.5th percentile), resampling tasks with every seed kept."""
    def per_task(runs):
        scores: dict[str, list[int]] = {}
        for run in runs:
            for r in run:
                scores.setdefault(r["id"], []).append(int(bool(r["passed"])))
        return {task: sum(v) / len(v) for task, v in scores.items()}
    sa, sb = per_task(a), per_task(b)
    tasks = sorted(sa.keys() & sb.keys())
    if not tasks:
        raise ValueError("No shared task.")
    diffs = [sb[t] - sa[t] for t in tasks]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in tasks) / len(tasks) for _ in range(resamples))
    return sum(diffs) / len(diffs), means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def table(arms: dict[str, list[list[dict]]], subsets: dict[str, set[str] | None]) -> str:
    """One row per arm: pass@1 per subset (with its task count), and the passed count of each seed."""
    seeds = max(map(len, arms.values()))
    head = "| Arm | " + " | ".join(f"{name}" for name in subsets) + " | Passed by seed |"
    lines = [head, "|---|" + "---:|" * len(subsets) + "---|"]
    for name, runs in arms.items():
        cells = []
        for skeletons in subsets.values():
            part = keep(runs, skeletons)
            tasks = len(part[0]) if part else 0
            cells.append(f"{pass_at_1(part):.3f} ({tasks})" if tasks else "—")
        counts = ", ".join(str(sum(bool(r["passed"]) for r in run)) for run in runs)
        note = f" (**{len(runs)} of {seeds} seeds**)" if len(runs) < seeds else ""
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {counts}{note} |")
    return "\n".join(lines) + "\n"


def pair_table(arms: dict[str, list[list[dict]]], pairs: list[tuple[str, str]],
               subsets: dict[str, set[str] | None]) -> str:
    lines = ["| B against A | Subset | Pairs | B wins | B loses | McNemar p | B − A, 95% interval |",
             "|---|---|---:|---:|---:|---:|---|"]
    for a_name, b_name in pairs:
        for subset, skeletons in subsets.items():
            a, b = keep(arms[a_name], skeletons), keep(arms[b_name], skeletons)
            if not a or not a[0]:
                continue
            p = paired(a, b)
            diff, low, high = bootstrap_diff(a, b)
            lines.append(f"| {b_name} vs {a_name} | {subset} | {p['pairs']} | {p['wins']} | {p['losses']} | "
                         f"{p['mcnemar_p']:.4f} | {diff:+.3f} [{low:+.3f}, {high:+.3f}] |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="NAME=RUN_DIR,RUN_DIR,...")
    parser.add_argument("--subset", action="append", default=[], help="NAME=SKELETON,SKELETON,...")
    parser.add_argument("--pair", action="append", default=[], help="A:B, B against A")
    parser.add_argument("--output", type=Path, help="Write the markdown here instead of stdout")
    args = parser.parse_args()
    try:
        arms = {name: load([Path(p) for p in paths.split(",")])
                for name, paths in (item.split("=", 1) for item in args.arm)}
        subsets: dict[str, set[str] | None] = {"all": None}
        subsets |= {name: set(skeletons.split(",")) for name, skeletons in (item.split("=", 1) for item in args.subset)}
        pairs = [tuple(item.split(":", 1)) for item in args.pair]
        unknown = [name for pair in pairs for name in pair if name not in arms]
        if unknown:
            parser.error(f"Unknown arms in --pair: {unknown}")
        text = table(arms, subsets) + ("\n" + pair_table(arms, pairs, subsets) if pairs else "")
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
