"""Combine k sampled bench runs of one split into per-task pass counts and pass rates.

Usage: python -m agent_from_scratch.evals aggregate RUN_DIR RUN_DIR ... [--output FILE]

For a task passed c times in k samples: mean pass@1 is c/k averaged over tasks, pass^k is the
share of tasks passed every time (reliability), and pass@k the share passed at least once.
Tasks with 0 < c < k are the ones whose outcome varies across samples, which Stage 10's GRPO
precondition needs.
"""

import argparse
import json
from pathlib import Path


def _rates(passes: list[int], k: int) -> dict:
    n = len(passes)
    return {"tasks": n, "mean_pass@1": round(sum(passes) / (n * k), 4) if n else None,
            "pass^k": round(sum(c == k for c in passes) / n, 4) if n else None,
            "pass@k": round(sum(c > 0 for c in passes) / n, 4) if n else None,
            "mixed": sum(0 < c < k for c in passes)}


def aggregate(runs: list[list[dict]], seeds: list[int]) -> dict:
    """`runs` are results.json record lists, one per seed, over the same task IDs."""
    if len(runs) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Aggregate at least two runs with distinct seeds.")
    ids = [{record["id"] for record in run} for run in runs]
    if any(task_ids != ids[0] for task_ids in ids):
        raise ValueError("Every run must cover the same task IDs.")
    k = len(runs)
    tasks: dict[str, dict] = {}
    for run in runs:
        for record in run:
            row = tasks.setdefault(record["id"], {"skeleton": record["skeleton"],
                                                  "family": record["family"], "passes": 0})
            row["passes"] += record["passed"]
    groups: dict[str, dict[str, list[int]]] = {"family": {}, "skeleton": {}}
    for row in tasks.values():
        for key in groups:
            groups[key].setdefault(row[key], []).append(row["passes"])
    claims = [r["false_completion"] for run in runs for r in run if r["false_completion"] is not None]
    return {"k": k, "seeds": seeds, "overall": _rates([r["passes"] for r in tasks.values()], k),
            "by_family": {name: _rates(p, k) for name, p in sorted(groups["family"].items())},
            "by_skeleton": {name: _rates(p, k) for name, p in sorted(groups["skeleton"].items())},
            "false_completion": {"count": sum(claims), "claims": len(claims)},
            "tasks": {task_id: tasks[task_id] for task_id in sorted(tasks)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="bench output directories, one per seed")
    parser.add_argument("--output", type=Path, help="Write the JSON report here instead of stdout")
    args = parser.parse_args()
    metadata = [json.loads((run / "metadata.json").read_text()) for run in args.runs]
    if len({(m.get("suite"), m.get("split")) for m in metadata}) != 1:
        parser.error("Runs must come from the same bench suite and split.")
    try:
        report = aggregate([json.loads((run / "results.json").read_text()) for run in args.runs],
                           [m["seed"] for m in metadata])
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
