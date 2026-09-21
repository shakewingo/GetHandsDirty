"""Domain-agnostic trajectory profile and paired comparison of eval runs.

Usage: python -m agent_from_scratch.evals.trajectory RUN_DIR [OTHER_RUN_DIR]
Reads each directory's results.json. With two runs, pairs tasks by ID (second minus first).
"""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
from statistics import mean, median

MEANS = ("elided_messages", "compact_requests", "plan_updates", "stuck_reminders", "model_requests")


def profile(records: list[dict]) -> dict:
    """Summarize run shape: stop reasons, survival by actor request, mechanism use and cost.

    Args:
        records: per-task records from `evals.run`, each carrying `metrics()` fields; an eval
            run always has at least one.

    Returns:
        dict: pass count, stop-reason counts, overflow (`context_limit`) rate, median actor
            requests and peak window share, per-run means, and one survival row per request
            index with the share of runs still active and their action mix.
    """
    lengths = [len(r["actions"]) for r in records]
    peaks = [r["peak_context_ratio"] for r in records if r["peak_context_ratio"] is not None]
    tokens = [r["usage"]["total_tokens"] for r in records]
    survival = []
    for index in range(max(lengths)):
        active = [r["actions"][index] for r in records if len(r["actions"]) > index]
        survival.append({"request": index + 1, "active": round(len(active) / len(records), 4),
                         "mix": dict(Counter(active))})
    stops = Counter(r["stop_reason"] for r in records)
    return {
        "runs": len(records), "passed": sum(r["passed"] for r in records),
        "stop_reasons": dict(stops),
        "context_limit_rate": round(stops["context_limit"] / len(records), 4),
        "median_actor_requests": median(lengths),
        "median_peak_context_ratio": median(peaks) if peaks else None,
        # Unknown usage in any run leaves the mean unknown rather than silently smaller.
        "mean": {**{key: round(mean(r[key] for r in records), 4) for key in MEANS},
                 "total_tokens": round(mean(tokens), 1) if None not in tokens else None},
        "survival": survival,
    }


def compare(first: list[dict], second: list[dict]) -> dict:
    """Pair two runs by task ID; exact two-sided McNemar p over the discordant tasks."""
    a = {r["id"]: r["passed"] for r in first}
    b = {r["id"]: r["passed"] for r in second}
    shared = sorted(a.keys() & b.keys())
    only_a = sum(a[t] and not b[t] for t in shared)
    only_b = sum(b[t] and not a[t] for t in shared)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"tasks": len(shared), "both": sum(a[t] and b[t] for t in shared),
            "only_a": only_a, "only_b": only_b,
            "neither": sum(not a[t] and not b[t] for t in shared), "mcnemar_p": round(p, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="One or two eval output directories")
    args = parser.parse_args()
    if len(args.runs) > 2:
        parser.error("Give one run to profile or two runs to compare.")
    records = [json.loads((run / "results.json").read_text()) for run in args.runs]
    report = {"profiles": {str(run): profile(r) for run, r in zip(args.runs, records)}}
    if len(records) == 2:
        report["paired"] = compare(*records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
