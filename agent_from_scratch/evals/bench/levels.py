"""Choose each Stage 10 shape's level from calibration runs (docs/STAGE10_HARD_TASKS_PLAN.md, K4).

Usage: python -m agent_from_scratch.evals levels RUN_DIR ... --output REPORT.json [--levels LEVELS.json]

RUN_DIRs are `bench --split calib` runs, one per seed. For every shape and level, the strict
pass@1 over all its attempts (GRPO's reward: passed, with the answer exactly as requested).
C1: per shape, the level whose rate lies in [0.25, 0.75] and is closest to 0.5, the lower level
on a tie; no level in band drops the shape. C2: at least 4 train, 2 val and 2 hardtest shapes
are kept. --levels writes the chosen levels in levels.json's format, to be reviewed before it
replaces evals/bench/levels.json.
"""

import argparse
import json
from pathlib import Path

BAND = (0.25, 0.75)
MINIMUM = {"train": 4, "val": 2, "hardtest": 2}


def choose_level(rates: dict[int, float]) -> int | None:
    """C1: the level in BAND closest to 0.5, the lower level on a tie; None when none is in band."""
    inside = [level for level, rate in rates.items() if BAND[0] <= rate <= BAND[1]]
    return min(inside, key=lambda level: (abs(rates[level] - 0.5), level)) if inside else None


def verdict(chosen: dict[str, int | None], splits: dict[str, str]) -> dict:
    """C2: the kept shapes per split (`splits`: shape -> split) and whether each minimum holds."""
    kept = {split: sorted(name for name, level in chosen.items() if level is not None and splits[name] == split)
            for split in MINIMUM}
    return {"kept": kept, "minimum": MINIMUM,
            "passed": all(len(kept[split]) >= count for split, count in MINIMUM.items())}


def rates_from_runs(runs: list[list[dict]]) -> dict[str, dict[int, float]]:
    """shape -> level -> strict pass@1, from calibration records named NAME@L<level>."""
    from ..sft.grpo import reward

    earned: dict[str, dict[int, list[int]]] = {}
    for records in runs:
        for record in records:
            name, _, level = record["skeleton"].rpartition("@L")
            earned.setdefault(name, {}).setdefault(int(level), []).append(reward(record))
    return {name: {level: round(sum(values) / len(values), 4) for level, values in sorted(levels.items())}
            for name, levels in sorted(earned.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="bench --split calib runs, one per seed")
    parser.add_argument("--output", type=Path, required=True, help="The report (rates, choices, C2)")
    parser.add_argument("--levels", type=Path, help="Also write the chosen levels as levels.json")
    args = parser.parse_args()
    for path in (args.output, args.levels):
        if path is not None and path.exists():
            parser.error(f"{path} already exists; evidence is never overwritten.")
    metadata = [json.loads((run / "metadata.json").read_text()) for run in args.runs]
    if any(m.get("split") != "calib" for m in metadata):
        parser.error("Only bench --split calib runs choose levels.")
    if len({m["seed"] for m in metadata}) != len(metadata):
        parser.error("Each run must be a different seed.")
    from .skeletons import SHAPES

    rates = rates_from_runs([json.loads((run / "results.json").read_text()) for run in args.runs])
    splits = {shape.name: shape.split for shape in SHAPES}
    unknown = sorted(set(rates) - set(splits))
    if unknown:
        parser.error(f"Runs hold shapes that are not registered: {unknown}")
    chosen = {name: choose_level(rates.get(name, {})) for name in splits}
    report = {"seeds": sorted(m["seed"] for m in metadata), "band": BAND, "rates": rates, "chosen": chosen,
              **verdict(chosen, splits)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.levels is not None:
        args.levels.write_text(json.dumps({n: l for n, l in chosen.items() if l is not None}, indent=2,
                                          sort_keys=True) + "\n")
    print(json.dumps({"chosen": chosen, "kept": report["kept"], "c2_passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
