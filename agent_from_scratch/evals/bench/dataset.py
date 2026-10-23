"""Generate Stage 9 fixtures and optional scripted gates, without loading a model.

Usage: python -m agent_from_scratch.evals dataset --output DIR [--split train|test|all] [--check]
Contracts live outside each workspace; only workspace files and Task.prompt reach the agent.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

from loguru import logger

from .manifest import build_manifest
from .run import run_case, specs
from .skeletons import answer
from ..run import save
from ...llm import LLM


class ScriptedModel:
    """A gate fixture, not a teacher; tools still execute inside the ordinary agent loop."""

    def __init__(self, responses):
        self.responses = iter(responses)

    def settings(self):
        return {"backend": "scripted", "context_measurement": "gate fixture only"}

    def generate(self, messages, tools, **kwargs):
        return next(self.responses)

    def measure_context(self, messages, tools, **kwargs):
        # The runtime requires this marker. These controlled-fit numbers are gate
        # fixtures, not tokenizer measurements; settings/metadata disclose that.
        return {"count_method": "exact", "prompt_tokens": 1000, "window_tokens": 32768,
                "response_reserve": 2048, "remaining_tokens": 29720}

    read_usage = staticmethod(LLM.read_usage)


def generate(out: Path, split: str = "all", check: bool = False) -> dict:
    out = out.resolve()
    package = Path(__file__).resolve().parents[2]
    if out.is_relative_to(package):
        raise ValueError("Output must be outside the source package.")
    if out.exists():
        raise ValueError("Output already exists; evidence is never overwritten.")
    entries = [(s, ctx) for s, ctx in specs() if ctx.split in ("train", "test")
               and (split == "all" or ctx.split == split)]
    if not entries:
        raise ValueError("No tasks selected.")
    out.mkdir(parents=True)
    save(out / "manifest.json", build_manifest())
    counts = dict(Counter(ctx.split for _, ctx in entries))
    save(out / "metadata.json", {"dataset_version": "stage9-v3", "task_counts": counts,
                                 "skeletons": sorted({s.name for s, _ in entries}),
                                 "scripted_check": check, "model_inference": False,
                                 "context_measurement": "scripted controlled-fit fixture, not actor token counts"})
    results = []
    for skeleton, ctx in entries:
        directory = out / "tasks" / ctx.id
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), workspace, ctx)
        save(directory / "task.json", asdict(task))
        if check:
            scripts = {"correct": skeleton.solution(task),
                       # One reply per turn: a follow-up turn needs a response of its own.
                       "answer_only": [answer(task.expect.answer.value)] * (1 + len(task.followups)),
                       "wrong_answer": skeleton.solution(task)[:-1] + [answer("WRONG")]}
            for label, script in scripts.items():
                result = run_case(ScriptedModel(script), skeleton, ctx, out / "runs" / f"{ctx.id}-{label}")
                results.append({"variant": label, **result})
    summary = {"task_counts": counts, "model_inference": False}
    if check:
        for label in ("correct", "answer_only", "wrong_answer"):
            rows = [r for r in results if r["variant"] == label]
            summary[label] = {"passed": sum(r["passed"] for r in rows), "attempts": len(rows)}
        summary["faults_observed"] = sum(r["fault_encountered"] is True for r in results if r["variant"] == "correct")
        summary["gates_passed"] = (summary["correct"]["passed"] == len(entries)
                                   and summary["answer_only"]["passed"] == summary["wrong_answer"]["passed"] == 0)
        save(out / "results.json", results)
    save(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--check", action="store_true", help="Run correct and deliberately failing scripted fixtures")
    args = parser.parse_args()
    logger.remove()
    try:
        summary = generate(args.output, args.split, args.check)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    if args.check and not summary["gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
