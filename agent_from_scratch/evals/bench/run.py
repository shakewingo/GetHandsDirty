"""Build Stage 8 benchmark tasks from seeds, run them, and score them.

Usage: python -m agent_from_scratch.evals bench --output DIR --split {dev,test} [--tasks a,b]
           [--seed N] [--limits '{"planning": true}'] [--final] [--allow-drift]
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

from .faults import FaultyTool
from .skeletons import SKELETONS, Skeleton
from .spec import BuildContext, Task
from .verify import bench_summary, check
from ..run import open_run, parse_limits, save
from ..trajectory import profile
from ..verify import metrics, snapshot
from ...agent import Agent
from ...tools.base import ToolRegistry
from ...tools.calculator import CalculatorTool
from ...tools.files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from ...tools.search import GlobFilesTool, GrepTextTool
from ...tools.shell import Command, ShellTool

HERE = Path(__file__).resolve().parent
CHECK_SCRIPT = HERE.parent / "check.py"


def specs(split: str | None = None) -> list[tuple[Skeleton, BuildContext]]:
    """(skeleton, BuildContext) pairs in registration order; touches no disk."""
    entries = []
    for skeleton in SKELETONS:
        if split is not None and skeleton.split != split:
            continue
        for seed in skeleton.seeds:
            for condition in (("clean", "fault") if skeleton.recovery else (None,)):
                entries.append((skeleton, BuildContext(skeleton.name, skeleton.family,
                                                        skeleton.split, seed, condition)))
    return entries


def registry_for(task: Task, workspace: Path, private: Path) -> ToolRegistry:
    """The confined general registry, narrowed to `task.tools`, with fault/shell wiring."""
    available = {"calculator": CalculatorTool(),
                "list_files": ListFilesTool(workspace, restrict_to_workspace=True),
                "glob_files": GlobFilesTool(workspace, restrict_to_workspace=True),
                "grep_text": GrepTextTool(workspace, restrict_to_workspace=True),
                "read_file": ReadFileTool(workspace, restrict_to_workspace=True),
                "write_file": WriteFileTool(workspace, restrict_to_workspace=True),
                "edit_file": EditFileTool(workspace, restrict_to_workspace=True)}
    if "shell" in task.tools:
        spec_path = private / "check-spec.json"
        spec_path.write_text(json.dumps({"path": "config.json", "value": task.debug["correct"]}))
        available["shell"] = ShellTool(workspace, {"check_fixture": Command(
            (sys.executable, "-I", str(CHECK_SCRIPT), str(spec_path)),
            "Check whether config.json matches the required configuration.")})
    if task.fault is not None:
        available[task.fault.tool] = FaultyTool(available[task.fault.tool], task.fault.on_call,
                                                task.fault.message)
    return ToolRegistry(available[name] for name in task.tools)


def run_case(model, skeleton: Skeleton, ctx: BuildContext, output: Path, *,
            overrides: dict | None = None) -> dict:
    """Build the workspace from `ctx`'s seed, run one turn, then score it. No history is carried."""
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="tiny-agent-bench-") as temporary:
        private = Path(temporary).resolve()
        workspace = private / "workspace"
        workspace.mkdir()
        rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
        task = skeleton.build(rng, workspace, ctx)
        save(output / "task.json", asdict(task))
        before = snapshot(workspace)
        registry = registry_for(task, workspace, private)
        save(output / "schemas.json", registry.schemas())
        agent = Agent(model, str(output / "state"), registry=registry)
        agent.limits = replace(agent.limits, **{**(overrides or {}), "max_iterations": task.max_iterations})
        result = agent.run_turn(task.prompt, session_id=task.id)
        score = check(task, result, workspace, before)
        record = {"id": task.id, "skeleton": task.skeleton, "family": task.family,
                 "split": task.split, "pair_id": task.pair_id, "condition": task.condition,
                 **metrics(result), **score}
        save(output / "result.json", record)
        return record


def main():
    from .manifest import manifest_drift
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--tasks", help="Comma-separated task IDs; default: every task in --split")
    parser.add_argument("--seed", type=int, default=11, help="Decoding seed; workspaces are seeded separately")
    parser.add_argument("--limits", default="{}", help='JSON AgentLimits overrides')
    parser.add_argument("--final", action="store_true", help="Required to run --split test")
    parser.add_argument("--allow-drift", action="store_true", help="Run --final despite manifest drift")
    args = parser.parse_args()
    overrides = parse_limits(parser, args.limits)
    if args.split == "test" and not args.final:
        parser.error("Running the test split needs --final (see docs/STAGE8_DESIGN.md).")
    entries = specs(args.split)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        entries = [(s, c) for s, c in entries if c.id in wanted]
        if len(entries) != len(wanted):
            parser.error("Unknown task ID in --tasks.")
    drift = manifest_drift() if args.final else {}
    if drift and not args.allow_drift:
        parser.error(f"Manifest drift in {sorted(drift)}; rerun with --allow-drift if intended.")
    out = args.output.resolve()
    model = open_run(out, parser, suite=f"bench-{args.split}-v1", seed=args.seed, overrides=overrides,
                     split=args.split, final=args.final, manifest_drift=drift,
                     selected_tasks=[c.id for _, c in entries])
    records = []
    for skeleton, ctx in entries:
        print("START", ctx.id, flush=True)
        record = run_case(model, skeleton, ctx, out / ctx.id, overrides=overrides)
        records.append(record)
        save(out / "results.json", records)
        save(out / "summary.json", bench_summary(records))
        print("RESULT", ctx.id, "PASS" if record["passed"] else "FAIL", record["stop_reason"],
              "requests", record["model_requests"], flush=True)
        if record["stop_reason"] == "interrupted":
            break
    save(out / "trajectory.json", profile(records))


if __name__ == "__main__":
    main()
