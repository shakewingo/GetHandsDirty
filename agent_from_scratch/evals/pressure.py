"""Window-pressure suite: tasks whose evidence outgrows the context window.

The 17-task dev suite peaks near 8% of a 32k window, so it never reaches the elision
(`elide_ratio`) or summary (`compact_ratio`) triggers. Each task here builds a fresh workspace of
nine ~15,000-character files. One `read_file` costs about 3.3k tokens, so 0.6 of the usable window
is crossed on the 6th read and 0.85 on the 8th. Files come from `--seed`; nothing large is stored.

Usage: python -m agent_from_scratch.evals pressure --output DIR [--tasks a,b] [--seed N]
           [--limits '{"planning": true}'] [--n-ctx N]
"""

import argparse
from dataclasses import replace
from pathlib import Path
import random
import string
from tempfile import TemporaryDirectory

from ..agent import Agent
from ..tools.base import ToolRegistry
from ..tools.calculator import CalculatorTool
from ..tools.files import ListFilesTool, ReadFileTool
from .run import open_run, parse_limits, save
from .trajectory import profile
from .verify import metrics, snapshot, summarize, verify

FILES = 9
FILE_CHARS = 15_000  # with line numbers this stays under read_file's 16,000-character window
MAX_ITERATIONS = 16
WORDS = ("valve pump sensor cable relay boiler filter torque gasket bearing coolant panel gauge "
         "rotor piston clamp fuse switch heater duct nozzle sealant bracket coupling damper "
         "flange hinge inlet outlet spindle turbine vent wiring lubricant compressor manifold "
         "thermostat actuator conveyor hopper cylinder shaft casing baffle pulley").split()


def _filler(rng: random.Random) -> list[str]:
    lines, size = [], 0
    while size < FILE_CHARS:
        words = (rng.choice(WORDS) for _ in range(rng.randint(10, 16)))
        lines.append(" ".join(words).capitalize() + ".")
        size += len(lines[-1]) + 1
    return lines


def _code(rng: random.Random) -> str:
    return "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))


def _write(root: Path, name: str, lines: list[str]) -> None:
    (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reports(rng: random.Random, root: Path, codes: dict[int, str]) -> None:
    """Write report_01..report_09 of filler, hiding each code on a random line of its file."""
    for number in range(1, FILES + 1):
        lines = _filler(rng)
        if number in codes:
            lines.insert(rng.randrange(len(lines)), f"ACCESS CODE: {codes[number]}")
        _write(root, f"report_{number:02d}.txt", lines)


def find_one(rng: random.Random, root: Path, position: int) -> tuple[str, str]:
    """One file hides a code; the reads before it are dead weight once it is found."""
    code = _code(rng)
    _reports(rng, root, {position: code})
    return (f"The folder has report_01.txt to report_{FILES:02d}.txt. Exactly one of them contains "
            "a line that starts with 'ACCESS CODE:'. Read the reports in order until you find it, "
            "then reply with only the code."), code


def find_two(rng: random.Random, root: Path, positions: tuple[int, int]) -> tuple[str, str]:
    """Two files hide a code each, so the first must survive being evicted before the answer."""
    codes = {position: _code(rng) for position in positions}
    _reports(rng, root, codes)
    return (f"The folder has report_01.txt to report_{FILES:02d}.txt. Exactly two of them contain "
            "a line that starts with 'ACCESS CODE:'. Read the reports one at a time in order, "
            "making a single tool call per response, until you have found both. Then reply with "
            "only the two codes in file order, separated by a comma with no spaces."), ",".join(codes.values())


def follow_chain(rng: random.Random, root: Path) -> tuple[str, str]:
    """Each file names the next at random, so only the latest read is needed to continue."""
    names = [f"note_{''.join(rng.choices(string.ascii_lowercase, k=5))}.txt" for _ in range(FILES)]
    code = _code(rng)
    for index, name in enumerate(names):
        last = index + 1 == FILES
        _write(root, name, [*_filler(rng), f"FINAL ANSWER: {code}" if last
                            else f"NEXT FILE: {names[index + 1]}"])
    return (f"Start with {names[0]}. Its last line names the next file to open. Follow the chain "
            "until a file gives a FINAL ANSWER, then reply with only that answer."), code


TASKS = [
    {"id": "find_report_7", "family": "find_one", "build": find_one, "args": {"position": 7}},
    {"id": "find_report_9", "family": "find_one", "build": find_one, "args": {"position": 9}},
    {"id": "follow_chain", "family": "follow_chain", "build": follow_chain, "args": {}},
    {"id": "find_two_reports", "family": "find_two", "build": find_two, "args": {"positions": (3, 8)}},
]


def run_case(model, spec: dict, output: Path, *, seed: int, overrides: dict | None = None) -> dict:
    """Build the workspace, run one turn with read-only tools, then score the final answer."""
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="tiny-agent-pressure-") as temporary:
        workspace = Path(temporary).resolve()
        prompt, expected = spec["build"](random.Random(f"{seed}:{spec['id']}"), workspace, **spec["args"])
        task = {"id": spec["id"], "family": spec["family"], "prompt": prompt,
                "expected": {"text": expected}, "verifier": "answer", "allowed_changes": [],
                "required_tools": ["read_file"], "max_iterations": MAX_ITERATIONS}
        save(output / "task.json", task)
        before = snapshot(workspace)
        registry = ToolRegistry([CalculatorTool(),
                                 ListFilesTool(workspace, restrict_to_workspace=True),
                                 ReadFileTool(workspace, restrict_to_workspace=True)])
        agent = Agent(model, str(output / "state"), registry=registry)
        agent.limits = replace(agent.limits, **{**(overrides or {}), "max_iterations": MAX_ITERATIONS})
        result = agent.run_turn(prompt, session_id=spec["id"])
        # `passed` needs the exact format; `answer_found` separates a format slip from lost evidence.
        record = {"id": spec["id"], "family": spec["family"], **metrics(result),
                  **verify(task, result, workspace, workspace, before),
                  "answer_found": all(part in (result.final_answer or "") for part in expected.split(","))}
        save(output / "result.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory outside the package")
    parser.add_argument("--tasks", help=f"Comma-separated task IDs; default: {','.join(t['id'] for t in TASKS)}")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limits", default="{}", help='JSON AgentLimits overrides, e.g. \'{"planning": true}\'')
    parser.add_argument("--n-ctx", type=int, help="Override the model window; default config.N_CTX")
    args = parser.parse_args()
    overrides = parse_limits(parser, args.limits)
    specs = TASKS
    if args.tasks:
        selected = args.tasks.split(",")
        if len(selected) != len(set(selected)) or not set(selected) <= {t["id"] for t in TASKS}:
            parser.error("Select unique existing task IDs.")
        specs = [t for t in TASKS if t["id"] in selected]
    out = args.output.resolve()
    model = open_run(out, parser, suite="pressure-v2", seed=args.seed, overrides=overrides,
                     n_ctx=args.n_ctx, selected_tasks=[t["id"] for t in specs],
                     files=FILES, file_chars=FILE_CHARS)
    records = []
    for spec in specs:
        print("START", spec["id"], flush=True)
        record = run_case(model, spec, out / spec["id"], seed=args.seed, overrides=overrides)
        records.append(record)
        save(out / "results.json", records)
        save(out / "summary.json", summarize(records))
        print("RESULT", spec["id"], "PASS" if record["passed"] else "FAIL", record["stop_reason"],
              "requests", record["model_requests"], "peak", record["peak_context_ratio"], flush=True)
        if record["stop_reason"] == "interrupted":
            break
    save(out / "trajectory.json", profile(records))


if __name__ == "__main__":
    main()
