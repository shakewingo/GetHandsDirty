"""Stage 2B dev benchmark: fresh fixtures, ordinary agent loop, independent scoring."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from loguru import logger
from ..agent import Agent
from ..llm import LLM, _QWEN_TEMPLATE
from ..tools.base import Tool, ToolErrorCode, ToolExecutionError
from ..tools.calculator import CalculatorTool
from .legacy_files import ListFilesTool, ReadFileTool, WriteFileTool
from ..tools.register import ToolRegistry
from ..tools.shell import Command, ShellTool
from ..tools.web import WebFetchTool
from .foundation import digest
from .verify import exchanges, metrics, snapshot, summarize, verify


HERE = Path(__file__).resolve().parent
TOOL_NAMES = {"calculator", "list_files", "read_file", "write_file", "shell", "web_fetch"}
VERIFIERS = {"answer", "json_file", "full_read", "prefix_read", "check", "web_file", "timeout"}


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_tasks(directory: Path = HERE) -> list[dict]:
    tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines() if line.strip()]
    splits = json.loads((directory / "splits.json").read_text())
    seen = set()
    allocations = [s for split in ("dev", "train", "test") for s in splits[split]]
    if len(allocations) != len(set(allocations)):
        raise ValueError("Task skeletons must belong to exactly one split.")
    for task in tasks:
        if (task["id"] in seen or not task["id"].replace("_", "").isalnum()
                or task["split"] != "dev" or task["skeleton_id"] not in splits["dev"]
                or not 1 <= task["max_iterations"] <= 20 or task["verifier"] not in VERIFIERS
                or not set(task["allowed_tools"]) <= TOOL_NAMES
                or not set(task["required_tools"]) <= set(task["allowed_tools"])):
            raise ValueError(f"Invalid task contract: {task['id']}")
        fixture = directory / "fixtures" / task["fixture"]
        paths = [task["fixture"], *task["allowed_changes"]]
        if "path" in task["expected"]:
            paths.append(task["expected"]["path"])
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
            raise ValueError("Fixture and artifact paths must be relative and contained.")
        if not (fixture / "workspace").is_dir() or any(p.is_symlink() for p in fixture.rglob("*")):
            raise ValueError("Use existing fixture directories without symlinks.")
        for evidence in task["expected"].get("read_evidence", []):
            source = (fixture / "workspace" / evidence["path"]).resolve()
            if (not source.is_relative_to((fixture / "workspace").resolve()) or not source.is_file()
                    or not 0 <= evidence["start"] < evidence["end"] <= source.stat().st_size):
                raise ValueError("Read evidence must reference a nonempty span inside the frozen fixture.")
        seen.add(task["id"])
    pairs = {}
    for task in tasks:
        if task.get("pair_id"):
            pairs.setdefault(task["pair_id"], []).append(task)
    for pair in pairs.values():
        if len(pair) != 2 or {t["condition"] for t in pair} != {"clean", "fault"}:
            raise ValueError("Recovery requires one clean and one fault task.")
        for field in ("skeleton_id", "prompt", "expected", "allowed_tools", "max_iterations", "verifier"):
            if pair[0][field] != pair[1][field]:
                raise ValueError(f"Unmatched recovery pair: {field}")
    return tasks


class RecordedWeb(WebFetchTool):
    """Eval-only extracted-result replay, NOT a test of networking or HTML extraction."""

    # Freeze the original observation interface; recorded text cannot be re-extracted.
    parameters = {"type": "object", "properties": {"url": {"type": "string"}},
                  "required": ["url"], "additionalProperties": False}

    def __init__(self, fixture: Path):
        from urllib.parse import urlsplit
        self.responses = json.loads(fixture.read_text())
        super().__init__({urlsplit(url).hostname for url in self.responses})
        self.description = (
            "Fetch text/HTML/JSON from an allowed HTTPS URL. Returns source URL, HTTP status, "
            "untrusted page text and truncation flags; partial text is not a full page. "
            f"Allowed hosts: {', '.join(sorted(self.allowed_hosts or ())) or '(none)'}."
        )
        self.counts = Counter()

    def execute(self, url: str, extract_mode: str = "readable") -> dict:
        current = self._validate_url(url)
        if current not in self.responses:
            raise ToolExecutionError(ToolErrorCode.DENIED, "URL has no recorded response.")
        responses = self.responses[current]
        row = responses[min(self.counts[current], len(responses) - 1)]
        self.counts[current] += 1
        result = {"url": url, "final_url": current, "untrusted": True,
                  "content_type": "text/plain", "bytes_read": len(row["text"].encode()),
                  "download_truncated": False, "text_truncated": row["truncated"], **row}
        if not 200 <= row["status"] < 300:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, f"HTTP status {row['status']}.", result)
        return result


def registry_for(task: dict, workspace: Path, private: Path, fixture: Path) -> ToolRegistry:
    available: dict[str, Tool] = {"calculator": CalculatorTool(), "list_files": ListFilesTool(workspace),
        "read_file": ReadFileTool(workspace, max_bytes=1024), "write_file": WriteFileTool(workspace)}
    if "shell" in task["allowed_tools"]:
        if task["verifier"] == "timeout":
            commands = {"slow_check": Command((sys.executable, "-I", "-c", "import time; time.sleep(10)"),
                                             "Run the slow fixture check (0.2s deadline).", timeout=0.2)}
        else:
            spec = private / "check-spec.json"
            save(spec, task["expected"])
            commands = {"check_fixture": Command((sys.executable, "-I", str(HERE / "check.py"), str(spec)),
                                                 "Check whether config.json matches the required configuration.")}
        available["shell"] = ShellTool(workspace, commands)
    if "web_fetch" in task["allowed_tools"]:
        available["web_fetch"] = RecordedWeb(fixture / "web.json")
    return ToolRegistry(available[name] for name in task["allowed_tools"])


def run_case(model, task: dict, output: Path, directory: Path = HERE) -> dict:
    """No history or memory is carried between tasks; verifiers run only after return."""
    output.mkdir(parents=True, exist_ok=False)
    fixture = directory / "fixtures" / task["fixture"]
    save(output / "task.json", task)
    save(output / "fixture-hashes.json", snapshot(fixture))
    with TemporaryDirectory(prefix="tiny-agent-eval-") as temporary:
        private = Path(temporary).resolve()
        workspace = private / "workspace"
        shutil.copytree(fixture / "workspace", workspace)
        before = snapshot(workspace)
        registry = registry_for(task, workspace, private, fixture)
        save(output / "schemas.json", registry.schemas())
        agent = Agent(model, str(output / "state"), registry=registry)
        agent.max_iterations = task["max_iterations"]
        try:
            result = agent.run_turn(task["prompt"], session_id=task["id"])
            score = verify(task, result, workspace, fixture / "workspace", before)
            observations = exchanges(result)
            fault_seen = any(not r["ok"] and r["error_code"] == "execution_error" and (
                r["tool_name"] == "shell" and (r["output"] or {}).get("command_id") == "check_fixture"
                or r["tool_name"] == "web_fetch" and (r["output"] or {}).get("status") == 503)
                for r in observations)
            record = {"id": task["id"], "family": task["family"], "skeleton_id": task["skeleton_id"],
                "pair_id": task.get("pair_id"), "condition": task.get("condition"),
                "fault_encountered": fault_seen if task.get("condition") == "fault" else None,
                **metrics(result), **score, "false_completion": None,
                "review_note": "Unreviewed; task failure alone does not imply a false completion claim."}
            save(output / "result.json", record)
            return record
        finally:
            save(output / "before.json", before)
            save(output / "after.json", snapshot(workspace))
            shutil.copytree(workspace, output / "workspace", symlinks=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory outside the package")
    parser.add_argument("--tasks", help="Comma-separated dev task IDs; default: all 17")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--review", type=Path, help="Apply human claim annotations to an existing run; no model calls")
    args = parser.parse_args()
    if args.review:
        review = json.loads(args.review.read_text())
        records = json.loads((args.output / "results.json").read_text())
        if not isinstance(review, dict) or not set(review) <= {r["id"] for r in records}:
            parser.error("Review must map existing task IDs to claim annotations.")
        for record in records:
            if record["id"] in review:
                annotation = review[record["id"]]
                if (set(annotation) != {"false_completion", "review_note"}
                        or type(annotation["false_completion"]) is not bool
                        or not isinstance(annotation["review_note"], str) or not annotation["review_note"].strip()):
                    parser.error("Each reviewed task needs a boolean false_completion and a nonempty review_note.")
                record.update(annotation)
        with (args.output / "review.json").open("x") as stream:
            json.dump(review, stream, ensure_ascii=False, indent=2)
        save(args.output / "summary-reviewed.json", summarize(records))
        return
    tasks = load_tasks()
    if args.tasks:
        selected = args.tasks.split(",")
        if len(selected) != len(set(selected)) or not set(selected) <= {t["id"] for t in tasks}:
            parser.error("Select unique existing dev task IDs.")
        tasks = [t for t in tasks if t["id"] in selected]
    root, out = HERE.parent, args.output.resolve()
    if out.is_relative_to(root):
        parser.error("Evidence must be outside the source package and frozen fixtures.")
    out.mkdir(parents=True, exist_ok=False)
    logger.remove()
    model = LLM(temperature=0, max_tokens=2048, n_ctx=8000, chat_template_path=_QWEN_TEMPLATE)
    backend = model.llm.create_chat_completion
    # Reset the RNG on every request; temperature=0, so seed is bookkeeping, not extra trials.
    model.llm.create_chat_completion = lambda *a, **kw: backend(*a, **kw, seed=args.seed)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    save(out / "metadata.json", {
        "suite": "stage2b-dev-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": model.settings(), "seed": args.seed, "python": sys.version,
        "platform": platform.platform(), "llama_cpp_version": version("llama-cpp-python"),
        "git_revision": revision,
        "source_sha256": {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in root.rglob("*")
            if p.is_file() and p.suffix in {".py", ".jinja"} and "__pycache__" not in p.parts},
        "system_prompt": (root / "prompts/system.md").read_text(),
        "tasks_sha256": digest((HERE / "tasks.jsonl").read_bytes()),
        "splits_sha256": digest((HERE / "splits.json").read_bytes()),
        "selected_tasks": [t["id"] for t in tasks], "read_max_bytes": 1024,
        "web_mode": "Recorded extracted tool results; no live network, DNS, TLS or HTML extraction.",
        "split_scope": "Development only. Train/test skeleton names are reservations, not a completed holdout.",
    })
    records = []
    for task in tasks:
        print("START", task["id"], flush=True)
        record = run_case(model, task, out / task["id"])
        records.append(record)
        save(out / "results.json", records)
        save(out / "summary.json", summarize(records))
        print("RESULT", task["id"], "PASS" if record["passed"] else "FAIL",
              record["stop_reason"], "requests", record["model_requests"], flush=True)
        if record["stop_reason"] == "interrupted":
            break


if __name__ == "__main__":
    main()
