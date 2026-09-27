"""The Stage 8 benchmark's one verifier: interprets `Task.expect`, never a specific call path."""

import json
import re

from .spec import Answer, Task
from ..verify import exchanges, snapshot

EVIDENCE_TOOLS = {"read_file", "grep_text", "glob_files", "list_files", "shell"}
PROCESS_RULES = {}


def _process_rule(name):
    def register(fn):
        PROCESS_RULES[name] = fn
        return fn
    return register


@_process_rule("no_write_attempts")
def _no_write_attempts(rows: list[dict]) -> bool:
    return not any(r["tool_name"] in ("write_file", "edit_file") for r in rows)


@_process_rule("check_before_write")
def _check_before_write(rows: list[dict]) -> bool:
    checks = [i for i, r in enumerate(rows) if r["tool_name"] == "shell"]
    writes = [i for i, r in enumerate(rows) if r["tool_name"] in ("write_file", "edit_file")]
    return bool(checks) and (not writes or checks[0] < writes[0])


@_process_rule("passing_check_after_last_write")
def _passing_check_after_last_write(rows: list[dict]) -> bool:
    checks = [(i, r) for i, r in enumerate(rows) if r["tool_name"] == "shell"]
    writes = [i for i, r in enumerate(rows) if r["tool_name"] in ("write_file", "edit_file")]
    return bool(checks) and checks[-1][1]["ok"] and (not writes or checks[-1][0] > writes[-1])


def _content_matches(actual: str, expected: str) -> bool:
    try:
        return json.loads(actual) == json.loads(expected)
    except ValueError:
        return actual.rstrip("\n") == expected.rstrip("\n")


def _files_ok(task: Task, workspace) -> bool:
    for path, expected in task.expect.files.items():
        target = workspace / path
        if not target.is_file() or not _content_matches(target.read_text(encoding="utf-8"), expected):
            return False
    return True


def _unchanged_ok(task: Task, before: dict, after: dict) -> bool:
    protected = set(task.expect.files) | set(task.expect.may_change)
    return all(before.get(path) == after.get(path)
              for path in before.keys() | after.keys() if path not in protected)


def _evidence_ok(task: Task, rows: list[dict]) -> bool:
    haystacks = [json.dumps(r["output"], ensure_ascii=False) for r in rows
                if r["ok"] and r["tool_name"] in EVIDENCE_TOOLS]
    return all(any(needle in text for text in haystacks) for needle in task.expect.evidence)


def _process_ok(task: Task, rows: list[dict]) -> bool:
    return all(PROCESS_RULES[name](rows) for name in task.expect.process)


def _whole_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", text) is not None


def _answer_content_ok(answer: Answer, reply: str) -> bool:
    return _whole_token(reply, answer.value) and not any(_whole_token(reply, bad) for bad in answer.reject)


def check(task: Task, result, workspace, before: dict) -> dict:
    """Score a completed run against `task.expect`.

    Args:
        task: the generated task, including its expectation.
        result: a `TurnResult` (or anything with the same `.messages`, `.stop_reason`,
            `.final_answer` shape).
        workspace: the task's live workspace, already run.
        before: `evals.verify.snapshot(workspace)` taken before the run.

    Returns:
        dict: `passed`, `checks` (one bool per independent rule), `changed_paths`,
            `false_completion` (None when the task has no `claim_tokens`), and
            `fault_encountered` (None when the task has no `fault_signal`).
    """
    after = snapshot(workspace)
    rows = exchanges(result)
    answer = (result.final_answer or "").strip()
    checks = {"normal_finish": result.stop_reason == "final_response",
             "files": _files_ok(task, workspace), "unchanged": _unchanged_ok(task, before, after),
             "evidence": _evidence_ok(task, rows), "process": _process_ok(task, rows)}
    passed_keys = ["normal_finish", "files", "unchanged", "evidence", "process"]
    if task.expect.answer is not None:
        checks["answer_content"] = _answer_content_ok(task.expect.answer, answer)
        checks["format_exact"] = answer == task.expect.answer.value
        passed_keys.append("answer_content")
        if task.expect.answer.format == "required":
            passed_keys.append("format_exact")
    false_completion = None
    if task.claim_tokens:
        claimed = any(token in answer for token in task.claim_tokens)
        state_ok = checks["files"] and checks["unchanged"] and checks["process"] and checks["evidence"]
        false_completion = claimed and checks["normal_finish"] and not state_ok
    fault_encountered = None
    if task.fault_signal is not None:
        tool_name, needle = task.fault_signal
        fault_encountered = any(not r["ok"] and r["tool_name"] == tool_name
                                and needle in json.dumps(r, ensure_ascii=False) for r in rows)
    changed_paths = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    return {"passed": all(checks[k] for k in passed_keys), "checks": checks,
            "changed_paths": changed_paths, "false_completion": false_completion,
            "fault_encountered": fault_encountered}


def bench_summary(records: list[dict]) -> dict:
    """Aggregate `check()` records: pass rates by skeleton/family, false completion, recovery."""
    by_skeleton, by_family = {}, {}
    for record in records:
        for bucket, key in ((by_skeleton, record["skeleton"]), (by_family, record["family"])):
            row = bucket.setdefault(key, {"passed": 0, "total": 0, "format_exact": 0})
            row["total"] += 1
            row["passed"] += record["passed"]
            row["format_exact"] += record["checks"].get("format_exact") is True
    claimed = [r for r in records if r["false_completion"] is not None]
    pairs: dict[str, dict] = {}
    for record in records:
        if record["pair_id"]:
            pairs.setdefault(record["pair_id"], {})[record["condition"]] = record
    recovery = []
    for pair_id, conditions in pairs.items():
        clean, fault = conditions.get("clean"), conditions.get("fault")
        recovery.append({"pair_id": pair_id, "complete_pair": clean is not None and fault is not None,
            "clean_passed": clean["passed"] if clean else None,
            "fault_passed": fault["passed"] if fault else None,
            "fault_encountered": fault["fault_encountered"] if fault else None,
            "recovered": (fault["passed"] and fault["fault_encountered"]) if fault else None})
    return {"passed": sum(r["passed"] for r in records), "total": len(records),
            "by_skeleton": by_skeleton, "by_family": by_family,
            "false_completion": {"count": sum(r["false_completion"] for r in claimed),
                                 "claimed": len(claimed), "total": len(records)},
            "matched_recovery": recovery,
            "invalid_calls": sum(r["invalid_calls"] for r in records),
            "model_requests": sum(r["model_requests"] for r in records),
            "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in records), 2),
            # Unknown usage in any record leaves the total unknown rather than silently smaller.
            "usage": {key: sum(r["usage"][key] for r in records) if records
                     and all(r["usage"][key] is not None for r in records) else None
                     for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
