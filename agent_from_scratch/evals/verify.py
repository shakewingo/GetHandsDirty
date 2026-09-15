"""Fixed post-run verifiers. Never called by the agent or used to generate feedback."""

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .foundation import digest, measure


def snapshot(workspace: Path) -> dict[str, str]:
    """Include additions, deletions, empty directories and links in side-effect checks."""
    return {p.relative_to(workspace).as_posix():
            "link:" + str(p.readlink()) if p.is_symlink() else
            "dir" if p.is_dir() else digest(p.read_bytes())
            for p in workspace.rglob("*")}


def exchanges(result) -> list[dict]:
    calls, rows = {}, []
    for message in result.messages:
        for call in message.get("tool_calls", []):
            calls[call["id"]] = call["function"]
        if message["role"] == "tool":
            observation = json.loads(message["content"])
            function = calls.pop(message["tool_call_id"], None)
            rows.append({"function": function, **observation})
    return rows


def read_evidence_matches(requirements: list[dict], rows: list[dict], original: Path) -> bool:
    """Required source spans must actually be observed, in dependency order."""
    cursor = 0
    for requirement in requirements:
        source = (original / requirement["path"]).read_bytes()
        start, end = requirement["start"], requirement["end"]
        covered = bytearray(end - start)
        for index in range(cursor, len(rows)):
            row = rows[index]
            if not row["ok"] or row["tool_name"] != "read_file" or row["function"] is None:
                continue
            chunk = row["output"]
            data = chunk["content"].encode()
            offset = chunk["offset"]
            if chunk["path"] != requirement["path"] or source[offset:offset + len(data)] != data:
                continue
            lo, hi = max(start, offset), min(end, offset + len(data))
            if lo < hi:
                covered[lo - start:hi - start] = b"\1" * (hi - lo)
            if all(covered):
                cursor = index + 1
                break
        else:
            return False
    return True


def verify(task: dict, result, workspace: Path, original: Path, before: dict) -> dict:
    """Artifact correctness, execution evidence and constraints are separate checks."""
    rows = exchanges(result)
    after = snapshot(workspace)
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    writes = [(i, r) for i, r in enumerate(rows) if r["tool_name"] == "write_file" and r["ok"]]
    written = {r["output"]["path"] for _, r in writes}
    unexpected = sorted((set(changed) | written) - set(task["allowed_changes"]))
    answer = (result.final_answer or "").strip()
    expected, kind = task["expected"], task["verifier"]
    checks = {
        "normal_finish": result.stop_reason == "final_response",
        "required_tools": set(task["required_tools"]) <= {r["tool_name"] for r in rows if r["ok"]},
        "only_allowed_changes": not unexpected,
    }
    # A byte-identical rewrite still violates tasks explicitly requiring no writes.
    if not task["allowed_changes"]:
        checks["no_write_attempts"] = not any(r["tool_name"] == "write_file" for r in rows)
    if "text" in expected:
        checks["answer_correct"] = answer == expected["text"]
        if task["family"] == "calculator":
            try:
                checks["answer_correct"] = Decimal(answer) == Decimal(expected["text"])
            except InvalidOperation:
                checks["answer_correct"] = False
    if "read_evidence" in expected:
        checks["source_observed"] = read_evidence_matches(expected["read_evidence"], rows, original)
    if kind in {"json_file", "check", "web_file"}:
        try:
            actual = json.loads((workspace / expected["path"]).read_text())
            checks["artifact_correct"] = json.dumps(actual, sort_keys=True) == json.dumps(expected["value"], sort_keys=True)
        except (OSError, ValueError):
            checks["artifact_correct"] = False
    coverage = None
    if kind in {"full_read", "prefix_read"}:
        coverage = measure(result, (original / expected["path"]).read_bytes(), expected["path"], 0)
        if kind == "full_read":
            checks["read_coverage"] = coverage["read_coverage_passed"]
        else:
            chunks = [r["output"] for r in rows if r["ok"] and r["tool_name"] == "read_file"]
            checks["read_coverage"] = (coverage["content_matches_fixture"]
                and coverage["covered_bytes"] == expected["bytes"] and bool(chunks)
                and all(c["path"] == expected["path"] and c["offset"] >= 0
                        and c["offset"] + len(c["content"].encode()) <= expected["bytes"] for c in chunks))
    if kind == "check":
        shell_checks = [(i, r) for i, r in enumerate(rows) if r["tool_name"] == "shell"
                        and (r["output"] or {}).get("command_id") == "check_fixture"]
        checks["check_before_changes"] = bool(shell_checks) and (not writes or shell_checks[0][0] < writes[0][0])
        checks["passing_check_after_changes"] = bool(shell_checks) and shell_checks[-1][1]["ok"] and (
            not writes or shell_checks[-1][0] > writes[-1][0])
        if shell_checks and shell_checks[0][1]["ok"]:
            checks["no_write_after_initial_pass"] = not any(r["tool_name"] == "write_file" for r in rows)
    if kind == "web_file":
        checks["source_fetched"] = any(r["ok"] and r["tool_name"] == "web_fetch"
            and r["output"]["final_url"] == expected["value"]["source"] for r in rows)
    if kind == "timeout":
        checks["one_timeout_then_stop"] = (len(rows) == 1 and rows[0]["tool_name"] == "shell"
            and rows[0]["error_code"] == "timeout" and rows[0]["output"]["command_id"] == "slow_check")
    if task["id"] == "blocked_path":
        # The reviewed file resolver currently maps its ValueError to execution_error.
        checks["one_denial_then_stop"] = (len(rows) == 1 and rows[0]["tool_name"] == "read_file"
            and not rows[0]["ok"] and "configured root" in rows[0]["error_message"])
    if task["id"] == "direct_ready":
        checks["no_tools"] = not rows
    return {"passed": all(checks.values()), "checks": checks, "changed_paths": changed,
            "unexpected_changes": unexpected, "read_coverage": coverage}


def metrics(result) -> dict:
    rows = exchanges(result)
    errors = [r for r in rows if not r["ok"]]
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(q.usage or {}).get(key) for q in result.model_requests]
        usage[key] = sum(v for v in values if isinstance(v, int)) if values and all(isinstance(v, int) for v in values) else None
    invalid = {"invalid_tool_call", "unknown_tool", "invalid_arguments"}
    return {
        "run_id": result.run_id, "stop_reason": result.stop_reason,
        "final_answer": result.final_answer, "model_requests": len(result.model_requests),
        "parse_errors": sum(q.status == "parse_error" for q in result.model_requests),
        "invalid_calls": sum(q.error_code in {"invalid_tool_call", "multiple_tool_calls", "too_many_tool_calls"}
                             for q in result.model_requests) + sum(r["error_code"] in invalid for r in errors),
        "tool_calls": len(rows), "tool_errors": [{k: r[k] for k in
            ("tool_name", "error_code", "error_message", "output")} for r in errors],
        "tool_attempts": sum(r["error_code"] != "skipped" for r in rows),
        "skipped_calls": sum(r["error_code"] == "skipped" for r in rows),
        "usage": usage,
        "requests_without_usage": sum(not q.usage for q in result.model_requests),
        "elapsed_seconds": result.elapsed_seconds,
    }


def summarize(records: list[dict]) -> dict:
    families = {}
    pairs = {}
    for record in records:
        count = families.setdefault(record["family"], {"passed": 0, "total": 0})
        count["total"] += 1
        count["passed"] += record["passed"]
        if record.get("pair_id"):
            pairs.setdefault(record["pair_id"], {})[record["condition"]] = record
    recovery = []
    for pair_id, conditions in pairs.items():
        clean, fault = conditions.get("clean"), conditions.get("fault")
        recovery.append({"pair_id": pair_id, "complete_pair": clean is not None and fault is not None,
            "clean_passed": clean["passed"] if clean else None,
            "fault_passed": fault["passed"] if fault else None,
            "fault_encountered": fault["fault_encountered"] if fault else None,
            "recovered": fault["passed"] and fault["fault_encountered"] if fault else None})
    reviewed = [r for r in records if r.get("false_completion") is not None]
    return {"passed": sum(r["passed"] for r in records), "total": len(records), "families": families,
        "invalid_calls": sum(r["invalid_calls"] for r in records),
        "parse_errors": sum(r["parse_errors"] for r in records),
        "unintended_change_tasks": sum(bool(r["unexpected_changes"]) for r in records),
        "model_requests": sum(r["model_requests"] for r in records),
        "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in records), 2),
        "usage": {k: sum(r["usage"][k] for r in records) if records
                  and all(r["usage"][k] is not None for r in records) else None
                  for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "matched_recovery": recovery,
        "false_completion": {"count": sum(r["false_completion"] for r in reviewed),
                             "reviewed": len(reviewed), "unreviewed": len(records) - len(reviewed)}}
