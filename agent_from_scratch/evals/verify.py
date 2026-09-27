"""Fixed post-run verifiers. Never called by the agent or used to generate feedback."""

import hashlib
import json
from pathlib import Path

from ..context import window_share_ratio


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def verify(task: dict, result, workspace: Path, before: dict) -> dict:
    """Answer correctness and side-effect constraints are separate checks (used by `pressure`)."""
    rows = exchanges(result)
    after = snapshot(workspace)
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    written = {r["output"]["path"] for r in rows if r["tool_name"] == "write_file" and r["ok"]}
    unexpected = sorted((set(changed) | written) - set(task["allowed_changes"]))
    checks = {
        "normal_finish": result.stop_reason == "final_response",
        "required_tools": set(task["required_tools"]) <= {r["tool_name"] for r in rows if r["ok"]},
        "only_allowed_changes": not unexpected,
    }
    # A byte-identical rewrite still violates tasks explicitly requiring no writes.
    if not task["allowed_changes"]:
        checks["no_write_attempts"] = not any(r["tool_name"] == "write_file" for r in rows)
    if "text" in task["expected"]:
        checks["answer_correct"] = (result.final_answer or "").strip() == task["expected"]["text"]
    return {"passed": all(checks.values()), "checks": checks, "changed_paths": changed,
            "unexpected_changes": unexpected}


# Tool categories for trajectory labels; deliberately not coding stages.
ACTION_KINDS = {"list_files": "explore", "read_file": "explore", "glob_files": "explore",
                "grep_text": "explore", "web_fetch": "explore", "web_search": "explore",
                "write_file": "modify", "edit_file": "modify", "shell": "execute",
                "calculator": "execute", "update_plan": "plan"}
ACTION_PRIORITY = ("modify", "execute", "explore", "plan", "other")


def metrics(result) -> dict:
    requests = [q for q in result.model_requests if q.status != "blocked"]
    actors = [q for q in requests if q.purpose == "agent"]
    summaries = [q for q in requests if q.purpose == "compact"]
    rows = exchanges(result)
    errors = [r for r in rows if not r["ok"]]
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        # No calls means zero usage; an actual call with missing usage stays unknown.
        total = 0
        for request in requests:
            value = (request.usage or {}).get(key)
            if not isinstance(value, int):
                total = None
                break
            total += value
        usage[key] = total
    invalid = {"invalid_tool_call", "unknown_tool", "invalid_arguments"}
    # One label per actor request: its most consequential call's category.
    names = {call["id"]: call["function"]["name"]
             for m in result.messages for call in m.get("tool_calls", [])}
    actions = []
    for request in actors:
        kinds = {ACTION_KINDS.get(names.get(call_id, ""), "other") for call_id in request.call_ids}
        actions.append("error" if request.status != "completed" else
                       next((k for k in ACTION_PRIORITY if k in kinds), "answer"))
    shares = [share for q in actors if (share := window_share_ratio(q.budget)) is not None]
    return {
        "run_id": result.run_id, "stop_reason": result.stop_reason,
        "final_answer": result.final_answer, "model_requests": len(requests),
        "actor_requests": len(actors),
        "compact_requests": len(summaries),
        # A published summary records its measured result and no error; a rejected one does not.
        "compactions_applied": sum(q.compact_after is not None and q.error_message is None
                                   for q in summaries),
        "max_actor_prompt_tokens": max(((q.budget or {}).get("prompt_tokens") or 0
                                        for q in actors), default=0),
        "parse_errors": sum(q.status == "parse_error" for q in requests),
        "invalid_calls": sum(q.error_code in {"invalid_tool_call", "multiple_tool_calls", "too_many_tool_calls"}
                             for q in requests) + sum(r["error_code"] in invalid for r in errors),
        "tool_calls": len(rows), "tool_errors": [{k: r[k] for k in
            ("tool_name", "error_code", "error_message", "output")} for r in errors],
        "tool_attempts": sum(r["error_code"] != "skipped" for r in rows),
        "skipped_calls": sum(r["error_code"] == "skipped" for r in rows),
        "usage": usage,
        "requests_without_usage": sum(not q.usage for q in requests),
        "elapsed_seconds": result.elapsed_seconds,
        "peak_context_ratio": round(max(shares), 4) if shares else None,
        "elided_messages": max((q.elided_messages for q in actors), default=0),
        "plan_updates": sum(r["tool_name"] == "update_plan" and r["ok"] for r in rows),
        "stuck_reminders": result.stuck_reminders,
        "actions": actions,
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
        "compact_requests": sum(r["compact_requests"] for r in records),
        "compactions_applied": sum(r["compactions_applied"] for r in records),
        "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in records), 2),
        "usage": {k: sum(r["usage"][k] for r in records) if records
                  and all(r["usage"][k] is not None for r in records) else None
                  for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "matched_recovery": recovery,
        "false_completion": {"count": sum(r["false_completion"] for r in reviewed),
                             "reviewed": len(reviewed), "unreviewed": len(records) - len(reviewed)}}
