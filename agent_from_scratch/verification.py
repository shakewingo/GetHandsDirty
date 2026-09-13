"""Optional evidence checks for explicit task contracts; no extra model call."""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from .utils import file_version, resolve_path


@dataclass
class CheckResult:
    name: str
    status: Literal["passed", "pending", "blocked"]
    feedback: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


CompletionCheck = Callable[[list[dict]], CheckResult]


def missing_ranges(size: int, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cursor = 0
    missing = []
    for start, end in sorted(ranges):
        if not 0 <= start <= end <= size:
            raise ValueError("Read evidence falls outside the target file.")
        if start > cursor:
            missing.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < size:
        missing.append((cursor, size))
    return missing


def full_file_check(workspace: str | Path, path: str | Path) -> CompletionCheck:
    """Create a per-turn snapshot; verify reads, not the quality of the final answer.

    The caller selects the full-read contract explicitly. Version checks assume
    the same single-writer workspace as the filesystem tools, without file locks.
    """
    root = Path(workspace).resolve()
    target = resolve_path(root, path)
    if not target.is_file():
        raise ValueError(f"Expected an existing regular file: {target}")
    snapshot = target.stat()
    size, version = snapshot.st_size, file_version(snapshot)

    def check(messages: list[dict]) -> CheckResult:
        if not target.is_file() or file_version(target.stat()) != version:
            return CheckResult("read_coverage", "blocked", "The target file changed; reads cannot be combined.")
        calls, ranges, saw_eof = {}, [], False
        for message in messages:
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []):
                    calls[call["id"]] = call["function"]
            if message["role"] != "tool":
                continue
            observed = json.loads(message["content"])
            function = calls.pop(message["tool_call_id"], None)
            if observed.get("ok") is not True or observed.get("tool_name") != "read_file":
                continue
            output = observed["output"]
            if resolve_path(root, output["path"]) != target:
                continue
            if (not function or function["name"] != "read_file"
                    or observed["call_id"] != message["tool_call_id"]):
                raise ValueError("Read observation has no matching call.")
            arguments = json.loads(function["arguments"])
            start = output["offset"]
            end = start + len(output["content"].encode("utf-8"))
            if (resolve_path(root, arguments["path"]) != target or start != arguments.get("offset", 0)
                    or type(start) is not int or not 0 <= start <= end <= size
                    or output["size_bytes"] != size or output["version"] != version
                    or output["eof"] is not (end == size)
                    or output["next_offset"] != (None if end == size else end)
                    or (end == start and end != size)):
                raise ValueError("Read observation has inconsistent range or version evidence.")
            ranges.append((start, end))
            saw_eof |= output["eof"]
        missing = missing_ranges(size, ranges)
        evidence = {"path": str(target.relative_to(root)), "version": version,
                    "size_bytes": size, "covered_bytes": size - sum(b - a for a, b in missing),
                    "missing_ranges": missing, "saw_eof": saw_eof}
        if ranges and not missing and saw_eof:
            return CheckResult("read_coverage", "passed", evidence=evidence)
        offset = missing[0][0] if missing else 0
        return CheckResult("read_coverage", "pending",
                           f"Full-file read is incomplete: {evidence['covered_bytes']}/{size} bytes. "
                           f"Continue read_file with path={str(target.relative_to(root))!r}, offset={offset}. "
                           "Use next_offset for subsequent chunks, then finish the original request.", evidence)

    return check
