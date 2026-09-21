from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger
from .utils import resolve_path, write_jsonl

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


class RunStopReason(StrEnum):
    FINAL_RESPONSE = "final_response"
    MAX_ITERATIONS = "max_iterations"
    MODEL_ERROR = "model_error"
    INTERRUPTED = "interrupted"
    NO_PROGRESS = "no_progress"
    TOOL_LIMIT = "tool_limit"
    CONTEXT_LIMIT = "context_limit"


class ModelRequestStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    PARSE_ERROR = "parse_error"
    MODEL_ERROR = "model_error"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"


@dataclass
class ModelRequest:
    iteration: int
    input_message_count: int
    status: ModelRequestStatus = ModelRequestStatus.STARTED
    call_ids: list[str] = field(default_factory=list)
    usage: dict[str, int | None] | None = None
    finish_reason: str | None = None
    raw_response: Any = None  # Includes malformed JSON envelopes, not only valid objects.
    error_code: str | None = None
    error_message: str | None = None

    # Pre-generation measurement for later action like compact, distinct from the backend's
    # post-generation usage. Recorded as "context" in schema_version 4 and earlier records.
    budget: dict[str, Any] | None = None
    purpose: str = "agent"
    input_messages: list | None = None  # None in legacy records: use the raw prefix.
    tools: dict | None = None
    covered_boundary: int = 1
    last_sent_boundary: int = 0
    compact_before: dict | None = None
    compact_after: dict | None = None
    instructions: dict | None = None  # Rule provenance when a compact boundary reloaded them.
    elided_messages: int = 0  # Tool outputs shown as stubs in this request's view.


def used_model_calls(requests: list[ModelRequest]) -> int:
    """Count requests charged against the turn's budget; blocked ones never reached the model."""
    return sum(request.status != ModelRequestStatus.BLOCKED for request in requests)


def request_budget(record: dict) -> dict | None:
    """Read one request's pre-generation measurement across the schema-5 rename.

    Records at schema_version 5 and later use `budget`; 4 and earlier use `context`.
    """
    budget = record.get("budget")
    return record.get("context") if budget is None else budget


@dataclass
class TurnResult:
    messages: list[ChatCompletionRequestMessage]
    final_answer: str | None = None
    stop_reason: RunStopReason = RunStopReason.MAX_ITERATIONS
    error_message: str | None = None
    run_id: str = ""
    elapsed_seconds: float = 0.0
    model_requests: list[ModelRequest] = field(default_factory=list)
    schema_version: int = 5  # 5 renamed ModelRequest.context to budget; later fields are additive.
    session_id: str | None = None
    input: str = ""
    started_at: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    stuck_reminders: int = 0  # Runtime reminders injected for repeated identical calls.


class TraceStore:
    """Run evidence, separate from replayable session history. None disables writes."""

    def __init__(self, directory: Path | None):
        self.directory = directory

    def _write(self, filename: str, record: dict, description: str) -> None:
        if self.directory is None:
            return
        try:
            write_jsonl(self.directory / filename, [record])
        except OSError as error:
            logger.error("Could not save {}: {}", description, error)

    def save_run(self, result: TurnResult) -> None:
        # Raw evidence is unchanged; requests record their actual model-facing inputs.
        self._write(f"{result.run_id}.jsonl", asdict(result), "run trace")

    def load_run(self, run_id: str) -> dict | None:
        """Resolve a session's run reference. Missing evidence is explicitly unavailable."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run ID.")
        if self.directory is None:
            return None
        path = resolve_path(self.directory, f"{run_id}.jsonl")
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            raise ValueError("Invalid run trace.")
        return record
