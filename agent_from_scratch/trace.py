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


class ModelRequestStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    PARSE_ERROR = "parse_error"
    MODEL_ERROR = "model_error"
    INTERRUPTED = "interrupted"


@dataclass
class ModelRequest:
    iteration: int
    input_message_count: int
    status: ModelRequestStatus = ModelRequestStatus.STARTED
    call_id: str | None = None
    usage: dict[str, int | None] | None = None
    finish_reason: str | None = None
    raw_response: Any = None  # Includes malformed JSON envelopes, not only valid objects.
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class TurnResult:
    messages: list[ChatCompletionRequestMessage]
    final_answer: str | None = None
    stop_reason: RunStopReason = RunStopReason.MAX_ITERATIONS
    error_message: str | None = None
    run_id: str = ""
    elapsed_seconds: float = 0.0
    model_requests: list[ModelRequest] = field(default_factory=list)
    schema_version: int = 2
    session_id: str | None = None
    input: str = ""
    started_at: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


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
        # Full messages remain available for ModelRequest.input_message_count prefixes.
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
