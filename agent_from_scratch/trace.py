from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger
from .utils import write_jsonl

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage
    from .llm import ResponseError


class RunStopReason(StrEnum):
    FINAL_RESPONSE = "final_response"
    MAX_ITERATIONS = "max_iterations"
    MODEL_ERROR = "model_error"
    INTERRUPTED = "interrupted"


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


@dataclass
class TurnResult:
    messages: list[ChatCompletionRequestMessage]
    final_answer: str | None = None
    stop_reason: RunStopReason = RunStopReason.MAX_ITERATIONS
    error_message: str | None = None
    run_id: str = ""
    elapsed_seconds: float = 0.0
    model_requests: list[ModelRequest] = field(default_factory=list)
    schema_version: int = 1
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

    def save_parse_error(self, result: TurnResult, iteration: int, error: ResponseError) -> None:
        self._write(f"{result.run_id}.parse-error-{iteration}.jsonl", {
            "schema_version": 1, "event": ModelRequestStatus.PARSE_ERROR,
            "run_id": result.run_id, "session_id": result.session_id,
            "iteration": iteration, "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": error.code, "error": str(error), "raw_response": error.raw_response,
        }, "parse-error event")

    def load_run(self, run_id: str) -> dict | None:
        """Resolve a session's run reference. Missing evidence is explicitly unavailable."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run ID.")
        if self.directory is None:
            return None
        root = self.directory.resolve()
        path = (root / f"{run_id}.jsonl").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Run path must stay inside the trace directory.")
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            raise ValueError("Invalid run trace.")
        return record
