from __future__ import annotations
from .utils import resolve_path, write_jsonl
from pathlib import Path
import json
import re
from typing import TYPE_CHECKING
from .trace import RunStopReason

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage

class SessionStore:
    """Single-writer run index; only completed runs contribute replayable messages."""

    def __init__(self, dir: str | Path):
        self.dir = Path(dir)

    def _path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("Session ID must contain only letters, numbers, '_' or '-'.")
        return resolve_path(self.dir, f"{session_id}.jsonl")

    @staticmethod
    def _validate_record(record: object) -> None:
        if (not isinstance(record, dict)
                or not isinstance(record.get("run_id"), str)
                or not isinstance(record.get("messages"), list)):
            raise ValueError("Expected a record with a string run_id and a messages list.")
        if "schema_version" in record:
            if type(record["schema_version"]) is not int or record["schema_version"] != 1:
                raise ValueError("Unsupported session schema_version.")
            reason = RunStopReason(record.get("stop_reason"))
            if record.get("started_at") is not None and not isinstance(record["started_at"], str):
                raise ValueError("started_at must be text or null.")
            if reason != RunStopReason.FINAL_RESPONSE and record["messages"]:
                raise ValueError("Unfinished runs must not contain replayable messages.")
        for message in record["messages"]:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant", "tool"):
                raise ValueError("Expected a user, assistant, or tool message object.")
            calls = message.get("tool_calls")
            if not isinstance(message.get("content"), str):
                if not (message["role"] == "assistant" and message.get("content") is None and calls):
                    raise ValueError("Message content must be text (or null for an assistant tool call).")
            if message["role"] == "tool" and not isinstance(message.get("tool_call_id"), str):
                raise ValueError("Tool messages require a string tool_call_id.")
            if calls is not None:
                if message["role"] != "assistant" or not isinstance(calls, list) or len(calls) != 1:
                    raise ValueError("Expected one assistant tool call.")
                call = calls[0]
                function = call.get("function") if isinstance(call, dict) else None
                if (not isinstance(call, dict) or not isinstance(call.get("id"), str)
                        or call.get("type") != "function" or not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError("Invalid assistant tool-call structure.")

    def load_records(self, session_id: str) -> list[dict]:
        """Read both legacy {run_id, messages} lines and versioned index entries."""
        session_file = self._path(session_id)
        if not session_file.exists():
            return []
        records = []
        for line_number, line in enumerate(session_file.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
                self._validate_record(record)
            except ValueError as error:
                raise ValueError(f"Invalid session {session_id!r}, line {line_number}: {error}") from error
            records.append(record)
        return records

    def load_history(self, session_id: str) -> list[ChatCompletionRequestMessage]:
        # Legacy records were written only for completed turns. Missing metadata stays absent.
        return [message for record in self.load_records(session_id)
                if "schema_version" not in record or record["stop_reason"] == RunStopReason.FINAL_RESPONSE
                for message in record["messages"]]

    def append(self, session_id: str, run_id: str, messages: list[ChatCompletionRequestMessage],
               *, started_at: str | None = None,
               stop_reason: RunStopReason = RunStopReason.FINAL_RESPONSE) -> None:
        records = self.load_records(session_id)
        record = {"schema_version": 1, "run_id": run_id, "started_at": started_at,
                  "stop_reason": stop_reason, "messages": messages}
        self._validate_record(record)
        records.append(record)
        write_jsonl(self._path(session_id), records)

    def reset(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
