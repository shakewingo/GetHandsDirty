from __future__ import annotations
from .utils import resolve_path, write_jsonl
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import TYPE_CHECKING
from .trace import RunStopReason

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage

def checkpoint_digest(messages: list[ChatCompletionRequestMessage]) -> str:
    """Bind a checkpoint to the exact raw prefix it claims to summarize."""
    return sha256(json.dumps(messages, sort_keys=True,
                             ensure_ascii=False).encode("utf-8")).hexdigest()


class SessionStore:
    """Single-writer run index; only completed runs contribute replayable messages."""

    def __init__(self, dir: str | Path):
        self.dir = Path(dir)

    def _path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("Session ID must contain only letters, numbers, '_' or '-'.")
        return resolve_path(self.dir, f"{session_id}.jsonl")

    @staticmethod
    def _validate_checkpoint(record: dict) -> None:
        """A checkpoint is only usable if its boundary, digest and configuration are intact."""
        if record.get("schema_version") != 2:
            raise ValueError("Unsupported checkpoint schema_version.")
        if type(record.get("covered")) is not int or record["covered"] < 1:
            raise ValueError("Checkpoint covered must be a positive integer.")
        for name in ("summary", "source_sha256", "created_at"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError(f"Checkpoint {name} must be nonempty text.")
        if not isinstance(record.get("config"), dict):
            raise ValueError("Checkpoint config must be an object.")

    @staticmethod
    def _validate_record(record: object) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
            raise ValueError("Expected a record with a string run_id.")
        if record.get("kind") == "checkpoint":
            SessionStore._validate_checkpoint(record)
            return
        if not isinstance(record.get("messages"), list):
            raise ValueError("Expected a record with a messages list.")
        if "schema_version" in record:
            if type(record["schema_version"]) is not int or record["schema_version"] != 1:
                raise ValueError("Unsupported session schema_version.")
            # Legacy runtime coverage checks emitted this retired stop reason.
            reason = record.get("stop_reason")
            if reason != "check_failed":
                reason = RunStopReason(reason)
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
                if message["role"] != "assistant" or not isinstance(calls, list) or not calls:
                    raise ValueError("Expected a nonempty list of assistant tool calls.")
                ids = set()
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    if (not isinstance(call, dict) or not isinstance(call.get("id"), str)
                            or call.get("type") != "function" or not isinstance(function, dict)
                            or not isinstance(function.get("name"), str)
                            or not isinstance(function.get("arguments"), str)):
                        raise ValueError("Invalid assistant tool-call structure.")
                    if call["id"] in ids:
                        raise ValueError("Duplicate assistant tool-call ID.")
                    ids.add(call["id"])

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
                if record.get("kind") != "checkpoint"
                and ("schema_version" not in record
                     or record["stop_reason"] == RunStopReason.FINAL_RESPONSE)
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

    def append_checkpoint(self, session_id: str, run_id: str, *, covered: int, summary: str,
                          history: list[ChatCompletionRequestMessage], config: dict) -> None:
        """Publish a summary checkpoint over an already-saved prefix of this session.

        Args:
            session_id: the session whose index gains the record.
            run_id: the run whose compaction produced the summary.
            covered: how many session messages the summary represents, counted from the
                start of replayable history.
            summary: the handoff text to replay in place of that prefix.
            history: the full replayable history this checkpoint was computed against.
            config: what produced the summary, for a later reader to judge staleness.

        Raises:
            ValueError: if the boundary is not already backed by saved messages, which is
                what keeps raw evidence on disk before anything references it.
        """
        saved = self.load_history(session_id)
        if covered > len(saved) or history[:covered] != saved[:covered]:
            raise ValueError("Checkpoint boundary is not backed by saved session messages.")
        record = {"schema_version": 2, "kind": "checkpoint", "run_id": run_id,
                  "created_at": datetime.now(timezone.utc).isoformat(), "covered": covered,
                  "summary": summary, "source_sha256": checkpoint_digest(history[:covered]),
                  "config": config}
        self._validate_record(record)
        records = self.load_records(session_id)
        records.append(record)
        write_jsonl(self._path(session_id), records)

    def load_checkpoint(self, session_id: str,
                        history: list[ChatCompletionRequestMessage]) -> dict | None:
        """Return the newest checkpoint still backed by this history, else None.

        A stale or out-of-range boundary is not an error: the caller replays raw history and
        lets the ordinary budget check decide. Only the newest checkpoint is considered, so
        an edited session cannot silently fall back to an older summary of the same prefix.
        """
        for record in reversed(self.load_records(session_id)):
            if record.get("kind") != "checkpoint":
                continue
            if (record["covered"] <= len(history)
                    and checkpoint_digest(history[:record["covered"]]) == record["source_sha256"]):
                return record
            return None
        return None

    def reset(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
