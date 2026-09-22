"""Long-poll Telegram bridge to Agent.run_turn.

Design: docs/superpowers/specs/2026-09-22-telegram-bot-design.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from ..agent import Agent
from ..context import InstructionLoadError
from ..session import SessionStore
from ..trace import RunStopReason, TurnResult

DEFAULT_API_BASE = "https://api.telegram.org"
LONG_POLL_TIMEOUT = 30
MAX_MESSAGE_LENGTH = 4096
_TRUNCATION_MARKER = "\n… [truncated]"
SESSION_RESET_COMMANDS = {"/new", "/reset"}


class TelegramAPIError(RuntimeError):
    """A Telegram Bot API call returned ok=false or an unexpected shape."""


class TelegramClient:
    """Thin wrapper over the two Bot API calls this bot needs."""

    def __init__(self, token: str, *, base_url: str = DEFAULT_API_BASE, timeout: float = 35.0):
        self._base = f"{base_url}/bot{token}"
        self._client = httpx.Client(timeout=timeout)

    def get_updates(self, offset: int | None, *,
                    poll_timeout: int = LONG_POLL_TIMEOUT) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": poll_timeout}
        if offset is not None:
            params["offset"] = offset
        response = self._client.get(f"{self._base}/getUpdates", params=params)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise TelegramAPIError(f"getUpdates failed: {payload}")
        result = payload.get("result")
        if not isinstance(result, list):
            raise TelegramAPIError(f"getUpdates returned no result list: {payload}")
        return result

    def send_message(self, chat_id: int | str, text: str) -> None:
        response = self._client.post(f"{self._base}/sendMessage",
                                      json={"chat_id": chat_id, "text": text})
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise TelegramAPIError(f"sendMessage failed: {payload}")

    def close(self) -> None:
        self._client.close()


def is_allowed(update: dict[str, Any], allowed_user_id: int) -> bool:
    message = update.get("message")
    if not isinstance(message, dict):
        return False
    sender = message.get("from")
    return isinstance(sender, dict) and sender.get("id") == allowed_user_id


def extract_message(update: dict[str, Any]) -> tuple[int | str, str] | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    text = message.get("text")
    if not isinstance(chat, dict) or "id" not in chat:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return chat["id"], text


def truncate_for_telegram(text: str) -> str:
    if len(text) <= MAX_MESSAGE_LENGTH:
        return text
    return text[: MAX_MESSAGE_LENGTH - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def format_reply(result: TurnResult) -> str:
    """Mirror Agent.run_repl's per-RunStopReason branches (agent.py:433-448), as text."""
    if result.stop_reason == RunStopReason.FINAL_RESPONSE:
        text = result.final_answer or ""
    elif result.stop_reason == RunStopReason.MODEL_ERROR:
        text = f"Stopped: model error occurred: {result.error_message}"
    elif result.stop_reason == RunStopReason.MAX_ITERATIONS:
        text = "Stopped: reached maximum iterations."
    elif result.stop_reason == RunStopReason.INTERRUPTED:
        text = "Turn interrupted."
    elif result.stop_reason == RunStopReason.CONTEXT_LIMIT:
        text = (f"Stopped: {result.error_message}\n"
               "Completed tool actions remain in effect. "
               "This turn is not included in replayable session history.")
    else:
        text = f"Stopped: {result.stop_reason}. {result.error_message or ''}"
    return truncate_for_telegram(text)


def load_offset(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def save_offset(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(offset))


def handle_update(update: dict[str, Any], *, agent: Agent, store: SessionStore,
                  client: Any, allowed_user_id: int,
                  active_sessions: dict[Any, str], compact_pending: dict[Any, bool]) -> None:
    """Process one Telegram update: allowlist, then command dispatch, then one agent turn."""
    if not is_allowed(update, allowed_user_id):
        sender = (update.get("message") or {}).get("from") or {}
        logger.warning("Ignoring message from disallowed sender {}", sender.get("id"))
        return
    parsed = extract_message(update)
    if parsed is None:
        return
    chat_id, text = parsed
    session_id = active_sessions.setdefault(chat_id, str(chat_id))
    command = text.strip()

    if command == "/compact":
        # Compaction runs inside a turn, against that turn's own view, so the request that
        # follows is the earliest point this can take effect (mirrors agent.py:406-411).
        compact_pending[chat_id] = True
        client.send_message(chat_id, "The next message will summarize earlier history before acting.")
        return

    if command in SESSION_RESET_COMMANDS or command.startswith("/session "):
        try:
            new_session_id = agent._session_command(command, session_id, store)
        except (OSError, ValueError) as error:
            client.send_message(chat_id, f"Could not update session: {error}")
            return
        active_sessions[chat_id] = new_session_id
        client.send_message(chat_id, f"Session: {new_session_id}")
        return

    try:
        history = store.load_history(session_id)
        checkpoint = store.load_checkpoint(session_id, history)
    except (OSError, ValueError) as error:
        client.send_message(chat_id,
            f"Session unavailable: {error}. Use /new, /reset, or /session <id> to recover.")
        return

    compact = compact_pending.pop(chat_id, False)
    try:
        result = agent.run_turn(text, history, session_id=session_id, compact=compact,
                                checkpoint=checkpoint)
    except InstructionLoadError as error:
        if compact:
            compact_pending[chat_id] = True  # the turn never started; a pending compact still applies
        client.send_message(chat_id, f"Instructions unavailable: {error}")
        return
    except Exception as error:  # noqa: BLE001 - one bad turn must not end 24/7 availability
        logger.exception("Turn failed for session {}", session_id)
        client.send_message(chat_id, f"Internal error: {error}")
        return
    client.send_message(chat_id, format_reply(result))
