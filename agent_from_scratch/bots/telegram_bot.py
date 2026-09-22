"""Long-poll Telegram bridge to Agent.run_turn.

Design: docs/superpowers/specs/2026-09-22-telegram-bot-design.md
"""

from __future__ import annotations

from typing import Any

import httpx

from ..trace import RunStopReason, TurnResult

DEFAULT_API_BASE = "https://api.telegram.org"
LONG_POLL_TIMEOUT = 30
MAX_MESSAGE_LENGTH = 4096
_TRUNCATION_MARKER = "\n… [truncated]"


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
