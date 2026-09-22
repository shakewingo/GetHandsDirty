"""Long-poll Telegram bridge to Agent.run_turn.

Design: docs/superpowers/specs/2026-09-22-telegram-bot-design.md
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_API_BASE = "https://api.telegram.org"
LONG_POLL_TIMEOUT = 30


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
