# Telegram Bot Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the existing agent be reached from a phone over Telegram, as a long-lived process
on an always-on Mac mini, without changing any agent core code.

**Architecture:** One new package, `agent_from_scratch/bots/telegram_bot.py`, wraps the two
Telegram Bot API calls it needs (long-poll `getUpdates`, `sendMessage`), maps each Telegram
`chat_id` to an `Agent.run_turn` session, and reuses `Agent._session_command` and
`SessionStore` exactly as `run_repl` already does. A `launchd` job keeps the process resident
on the Mac mini.

**Tech Stack:** Python (stdlib + `httpx`, already a project dependency), `unittest`
(project convention — not `pytest`), `launchd` (macOS process supervision).

**Spec:** [docs/superpowers/specs/2026-09-22-telegram-bot-design.md](../specs/2026-09-22-telegram-bot-design.md)

## Global Constraints

- No changes to `agent.py`, `llm.py`, `session.py`, `context.py`, `compact.py`, or any
  `tools/*` module — this is purely additive (spec: Architecture).
- No new dependencies — `httpx` is already in `requirements-tools.txt`; `loguru` is already a
  core dependency (spec: Constraints that shaped this).
- Message processing is strictly sequential; no concurrency, no async (spec: Architecture).
- The allowlist check (single Telegram numeric user ID) runs before anything else on every
  update and is not configurable-off (spec: Security).
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USER_ID` come from the environment only, never
  hardcoded or committed (spec: Security).
- No `on_progress` callback is wired up — one reply per turn, sent only once the turn ends
  (spec: Architecture).
- Tests use `unittest.TestCase` and run via
  `python -m unittest discover -s agent_from_scratch/tests -v` from the repository root
  (existing project convention — see `tests/test_turn.py`, `tests/test_session.py`).
- Run every command from the worktree root
  (`/Users/yingyao/Desktop/Code.nosync/GetHandsDirty.nosync/.claude/worktrees/feat+telegram-bot`),
  never from the original repository checkout.

---

## Baseline

Confirmed before writing this plan: `python -m unittest discover -s agent_from_scratch/tests -q`
passes 250/250 tests in ~4 seconds on a clean `feat/telegram-bot` checkout. Re-run this after
every task to confirm no regression.

---

### Task 1: `TelegramClient` — the Bot API wrapper

**Files:**
- Create: `agent_from_scratch/bots/__init__.py`
- Create: `agent_from_scratch/bots/telegram_bot.py`
- Test: `agent_from_scratch/tests/test_telegram_bot.py`

**Interfaces:**
- Produces: `TelegramAPIError(RuntimeError)`; `TelegramClient(token: str, *, base_url: str = DEFAULT_API_BASE, timeout: float = 35.0)` with `.get_updates(offset: int | None, *, poll_timeout: int = LONG_POLL_TIMEOUT) -> list[dict[str, Any]]`, `.send_message(chat_id: int | str, text: str) -> None`, `.close() -> None`. Constants `DEFAULT_API_BASE`, `LONG_POLL_TIMEOUT`.

- [ ] **Step 1: Create the package and write the failing test**

Create `agent_from_scratch/bots/__init__.py` (empty file).

Create `agent_from_scratch/tests/test_telegram_bot.py`:

```python
from __future__ import annotations

import unittest
from unittest.mock import Mock

from agent_from_scratch.bots.telegram_bot import TelegramAPIError, TelegramClient


class TelegramClientTests(unittest.TestCase):
    def setUp(self):
        self.client = TelegramClient("test-token")
        self.client._client = Mock()

    def test_get_updates_returns_result_list(self):
        response = Mock()
        response.json.return_value = {"ok": True, "result": [{"update_id": 1}]}
        self.client._client.get.return_value = response
        result = self.client.get_updates(5)
        self.assertEqual(result, [{"update_id": 1}])
        args, kwargs = self.client._client.get.call_args
        self.assertEqual(args[0], "https://api.telegram.org/bottest-token/getUpdates")
        self.assertEqual(kwargs["params"], {"timeout": 30, "offset": 5})

    def test_get_updates_omits_offset_when_none(self):
        response = Mock()
        response.json.return_value = {"ok": True, "result": []}
        self.client._client.get.return_value = response
        self.client.get_updates(None)
        args, kwargs = self.client._client.get.call_args
        self.assertEqual(kwargs["params"], {"timeout": 30})

    def test_get_updates_raises_on_not_ok(self):
        response = Mock()
        response.json.return_value = {"ok": False, "description": "bad token"}
        self.client._client.get.return_value = response
        with self.assertRaises(TelegramAPIError):
            self.client.get_updates(None)

    def test_send_message_posts_chat_id_and_text(self):
        response = Mock()
        response.json.return_value = {"ok": True, "result": {}}
        self.client._client.post.return_value = response
        self.client.send_message(42, "hello")
        args, kwargs = self.client._client.post.call_args
        self.assertEqual(args[0], "https://api.telegram.org/bottest-token/sendMessage")
        self.assertEqual(kwargs["json"], {"chat_id": 42, "text": "hello"})

    def test_send_message_raises_on_not_ok(self):
        response = Mock()
        response.json.return_value = {"ok": False, "description": "chat not found"}
        self.client._client.post.return_value = response
        with self.assertRaises(TelegramAPIError):
            self.client.send_message(42, "hello")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `ModuleNotFoundError: No module named 'agent_from_scratch.bots.telegram_bot'`

- [ ] **Step 3: Implement `TelegramClient`**

Create `agent_from_scratch/bots/telegram_bot.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `Ran 5 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add agent_from_scratch/bots/__init__.py agent_from_scratch/bots/telegram_bot.py agent_from_scratch/tests/test_telegram_bot.py
git commit -m "feat: add the Telegram Bot API client wrapper" --author="claude <claude@noreply>"
```

---

### Task 2: Allowlist, message extraction, and reply formatting

**Files:**
- Modify: `agent_from_scratch/bots/telegram_bot.py`
- Test: `agent_from_scratch/tests/test_telegram_bot.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (independent helpers), but lives in the same module.
- Produces: `MAX_MESSAGE_LENGTH: int`; `is_allowed(update: dict, allowed_user_id: int) -> bool`;
  `extract_message(update: dict) -> tuple[int | str, str] | None`;
  `truncate_for_telegram(text: str) -> str`; `format_reply(result: TurnResult) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `agent_from_scratch/tests/test_telegram_bot.py` (add these imports to the top of the
file, alongside the existing ones):

```python
from agent_from_scratch.bots.telegram_bot import (
    MAX_MESSAGE_LENGTH, TelegramAPIError, TelegramClient,
    extract_message, format_reply, is_allowed, truncate_for_telegram,
)
from agent_from_scratch.trace import RunStopReason, TurnResult
```

Append these classes to the end of the file (before the `if __name__ == "__main__":` block):

```python
def _result(stop_reason, **kwargs):
    return TurnResult(messages=[], stop_reason=stop_reason, **kwargs)


class AllowlistTests(unittest.TestCase):
    def test_allows_configured_sender(self):
        update = {"message": {"from": {"id": 99}, "chat": {"id": 99}, "text": "hi"}}
        self.assertTrue(is_allowed(update, 99))

    def test_rejects_other_sender(self):
        update = {"message": {"from": {"id": 1}, "chat": {"id": 1}, "text": "hi"}}
        self.assertFalse(is_allowed(update, 99))

    def test_rejects_non_message_update(self):
        self.assertFalse(is_allowed({"edited_message": {}}, 99))


class ExtractMessageTests(unittest.TestCase):
    def test_extracts_chat_id_and_text(self):
        update = {"message": {"chat": {"id": 7}, "text": "hello"}}
        self.assertEqual(extract_message(update), (7, "hello"))

    def test_returns_none_for_non_text_message(self):
        update = {"message": {"chat": {"id": 7}, "sticker": {}}}
        self.assertIsNone(extract_message(update))

    def test_returns_none_for_blank_text(self):
        update = {"message": {"chat": {"id": 7}, "text": "   "}}
        self.assertIsNone(extract_message(update))

    def test_returns_none_for_missing_message(self):
        self.assertIsNone(extract_message({"edited_message": {}}))


class FormatReplyTests(unittest.TestCase):
    def test_final_response_returns_answer(self):
        result = _result(RunStopReason.FINAL_RESPONSE, final_answer="42")
        self.assertEqual(format_reply(result), "42")

    def test_model_error_includes_detail(self):
        result = _result(RunStopReason.MODEL_ERROR, error_message="backend down")
        self.assertEqual(format_reply(result), "Stopped: model error occurred: backend down")

    def test_max_iterations(self):
        result = _result(RunStopReason.MAX_ITERATIONS)
        self.assertEqual(format_reply(result), "Stopped: reached maximum iterations.")

    def test_interrupted(self):
        result = _result(RunStopReason.INTERRUPTED)
        self.assertEqual(format_reply(result), "Turn interrupted.")

    def test_context_limit_notes_effects_kept(self):
        result = _result(RunStopReason.CONTEXT_LIMIT, error_message="too large")
        reply = format_reply(result)
        self.assertIn("Stopped: too large", reply)
        self.assertIn("Completed tool actions remain in effect.", reply)

    def test_unrecognized_stop_reason_falls_back(self):
        result = _result(RunStopReason.TOOL_LIMIT, error_message="cap hit")
        self.assertEqual(format_reply(result), "Stopped: tool_limit. cap hit")

    def test_long_reply_is_truncated(self):
        result = _result(RunStopReason.FINAL_RESPONSE, final_answer="x" * 5000)
        reply = format_reply(result)
        self.assertLessEqual(len(reply), MAX_MESSAGE_LENGTH)
        self.assertTrue(reply.endswith("[truncated]"))


class TruncateForTelegramTests(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(truncate_for_telegram("hi"), "hi")

    def test_long_text_is_cut_to_the_limit(self):
        truncated = truncate_for_telegram("x" * 5000)
        self.assertEqual(len(truncated), MAX_MESSAGE_LENGTH)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `ImportError: cannot import name 'is_allowed' ...`

- [ ] **Step 3: Implement the helpers**

In `agent_from_scratch/bots/telegram_bot.py`, replace the top import block with:

```python
from __future__ import annotations

from typing import Any

import httpx

from ..trace import RunStopReason, TurnResult

DEFAULT_API_BASE = "https://api.telegram.org"
LONG_POLL_TIMEOUT = 30
MAX_MESSAGE_LENGTH = 4096
_TRUNCATION_MARKER = "\n… [truncated]"
```

Append to the end of the file (after the `TelegramClient` class):

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `Ran 21 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add agent_from_scratch/bots/telegram_bot.py agent_from_scratch/tests/test_telegram_bot.py
git commit -m "feat: add Telegram allowlist, message extraction and reply formatting" --author="claude <claude@noreply>"
```

---

### Task 3: Offset persistence and per-update session/command dispatch

**Files:**
- Modify: `agent_from_scratch/bots/telegram_bot.py`
- Test: `agent_from_scratch/tests/test_telegram_bot.py`

**Interfaces:**
- Consumes: `Agent` (`agent_from_scratch/agent.py:60`, specifically `.run_turn(...)` and
  `._session_command(command, session_id, store)`), `SessionStore`
  (`agent_from_scratch/session.py:20`, `.load_history`, `.load_checkpoint`), `InstructionLoadError`
  (`agent_from_scratch/context.py:19`), `format_reply`/`is_allowed`/`extract_message` from Task 2.
- Produces: `SESSION_RESET_COMMANDS: set[str]`; `load_offset(path: Path) -> int | None`;
  `save_offset(path: Path, offset: int) -> None`;
  `handle_update(update: dict, *, agent: Agent, store: SessionStore, client: Any, allowed_user_id: int, active_sessions: dict[Any, str], compact_pending: dict[Any, bool]) -> None`.

Note: `active_sessions` is a per-process dict that tracks each Telegram chat's *current*
`session_id`, because `/new` and `/session <id>` change which session a chat is bound to;
`run_repl` gets this for free from a loop-local variable, but this bot handles one update at a
time, so the mapping must be kept explicitly across calls (spec: Architecture — "each Telegram
chat its own persisted conversation"). `compact_pending` mirrors `run_repl`'s local
`compact_next` flag the same way, per chat.

- [ ] **Step 1: Write the failing tests**

Add to the top imports of `agent_from_scratch/tests/test_telegram_bot.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_from_scratch.agent import Agent
from agent_from_scratch.bots.telegram_bot import (
    SESSION_RESET_COMMANDS, handle_update, load_offset, save_offset,
)
from agent_from_scratch.context import InstructionLoadError
from agent_from_scratch.llm import LLM
from agent_from_scratch.session import SessionStore
```

Append to the end of the file:

```python
def _text_update(chat_id, text, update_id=1, sender_id=99):
    return {"update_id": update_id,
            "message": {"chat": {"id": chat_id}, "from": {"id": sender_id}, "text": text}}


class OffsetPersistenceTests(unittest.TestCase):
    def test_load_offset_returns_none_when_missing(self):
        with TemporaryDirectory() as directory:
            self.assertIsNone(load_offset(Path(directory, "offset.txt")))

    def test_save_and_load_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "offset.txt")
            save_offset(path, 42)
            self.assertEqual(load_offset(path), 42)

    def test_load_offset_returns_none_on_corrupt_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "offset.txt")
            path.write_text("not-a-number")
            self.assertIsNone(load_offset(path))


class HandleUpdateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        state_dir = self.directory.name
        self.agent = Agent(Mock(spec=LLM), state_dir=state_dir)
        self.store = SessionStore(Path(state_dir, "sessions"))
        self.client = Mock()
        self.active_sessions: dict[Any, str] = {}
        self.compact_pending: dict[Any, bool] = {}
        self.allowed_user_id = 99

    def _handle(self, update):
        handle_update(update, agent=self.agent, store=self.store, client=self.client,
                      allowed_user_id=self.allowed_user_id, active_sessions=self.active_sessions,
                      compact_pending=self.compact_pending)

    def test_ignores_sender_outside_allowlist(self):
        self._handle(_text_update(1, "hello", sender_id=1))
        self.client.send_message.assert_not_called()

    def test_runs_a_turn_and_sends_final_answer(self):
        self.agent.run_turn = Mock(return_value=TurnResult(
            messages=[], stop_reason=RunStopReason.FINAL_RESPONSE, final_answer="42"))
        self._handle(_text_update(1, "what is 6*7?"))
        self.agent.run_turn.assert_called_once()
        args, kwargs = self.agent.run_turn.call_args
        self.assertEqual(args[0], "what is 6*7?")
        self.assertEqual(kwargs["session_id"], "1")
        self.assertFalse(kwargs["compact"])
        self.client.send_message.assert_called_once_with(1, "42")

    def test_new_command_switches_to_a_fresh_session_and_persists_it(self):
        self.agent.run_turn = Mock(return_value=TurnResult(
            messages=[], stop_reason=RunStopReason.FINAL_RESPONSE, final_answer="ok"))
        self._handle(_text_update(1, "/new", update_id=1))
        self.agent.run_turn.assert_not_called()
        new_session_id = self.active_sessions[1]
        self.assertNotEqual(new_session_id, "1")
        self._handle(_text_update(1, "hello again", update_id=2))
        kwargs = self.agent.run_turn.call_args.kwargs
        self.assertEqual(kwargs["session_id"], new_session_id)

    def test_compact_command_is_applied_to_the_next_turn_only(self):
        self.agent.run_turn = Mock(return_value=TurnResult(
            messages=[], stop_reason=RunStopReason.FINAL_RESPONSE, final_answer="ok"))
        self._handle(_text_update(1, "/compact", update_id=1))
        self.client.send_message.assert_called_once()
        self._handle(_text_update(1, "continue", update_id=2))
        self.assertTrue(self.agent.run_turn.call_args.kwargs["compact"])
        self._handle(_text_update(1, "again", update_id=3))
        self.assertFalse(self.agent.run_turn.call_args.kwargs["compact"])

    def test_instruction_load_error_replies_without_crashing(self):
        self.agent.run_turn = Mock(side_effect=InstructionLoadError("bad rules file"))
        self._handle(_text_update(1, "hello"))
        self.client.send_message.assert_called_once_with(
            1, "Instructions unavailable: bad rules file")

    def test_unexpected_error_replies_and_does_not_propagate(self):
        self.agent.run_turn = Mock(side_effect=RuntimeError("boom"))
        self._handle(_text_update(1, "hello"))
        self.client.send_message.assert_called_once_with(1, "Internal error: boom")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `ImportError: cannot import name 'handle_update' ...`

- [ ] **Step 3: Implement offset persistence and `handle_update`**

In `agent_from_scratch/bots/telegram_bot.py`, replace the top import block with:

```python
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
```

Append to the end of the file:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `Ran 30 tests ... OK`

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `python -m unittest discover -s agent_from_scratch/tests -q`
Expected: `Ran 280 tests ... OK` (250 baseline + 30 in `test_telegram_bot.py` so far)

- [ ] **Step 6: Commit**

```bash
git add agent_from_scratch/bots/telegram_bot.py agent_from_scratch/tests/test_telegram_bot.py
git commit -m "feat: dispatch Telegram commands and turns through the agent" --author="claude <claude@noreply>"
```

---

### Task 4: Batch processing, the poll loop, and env config

**Files:**
- Modify: `agent_from_scratch/bots/telegram_bot.py`
- Test: `agent_from_scratch/tests/test_telegram_bot.py`

**Interfaces:**
- Consumes: `handle_update`, `load_offset`, `save_offset`, `TelegramAPIError` from earlier tasks.
- Produces: `DEFAULT_STATE_DIR: str`; `BotConfig` (frozen dataclass: `token: str`,
  `allowed_user_id: int`, `state_dir: str`); `load_config_from_env(env: Mapping[str, str]) -> BotConfig`;
  `process_updates(updates: list[dict], *, agent, store, client, allowed_user_id, active_sessions, compact_pending, offset_path: Path) -> int | None`;
  `poll_loop(agent: Agent, store: SessionStore, client: Any, allowed_user_id: int, *, state_dir: str, backoff_seconds: float = 5.0) -> None`.

- [ ] **Step 1: Write the failing tests**

Add to the top imports of `agent_from_scratch/tests/test_telegram_bot.py`:

```python
from unittest.mock import Mock, patch

from agent_from_scratch.bots.telegram_bot import (
    BotConfig, DEFAULT_STATE_DIR, load_config_from_env, poll_loop, process_updates,
)
```

Append to the end of the file:

```python
class LoadConfigFromEnvTests(unittest.TestCase):
    def test_reads_required_and_optional_values(self):
        env = {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_ALLOWED_USER_ID": "99",
               "TELEGRAM_STATE_DIR": "/tmp/x"}
        config = load_config_from_env(env)
        self.assertEqual(config, BotConfig(token="tok", allowed_user_id=99, state_dir="/tmp/x"))

    def test_state_dir_defaults_when_absent(self):
        env = {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_ALLOWED_USER_ID": "99"}
        config = load_config_from_env(env)
        self.assertEqual(config.state_dir, DEFAULT_STATE_DIR)

    def test_raises_when_token_missing(self):
        with self.assertRaises(ValueError):
            load_config_from_env({"TELEGRAM_ALLOWED_USER_ID": "99"})

    def test_raises_when_allowed_user_id_is_not_an_integer(self):
        with self.assertRaises(ValueError):
            load_config_from_env({"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_ALLOWED_USER_ID": "abc"})


class ProcessUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state_dir = self.directory.name
        self.agent = Agent(Mock(spec=LLM), state_dir=self.state_dir)
        self.agent.run_turn = Mock(return_value=TurnResult(
            messages=[], stop_reason=RunStopReason.FINAL_RESPONSE, final_answer="ok"))
        self.store = SessionStore(Path(self.state_dir, "sessions"))
        self.client = Mock()
        self.offset_path = Path(self.state_dir, "offset.txt")

    def test_advances_offset_and_persists_it(self):
        updates = [_text_update(1, "hi", update_id=10), _text_update(1, "there", update_id=11)]
        next_offset = process_updates(updates, agent=self.agent, store=self.store,
                                      client=self.client, allowed_user_id=99,
                                      active_sessions={}, compact_pending={},
                                      offset_path=self.offset_path)
        self.assertEqual(next_offset, 12)
        self.assertEqual(load_offset(self.offset_path), 12)
        self.assertEqual(self.client.send_message.call_count, 2)

    def test_empty_batch_returns_none_and_leaves_offset_untouched(self):
        result = process_updates([], agent=self.agent, store=self.store, client=self.client,
                                 allowed_user_id=99, active_sessions={}, compact_pending={},
                                 offset_path=self.offset_path)
        self.assertIsNone(result)
        self.assertFalse(self.offset_path.exists())

    def test_one_bad_update_does_not_stop_the_batch(self):
        self.agent.run_turn = Mock(side_effect=[
            RuntimeError("boom"),
            TurnResult(messages=[], stop_reason=RunStopReason.FINAL_RESPONSE, final_answer="ok"),
        ])
        updates = [_text_update(1, "first", update_id=1), _text_update(1, "second", update_id=2)]
        next_offset = process_updates(updates, agent=self.agent, store=self.store,
                                      client=self.client, allowed_user_id=99,
                                      active_sessions={}, compact_pending={},
                                      offset_path=self.offset_path)
        self.assertEqual(next_offset, 3)
        self.assertEqual(self.client.send_message.call_count, 2)


class PollLoopTests(unittest.TestCase):
    def test_retries_after_a_telegram_api_error(self):
        client = Mock()
        client.get_updates.side_effect = [TelegramAPIError("boom"), StopIteration]
        with TemporaryDirectory() as directory:
            agent = Agent(Mock(spec=LLM), state_dir=directory)
            store = SessionStore(Path(directory, "sessions"))
            with patch("agent_from_scratch.bots.telegram_bot.time.sleep") as sleep_mock:
                with self.assertRaises(StopIteration):
                    poll_loop(agent, store, client, 99, state_dir=directory, backoff_seconds=0.01)
            sleep_mock.assert_called_once_with(0.01)
        self.assertEqual(client.get_updates.call_count, 2)

    def test_retries_after_a_network_error(self):
        client = Mock()
        client.get_updates.side_effect = [httpx.ConnectError("offline"), StopIteration]
        with TemporaryDirectory() as directory:
            agent = Agent(Mock(spec=LLM), state_dir=directory)
            store = SessionStore(Path(directory, "sessions"))
            with patch("agent_from_scratch.bots.telegram_bot.time.sleep"):
                with self.assertRaises(StopIteration):
                    poll_loop(agent, store, client, 99, state_dir=directory, backoff_seconds=0.01)
        self.assertEqual(client.get_updates.call_count, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `ImportError: cannot import name 'process_updates' ...`

- [ ] **Step 3: Implement batch processing, the poll loop, and env config**

In `agent_from_scratch/bots/telegram_bot.py`, replace the top import block with:

```python
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
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
DEFAULT_STATE_DIR = "./outputs/telegram_sessions"
```

Append to the end of the file:

```python
@dataclass(frozen=True)
class BotConfig:
    token: str
    allowed_user_id: int
    state_dir: str


def load_config_from_env(env: Mapping[str, str]) -> BotConfig:
    token = env.get("TELEGRAM_BOT_TOKEN")
    allowed_raw = env.get("TELEGRAM_ALLOWED_USER_ID")
    if not token or not allowed_raw:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_ID must be set.")
    try:
        allowed_user_id = int(allowed_raw)
    except ValueError as error:
        raise ValueError(f"TELEGRAM_ALLOWED_USER_ID must be an integer: {error}") from error
    state_dir = env.get("TELEGRAM_STATE_DIR", DEFAULT_STATE_DIR)
    return BotConfig(token=token, allowed_user_id=allowed_user_id, state_dir=state_dir)


def process_updates(updates: list[dict[str, Any]], *, agent: Agent, store: SessionStore,
                    client: Any, allowed_user_id: int, active_sessions: dict[Any, str],
                    compact_pending: dict[Any, bool], offset_path: Path) -> int | None:
    """Handle one batch of updates; return the next getUpdates offset, or None if unchanged."""
    next_offset = None
    for update in updates:
        try:
            handle_update(update, agent=agent, store=store, client=client,
                          allowed_user_id=allowed_user_id, active_sessions=active_sessions,
                          compact_pending=compact_pending)
        except Exception:
            logger.exception("Failed to handle update {}", update.get("update_id"))
        next_offset = update["update_id"] + 1
        save_offset(offset_path, next_offset)
    return next_offset


def poll_loop(agent: Agent, store: SessionStore, client: Any, allowed_user_id: int,
             *, state_dir: str, backoff_seconds: float = 5.0) -> None:
    offset_path = Path(state_dir, "telegram_offset.txt")
    offset = load_offset(offset_path)
    active_sessions: dict[Any, str] = {}
    compact_pending: dict[Any, bool] = {}
    while True:
        try:
            updates = client.get_updates(offset)
        except (httpx.HTTPError, TelegramAPIError) as error:
            logger.warning("getUpdates failed, retrying: {}", error)
            time.sleep(backoff_seconds)
            continue
        new_offset = process_updates(updates, agent=agent, store=store, client=client,
                                     allowed_user_id=allowed_user_id,
                                     active_sessions=active_sessions,
                                     compact_pending=compact_pending, offset_path=offset_path)
        if new_offset is not None:
            offset = new_offset
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest agent_from_scratch.tests.test_telegram_bot -v`
Expected: `Ran 39 tests ... OK`

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `python -m unittest discover -s agent_from_scratch/tests -q`
Expected: `Ran 289 tests ... OK` (250 baseline + 39 in `test_telegram_bot.py`)

- [ ] **Step 6: Commit**

```bash
git add agent_from_scratch/bots/telegram_bot.py agent_from_scratch/tests/test_telegram_bot.py
git commit -m "feat: add the Telegram poll loop, retry, and env config" --author="claude <claude@noreply>"
```

---

### Task 5: `main()` entrypoint and Mac mini deployment files

**Files:**
- Modify: `agent_from_scratch/bots/telegram_bot.py`
- Create: `agent_from_scratch/scripts/run_telegram_bot.sh`
- Create: `agent_from_scratch/deploy/telegram-bot.env.example`
- Create: `agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist`
- Create: `agent_from_scratch/docs/telegram_bot_setup.md`

**Interfaces:**
- Consumes: `load_config_from_env`, `poll_loop`, `TelegramClient` from earlier tasks; `LLM`
  (`agent_from_scratch/llm.py:99`), `Agent` (`agent_from_scratch/agent.py:60`),
  `InstructionConfig` (`agent_from_scratch/context.py:24`), `SessionStore`
  (`agent_from_scratch/session.py:20`), `default_workspace` = `workspace` from
  `agent_from_scratch/tools/register.py:12`.
- Produces: `main() -> None`, invoked by `if __name__ == "__main__":`, run as
  `python3 -m agent_from_scratch.bots.telegram_bot`.

`main()` is thin process wiring (env parsing already covered by `load_config_from_env`'s tests,
model/network construction cannot be unit tested without a real model and a real bot token) —
consistent with `agent.py`'s own untested `if __name__ == "__main__":` block. No new unit tests
in this task; the deliverable is verified by linting the two deploy files and a manual smoke
test against a real bot (documented in the runbook, not run by this plan).

- [ ] **Step 1: Implement `main()`**

In `agent_from_scratch/bots/telegram_bot.py`, replace the top import block with:

```python
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from ..agent import Agent
from ..context import InstructionConfig, InstructionLoadError
from ..llm import LLM
from ..session import SessionStore
from ..tools.register import workspace as default_workspace
from ..trace import RunStopReason, TurnResult

DEFAULT_API_BASE = "https://api.telegram.org"
LONG_POLL_TIMEOUT = 30
MAX_MESSAGE_LENGTH = 4096
_TRUNCATION_MARKER = "\n… [truncated]"
SESSION_RESET_COMMANDS = {"/new", "/reset"}
DEFAULT_STATE_DIR = "./outputs/telegram_sessions"
```

Append to the end of the file:

```python
def main() -> None:
    try:
        config = load_config_from_env(os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    Path(config.state_dir).mkdir(parents=True, exist_ok=True)
    llm = LLM()
    agent = Agent(llm, state_dir=config.state_dir,
                  instruction_config=InstructionConfig(workspace=default_workspace))
    store = SessionStore(Path(config.state_dir, "sessions"))
    client = TelegramClient(config.token)
    try:
        poll_loop(agent, store, client, config.allowed_user_id, state_dir=config.state_dir)
    finally:
        client.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full suite to confirm no regression**

Run: `python -m unittest discover -s agent_from_scratch/tests -q`
Expected: `OK` (this step adds no new tests; it only confirms `main()` didn't break imports)

- [ ] **Step 3: Write the launch wrapper script**

Create `agent_from_scratch/scripts/run_telegram_bot.sh`:

```sh
#!/bin/sh
# Loads local secrets, then runs the Telegram bot.
# Copy deploy/telegram-bot.env.example to deploy/telegram-bot.env (untracked) and fill in
# real values before using this script.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE="$SCRIPT_DIR/deploy/telegram-bot.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi
cd "$SCRIPT_DIR/.."
exec python3 -m agent_from_scratch.bots.telegram_bot
```

Run: `chmod +x agent_from_scratch/scripts/run_telegram_bot.sh`

Verify syntax — run: `sh -n agent_from_scratch/scripts/run_telegram_bot.sh`
Expected: no output, exit code 0

- [ ] **Step 4: Write the env template**

Create `agent_from_scratch/deploy/telegram-bot.env.example`:

```sh
# Copy this file to deploy/telegram-bot.env (untracked — do not commit real values) and fill
# in the values below. See docs/telegram_bot_setup.md for how to obtain them.

# Bot token from @BotFather.
TELEGRAM_BOT_TOKEN=

# Your own numeric Telegram user ID (e.g. from @userinfobot). Only this user's messages
# are ever executed; every other sender is silently ignored.
TELEGRAM_ALLOWED_USER_ID=

# Optional. Defaults to ./outputs/telegram_sessions (relative to the repository root).
TELEGRAM_STATE_DIR=
```

- [ ] **Step 5: Write the launchd plist**

Create `agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.shakewingo.agent-telegram-bot</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <!-- Replace with the absolute path to this repo on the Mac mini. -->
        <string>/REPLACE/WITH/REPO/PATH/agent_from_scratch/scripts/run_telegram_bot.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/REPLACE/WITH/REPO/PATH/agent_from_scratch/outputs/telegram_bot.log</string>
    <key>StandardErrorPath</key>
    <string>/REPLACE/WITH/REPO/PATH/agent_from_scratch/outputs/telegram_bot.log</string>
</dict>
</plist>
```

Verify syntax — run: `plutil -lint agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist`
Expected: `agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist: OK`

- [ ] **Step 6: Write the setup runbook**

Create `agent_from_scratch/docs/telegram_bot_setup.md`:

```markdown
# Telegram bot setup (Mac mini)

One-time setup to run the agent as a Telegram bot on the always-on Mac mini. Design:
[superpowers/specs/2026-09-22-telegram-bot-design.md](superpowers/specs/2026-09-22-telegram-bot-design.md).

## 1. Get the repo and weights onto the Mac mini

Clone/copy this repository, then copy the GGUF weights to the path `config.py` expects
(`MODEL_PATH`) — about 4.7 GB. Install the same Python environment used for local development
(`llama-cpp-python`, `httpx`, `loguru`, and the rest of `requirements-tools.txt`).

## 2. Create a Telegram bot and find your user ID

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, follow the
   prompts. Save the token it gives you.
2. Message [@userinfobot](https://t.me/userinfobot) to get your own numeric Telegram user ID.

## 3. Configure secrets

```sh
cp agent_from_scratch/deploy/telegram-bot.env.example agent_from_scratch/deploy/telegram-bot.env
```

Edit `agent_from_scratch/deploy/telegram-bot.env` and fill in `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_ALLOWED_USER_ID`. This file is untracked (matches the existing `agent_from_scratch/docs/`
gitignore pattern's spirit — never commit it).

## 4. Try it once in the foreground

```sh
cd /path/to/repo
sh agent_from_scratch/scripts/run_telegram_bot.sh
```

Message your bot from Telegram. Confirm you get a reply, `/new` starts a fresh session, and
(if you can test from a second account) a non-allowlisted sender gets no reply at all.
Stop with Ctrl-C.

## 5. Install the launchd job

```sh
cp agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist ~/Library/LaunchAgents/
# Edit ~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist:
# replace /REPLACE/WITH/REPO/PATH with the absolute path to this repo on this machine.
launchctl load ~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist
```

Check it's running and tail the log:

```sh
launchctl list | grep agent-telegram-bot
tail -f agent_from_scratch/outputs/telegram_bot.log
```

## 6. Everyday use

- `/new` — start a fresh session in this chat.
- `/reset` — clear the current session's history.
- `/session <id>` — switch this chat to a named session.
- `/compact` — summarize history before the next message.
- To stop the service: `launchctl unload ~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist`.
- To restart after a code change: unload, then `launchctl load` again.

## Energy Saver

The Mac mini must not sleep for the bot to stay reachable. System Settings → Energy Saver
(or Battery, on a laptop) → set "Prevent automatic sleeping when the display is off" while
plugged in. Screen sleep is fine; system sleep is not.
```

- [ ] **Step 7: Commit**

```bash
git add agent_from_scratch/bots/telegram_bot.py agent_from_scratch/scripts/run_telegram_bot.sh \
       agent_from_scratch/deploy/telegram-bot.env.example \
       agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist \
       agent_from_scratch/docs/telegram_bot_setup.md
git commit -m "feat: add the Telegram bot entrypoint and Mac mini deployment files" --author="claude <claude@noreply>"
```

---

## Final check

- [ ] Run the full suite once more from the worktree root:
  `python -m unittest discover -s agent_from_scratch/tests -q`
  Expected: `Ran 289 tests ... OK` (250 baseline + 39 in `test_telegram_bot.py`).
- [ ] Confirm no file outside `agent_from_scratch/bots/`, `agent_from_scratch/tests/`,
  `agent_from_scratch/scripts/`, `agent_from_scratch/deploy/`, and `agent_from_scratch/docs/`
  changed: `git diff --stat main`. The only changes outside those directories should be the
  `docs/STAGE.md` and `docs/superpowers/specs/2026-09-22-telegram-bot-design.md` files already
  committed before this plan.
- [ ] Report remaining manual-only step to the user: actually running the smoke test in
  `docs/telegram_bot_setup.md` step 4 on the physical Mac mini, since no CI here has a real
  Telegram bot token or the Mac mini's hardware.
