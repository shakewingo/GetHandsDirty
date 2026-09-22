from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

import httpx

from agent_from_scratch.agent import Agent
from agent_from_scratch.bots.telegram_bot import (
    BotConfig, DEFAULT_STATE_DIR, MAX_MESSAGE_LENGTH, SESSION_RESET_COMMANDS, TelegramAPIError,
    TelegramClient, extract_message, format_reply, handle_update, is_allowed,
    load_config_from_env, load_offset, poll_loop, process_updates, save_offset,
    truncate_for_telegram,
)
from agent_from_scratch.context import InstructionLoadError
from agent_from_scratch.llm import LLM
from agent_from_scratch.session import SessionStore
from agent_from_scratch.trace import RunStopReason, TurnResult


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


def _text_update(chat_id, text, update_id=1, sender_id=99):
    return {"update_id": update_id,
            "message": {"chat": {"id": chat_id}, "from": {"id": sender_id}, "text": text}}


class SessionResetCommandsTests(unittest.TestCase):
    def test_includes_new_and_reset(self):
        self.assertEqual(SESSION_RESET_COMMANDS, {"/new", "/reset"})


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


if __name__ == "__main__":
    unittest.main()
