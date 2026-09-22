from __future__ import annotations

import unittest
from unittest.mock import Mock

from agent_from_scratch.bots.telegram_bot import (
    MAX_MESSAGE_LENGTH, TelegramAPIError, TelegramClient,
    extract_message, format_reply, is_allowed, truncate_for_telegram,
)
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


if __name__ == "__main__":
    unittest.main()
