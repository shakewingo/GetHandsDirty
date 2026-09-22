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
