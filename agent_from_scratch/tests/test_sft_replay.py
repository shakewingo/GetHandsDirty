"""Replay asks the server for a greedy continuation of each prompt and compares it with the target."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from agent_from_scratch.evals.sft import replay as replay_module
from agent_from_scratch.evals.sft.replay import replay

EXAMPLES = [{"id": "e0", "prompt": "P0", "completion": "<tool_call>\n{}\n</tool_call><|im_end|>"},
            {"id": "e1", "prompt": "P1", "completion": "BLOCKED<|im_end|>"}]


class FakeServer:
    """A scripted vLLM server: /v1/models and /v1/completions keyed by prompt and model."""

    def __init__(self, texts: dict, adapters=("sft-smoke",)):
        self.texts, self.adapters, self.sent = texts, adapters, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}] + [{"id": name, "parent": "base"}
                                                                         for name in self.adapters]})
        body = json.loads(request.content)
        self.sent.append(body)
        return httpx.Response(200, json={"choices": [{"text": self.texts[(body["model"], body["prompt"])],
                                                      "finish_reason": "stop"}]})


def client(server: FakeServer) -> httpx.Client:
    return httpx.Client(base_url="http://vllm.test", transport=httpx.MockTransport(server))


TEXTS = {("sft-smoke", "P0"): "<tool_call>\n{}\n</tool_call>", ("sft-smoke", "P1"): "BLOCKED",
         ("base", "P0"): "<tool_call>\n{}\n</tool_call>", ("base", "P1"): "The answer is BLOCKED."}


class ReplayTests(unittest.TestCase):
    def test_requests_are_greedy_and_stop_at_the_end_marker(self):
        server = FakeServer(TEXTS)
        report = replay(EXAMPLES, client(server), "sft-smoke")
        self.assertEqual((report["model"], report["examples"], report["exact"], report["mismatches"]),
                         ("sft-smoke", 2, 2, []))
        self.assertEqual([body["prompt"] for body in server.sent], ["P0", "P1"])
        for body in server.sent:
            self.assertEqual((body["model"], body["temperature"], body["stop"]), ("sft-smoke", 0.0, ["<|im_end|>"]))
            for unused in ("top_p", "top_k", "seed"):
                self.assertNotIn(unused, body)

    def test_a_different_continuation_is_reported_with_both_texts(self):
        report = replay(EXAMPLES, client(FakeServer(TEXTS)), "base")
        self.assertEqual((report["examples"], report["exact"]), (2, 1))
        self.assertEqual(report["mismatches"], [{"id": "e1", "target": "BLOCKED<|im_end|>",
                                                 "generated": "The answer is BLOCKED.", "finish_reason": "stop"}])

    def test_an_adapter_the_server_does_not_serve_is_an_error(self):
        server = FakeServer(TEXTS, adapters=())
        with self.assertRaisesRegex(ValueError, "sft-smoke"):
            replay(EXAMPLES, client(server), "sft-smoke")
        self.assertEqual(server.sent, [])

    def test_command_replays_the_examples_a_training_run_used(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sft").mkdir()
            (root / "sft" / "examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in EXAMPLES))
            (root / "train.json").write_text(json.dumps({"selection": {"example_ids": ["e1"]}}))
            http = client(FakeServer(TEXTS))
            argv = ["replay", "--examples", str(root / "sft"), "--ids", str(root / "train.json"),
                    "--adapter", "sft-smoke", "--output", str(root / "report.json")]
            with patch.object(sys, "argv", argv), patch("builtins.print"), \
                    patch("httpx.Client", lambda **kwargs: http):
                replay_module.main()
                report = json.loads((root / "report.json").read_text())
                self.assertEqual((report["examples"], report["exact"]), (1, 1))
                with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                    replay_module.main()  # the report exists now
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
