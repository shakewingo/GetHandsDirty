"""The vLLM backend renders the project template itself and parses raw text like the local one."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import httpx

from agent_from_scratch.config import BENCH_DECODING, BENCH_MODEL, BENCH_N_CTX, CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.llm import (ResponseError, ResponseErrorCode, VLLMClient,
                                    compile_chat_template, render_chat)
from agent_from_scratch.tools.register import default_tool_schemas

MODEL = BENCH_MODEL["id"]
CONVERSATION = [
    {"role": "system", "content": "System rules."},
    {"role": "user", "content": "Read a.json — then <stop>."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "a.json"})}}]},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": True, "output": "1| {\"a\": \"<x>\"}"})},
    {"role": "assistant", "content": "Done: 值"},
    {"role": "user", "content": "Again."},
]


class FakeServer:
    """A scripted OpenAI-compatible server: /v1/models, /tokenize and /v1/completions."""

    def __init__(self, replies, *, max_model_len=BENCH_N_CTX, failures=0, adapters=()):
        self.replies, self.failures, self.requests = list(replies), failures, []
        self.models = [{"id": MODEL, "root": MODEL, "max_model_len": max_model_len}] + [
            {"id": name, "root": "/adapters/" + name, "parent": MODEL} for name in adapters]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": self.models})
        if request.url.path == "/tokenize":
            # One token per special marker, one per remaining character: enough to be exact.
            text = body["prompt"]
            specials = text.count("<|im_start|>") + text.count("<|im_end|>")
            rest = len(text.replace("<|im_start|>", "").replace("<|im_end|>", ""))
            return httpx.Response(200, json={"count": specials + rest})
        if self.failures:
            self.failures -= 1
            return httpx.Response(503, text="warming up")
        text, finish = self.replies.pop(0)
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": finish}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                                   "total_tokens": 15}})


def client(server: FakeServer, **kwargs) -> VLLMClient:
    decoding = dict(BENCH_DECODING)
    temperature, max_tokens = decoding.pop("temperature"), decoding.pop("max_tokens")
    http = httpx.Client(base_url="http://vllm.test", transport=httpx.MockTransport(server))
    return VLLMClient("http://vllm.test", MODEL, client=http, temperature=temperature,
                      max_tokens=max_tokens, sampling=decoding, n_ctx=BENCH_N_CTX, **kwargs)


def tool_text(name: str, **arguments) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


class RenderTests(unittest.TestCase):
    def test_render_matches_llama_cpp_formatter_byte_for_byte(self):
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        except ImportError:
            self.skipTest("llama_cpp not installed")
        template = CHAT_TEMPLATE_PATH.read_text()
        formatter = Jinja2ChatFormatter(template=template, eos_token="<|im_end|>",
                                        bos_token="<|endoftext|>")
        for tools in (default_tool_schemas, {}):
            with self.subTest(tools=bool(tools)):
                expected = formatter(messages=CONVERSATION, tools=list(tools.values()),
                                     tool_choice="auto").prompt
                self.assertEqual(render_chat(compile_chat_template(template), CONVERSATION, tools),
                                 expected)


class ClientTests(unittest.TestCase):
    def test_generate_sends_the_rendered_prompt_and_frozen_sampling(self):
        server = FakeServer([(tool_text("read_file", path="a.json"), "stop")])
        model = client(server, seed=2)
        response = model.generate(CONVERSATION, default_tool_schemas)
        self.assertEqual([(c.name, c.arguments) for c in response.tool_calls], [("read_file", {"path": "a.json"})])
        self.assertEqual(response.usage, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        path, body = server.requests[-1]
        self.assertEqual(path, "/v1/completions")
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["prompt"], render_chat(compile_chat_template(CHAT_TEMPLATE_PATH.read_text()),
                                                     CONVERSATION, default_tool_schemas))
        self.assertEqual(body["seed"], 2)
        for key, value in BENCH_DECODING.items():
            self.assertEqual(body[key], value)

    def test_adapter_is_the_request_model_and_must_be_served(self):
        server = FakeServer([("DONE", "stop")], adapters=("sft-r1",))
        self.assertEqual(client(server, adapter="sft-r1").generate(CONVERSATION, {}).content, "DONE")
        self.assertEqual(server.requests[-1][1]["model"], "sft-r1")
        with self.assertRaisesRegex(ValueError, "adapter"):
            client(FakeServer([]), adapter="sft-r1")

    def test_measure_context_counts_with_the_server_tokenizer(self):
        model = client(FakeServer([]))
        measured = model.measure_context(CONVERSATION, {}, max_tokens=100)
        self.assertEqual(measured["count_method"], "exact")
        self.assertEqual(measured["remaining_tokens"], BENCH_N_CTX - measured["prompt_tokens"] - 100)

    def test_truncation_and_bad_tool_text_are_response_errors(self):
        model = client(FakeServer([("partial", "length"), ("<tool_call>\n{broken\n</tool_call>", "stop")]))
        for code in (ResponseErrorCode.TRUNCATED_RESPONSE, ResponseErrorCode.INVALID_TOOL_CALL):
            with self.subTest(code=code), self.assertRaises(ResponseError) as caught:
                model.generate(CONVERSATION, {})
            self.assertEqual(caught.exception.code, code)

    def test_server_errors_are_retried_and_a_small_window_is_refused(self):
        server = FakeServer([("DONE", "stop")], failures=2)
        self.assertEqual(client(server).generate(CONVERSATION, {}).content, "DONE")
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            client(FakeServer([], max_model_len=8192))

    def test_settings_satisfy_the_frozen_protocol(self):
        from agent_from_scratch.evals.bench.manifest import protocol_mismatch
        self.assertEqual(protocol_mismatch(client(FakeServer([])).settings(), 0), [])
        local = {"backend": "llama_cpp", "temperature": 0.0, "max_tokens": 2048, "n_ctx": 32768}
        self.assertTrue(protocol_mismatch(local, 0))
        self.assertTrue(protocol_mismatch(client(FakeServer([])).settings(), 7))

    def test_a_bench_task_runs_end_to_end_through_the_http_backend(self):
        skeleton, ctx = next((s, c) for s, c in specs("dev") if s.name == "single_field_edit")
        directory = Path(self.enterContext(TemporaryDirectory()))
        task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), directory, ctx)
        expected = json.loads(task.expect.files["config.json"])
        server = FakeServer([(tool_text("read_file", path="config.json"), "stop"),
                             (tool_text("write_file", path="config.json", content=json.dumps(expected)), "stop"),
                             ("DONE", "stop")])
        record = run_case(client(server, seed=0), skeleton, ctx, directory / "run")
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(record["usage"]["total_tokens"], 45)


if __name__ == "__main__":
    unittest.main()
