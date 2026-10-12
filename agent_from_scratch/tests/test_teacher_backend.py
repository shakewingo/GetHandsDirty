"""The teacher backend speaks DeepSeek's chat API, shares the one parser, and budgets in student tokens."""

import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from agent_from_scratch.config import BENCH_DECODING, BENCH_N_CTX, CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.bench import run
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.sft.export import TRACK_BACKENDS, examples_from_trace, rejection
from agent_from_scratch.evals.sft.teacher import TeacherClient
from agent_from_scratch.llm import ResponseError, ResponseErrorCode, compile_chat_template, render_chat
from agent_from_scratch.tools.register import default_tool_schemas

TEMPLATE = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
CONVERSATION = [
    {"role": "system", "content": "System rules."},
    {"role": "user", "content": "Read both files."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "a.json"})}},
        {"id": "t2", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "b.json"})}}]},
    {"role": "tool", "tool_call_id": "t1", "content": json.dumps({"ok": False, "error_message": "Temporary failure."})},
    {"role": "tool", "tool_call_id": "t2", "content": json.dumps({"ok": False, "error_code": "skipped"})},
    {"role": "user", "content": "[Runtime feedback] Try again."},
]


def count(text: str) -> int:
    """A stand-in student tokenizer: one token per control marker, one per other character."""
    specials = text.count("<|im_start|>") + text.count("<|im_end|>")
    return specials + len(text.replace("<|im_start|>", "").replace("<|im_end|>", ""))


def reply(content="", *calls, reasoning="Thinking it through.", finish=None):
    """One chat completion as the API returns it; `calls` are (id, name, arguments)."""
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning,
               "tool_calls": [{"index": index, "id": call_id, "type": "function",
                               "function": {"name": name, "arguments": json.dumps(arguments)}}
                              for index, (call_id, name, arguments) in enumerate(calls)] or None}
    return {"model": "deepseek-flash",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170,
                      "prompt_cache_hit_tokens": 50, "prompt_cache_miss_tokens": 100,
                      "completion_tokens_details": {"reasoning_tokens": 6}}}


class FakeDeepSeek:
    """A scripted server: GET /models and POST /chat/completions."""

    def __init__(self, replies=(), *, failures=0):
        self.replies, self.failures, self.sent = list(replies), failures, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash"}]})
        if self.failures:
            self.failures -= 1
            return httpx.Response(503, text="busy")
        self.sent.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))


def teacher(server: FakeDeepSeek, model: str = "deepseek-flash", **kwargs) -> TeacherClient:
    http = httpx.Client(base_url="https://deepseek.test", transport=httpx.MockTransport(server))
    return TeacherClient(model, client=http, **{"count": count, "budget_tokenizer": "fake", "backoff": 0, **kwargs})


class ConversionTests(unittest.TestCase):
    def test_request_passes_the_harness_messages_and_schemas_through(self):
        server = FakeDeepSeek([reply("Done.")])
        teacher(server).generate(CONVERSATION, default_tool_schemas)
        sent = server.sent[0]
        self.assertEqual(sent["model"], "deepseek-flash")
        self.assertEqual(sent["messages"], CONVERSATION)
        self.assertEqual(sent["tools"], list(default_tool_schemas.values()))

    def test_request_sets_thinking_effort_and_no_sampling_parameters(self):
        server = FakeDeepSeek([reply("Done."), reply("Done.")])
        teacher(server, effort="high").generate(CONVERSATION, default_tool_schemas, max_tokens=512)
        sent = server.sent[0]
        for unused in ("temperature", "top_p", "top_k", "seed", "tool_choice"):
            self.assertNotIn(unused, sent)
        self.assertEqual((sent["thinking"], sent["reasoning_effort"]), ({"type": "enabled"}, "high"))
        self.assertEqual(sent["max_tokens"], 8192)  # thinking counts toward it; not the student's reserve
        teacher(server, effort="none").generate(CONVERSATION, default_tool_schemas)
        self.assertEqual(server.sent[1]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", server.sent[1])

    def test_reply_goes_through_the_shared_parser(self):
        server = FakeDeepSeek([reply("Reading.", ("call_00_x", "read_file", {"path": "a.json"}))])
        response = teacher(server).generate(CONVERSATION[:2], default_tool_schemas)
        self.assertEqual([(c.call_id, c.name, c.arguments) for c in response.tool_calls],
                         [("call_00_x", "read_file", {"path": "a.json"})])
        self.assertEqual((response.content, response.finish_reason), ("Reading.", "tool_calls"))
        self.assertEqual(response.usage, {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170})
        self.assertEqual(response.raw_response["model"], "deepseek-flash")
        self.assertEqual(response.raw_response["usage"]["prompt_cache_hit_tokens"], 50)
        json.dumps(response.raw_response)  # the trace must be able to store it

    def test_own_turns_are_replayed_with_their_reasoning(self):
        first = reply("", ("call_00_x", "read_file", {"path": "a.json"}), reasoning="Read a.json first.")
        server = FakeDeepSeek([first, reply("Done.")])
        model = teacher(server)
        message = model.generate(CONVERSATION[:2], default_tool_schemas).to_message()
        model.generate([*CONVERSATION[:2], message,
                        {"role": "tool", "tool_call_id": "call_00_x", "content": json.dumps({"ok": True})}],
                       default_tool_schemas)
        replayed = server.sent[1]["messages"][2]
        self.assertEqual(replayed, {**message, "reasoning_content": "Read a.json first."})
        # A turn this client did not produce has no reasoning to return, and none is invented.
        self.assertNotIn("reasoning_content", server.sent[0]["messages"][1])

    def test_finish_reasons_reach_the_shared_parser(self):
        cases = {"length": ResponseErrorCode.TRUNCATED_RESPONSE,
                 "content_filter": ResponseErrorCode.UNSUPPORTED_FINISH_REASON,
                 "insufficient_system_resource": ResponseErrorCode.UNSUPPORTED_FINISH_REASON}
        for finish, code in cases.items():
            with self.subTest(finish=finish), self.assertRaises(ResponseError) as caught:
                teacher(FakeDeepSeek([reply("partial", finish=finish)])).generate(
                    CONVERSATION[:2], default_tool_schemas)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.raw_response["usage"]["prompt_tokens"], 150)
            self.assertEqual(caught.exception.raw_response["choices"][0]["finish_reason"], finish)

    def test_a_reply_without_a_choice_is_an_invalid_response(self):
        with self.assertRaises(ResponseError) as caught:
            teacher(FakeDeepSeek([{"error": "unexpected"}])).generate(CONVERSATION[:2], default_tool_schemas)
        self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_RESPONSE)
        self.assertEqual(caught.exception.raw_response, {"error": "unexpected"})

    def test_budget_is_the_student_token_count_of_the_student_rendering(self):
        server = FakeDeepSeek()
        measured = teacher(server).measure_context(CONVERSATION, default_tool_schemas)
        tokens = count(render_chat(TEMPLATE, CONVERSATION, default_tool_schemas))
        self.assertEqual(measured, {"count_method": "exact", "prompt_tokens": tokens,
                                    "window_tokens": BENCH_N_CTX,
                                    "response_reserve": BENCH_DECODING["max_tokens"],
                                    "remaining_tokens": BENCH_N_CTX - tokens - BENCH_DECODING["max_tokens"]})
        self.assertEqual(server.sent, [])  # counted locally; the API has no count endpoint

    def test_settings_identify_the_teacher(self):
        model = teacher(FakeDeepSeek(), sample=3)
        settings = model.settings()
        self.assertEqual((settings["backend"], settings["model"], settings["model_name"], settings["effort"],
                          settings["sample"], settings["budget_tokenizer"]),
                         ("deepseek", "deepseek-flash", "DeepSeek-V4.1-Flash", "low", 3, "fake"))
        self.assertEqual(settings["backend"], TRACK_BACKENDS["teacher"])
        self.assertIsNone(settings["chat_template_sha256"])  # the teacher never sampled under it
        model.close()
        self.assertTrue(model.http.is_closed)

    def test_an_unserved_model_a_wrong_tokenizer_or_a_missing_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "deepseek-v9"):
            teacher(FakeDeepSeek(), "deepseek-v9")
        with self.assertRaisesRegex(ValueError, "single tokens"):
            teacher(FakeDeepSeek(), count=len)
        with patch.dict(os.environ, clear=True), self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            TeacherClient(count=count, budget_tokenizer="fake")

    def test_server_errors_are_retried(self):
        server = FakeDeepSeek([reply("Done.")], failures=2)
        self.assertEqual(teacher(server).generate(CONVERSATION, default_tool_schemas).content, "Done.")
        with self.assertRaises(RuntimeError):
            teacher(FakeDeepSeek([reply("Done.")], failures=3)).generate(CONVERSATION, default_tool_schemas)


class HarnessTests(unittest.TestCase):
    def scripted(self, skeleton, ctx):
        """The skeleton's scripted solution, spoken as chat completions."""
        with TemporaryDirectory() as temporary:
            task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"),
                                  Path(temporary), ctx)
        return [reply(response.content, *[(f"call_{index:02d}_{n}", c.name, c.arguments)
                                          for n, c in enumerate(response.tool_calls)],
                      reasoning=f"step {index}")
                for index, response in enumerate(skeleton.solution(task))]

    def test_a_teacher_rollout_passes_the_verifier_and_exports(self):
        skeleton, ctx = next((s, c) for s, c in specs("train") if c.id == "points_save_recovery-0-fault")
        server = FakeDeepSeek(self.scripted(skeleton, ctx))
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            record = run_case(teacher(server), skeleton, ctx, out)
            self.assertTrue(record["passed"], record["checks"])
            trace = json.loads((out / "state" / "runs" / f"{record['run_id']}.jsonl").read_text())
            task = json.loads((out / "task.json").read_text())
        self.assertEqual(record["usage"]["prompt_tokens"], 150 * record["model_requests"])
        self.assertIsNone(rejection(record, task, trace))
        examples = examples_from_trace(trace, TEMPLATE)
        self.assertEqual(len(examples), record["model_requests"])
        self.assertTrue(examples[0]["completion"].startswith('<tool_call>\n{"name": "read_file"'))
        self.assertEqual(examples[-1]["completion"], "SAVED<|im_end|>")
        # The budget the harness acted on is the student's count of exactly the exported prompt.
        for example in examples:
            budget = trace["model_requests"][example["request_index"]]["budget"]
            self.assertEqual(budget["prompt_tokens"], count(example["prompt"]))
        # Every earlier turn went back with the reasoning the model produced for it.
        for index, sent in enumerate(server.sent):
            turns = [m for m in sent["messages"] if m["role"] == "assistant"]
            self.assertEqual([m["reasoning_content"] for m in turns], [f"step {n}" for n in range(index)])


class BenchTests(unittest.TestCase):
    def test_the_teacher_may_only_see_the_train_split(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            for split in ("dev", "test", "holdout"):
                args = ["bench", "--backend", "teacher", "--split", split, "--final",
                        "--student-gguf", __file__, "--output", str(Path(temporary) / split)]
                with self.subTest(split=split), patch.object(sys, "argv", args), patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_the_teacher_needs_the_student_tokenizer_file(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            for extra in ([], ["--student-gguf", str(Path(temporary) / "missing.gguf")]):
                args = ["bench", "--backend", "teacher", "--split", "train", *extra,
                        "--output", str(Path(temporary) / "out")]
                with self.subTest(extra=extra), patch.object(sys, "argv", args), patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_bench_builds_the_teacher_backend(self):
        args = SimpleNamespace(backend="teacher", teacher_model="deepseek-flash", teacher_effort="low",
                               seed=4, student_gguf=Path("student.gguf"))
        with patch("agent_from_scratch.evals.sft.teacher.TeacherClient") as client, \
                patch("agent_from_scratch.evals.sft.teacher.student_token_counter",
                      return_value=(count, "Qwen3 (student.gguf)")) as counter:
            run.load_model(args)
        counter.assert_called_once_with(Path("student.gguf"))
        client.assert_called_once_with("deepseek-flash", count=count, budget_tokenizer="Qwen3 (student.gguf)",
                                       effort="low", sample=4, n_ctx=BENCH_N_CTX,
                                       reserve=BENCH_DECODING["max_tokens"])


if __name__ == "__main__":
    unittest.main()
