import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.evals.verify import metrics
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType
from agent_from_scratch.tools.base import ToolCall, ToolRegistry
from agent_from_scratch.tools.files import ReadFileTool
from agent_from_scratch.trace import (ModelRequest, ModelRequestStatus, RunStopReason,
                                     TurnResult, request_budget)


class ReadEvaluationTests(unittest.TestCase):
    def test_blocked_requests_do_not_count_as_model_calls_or_missing_usage(self):
        usage: dict[str, int | None] = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        completed = ModelRequest(1, 2, status=ModelRequestStatus.COMPLETED, usage=usage)
        blocked = ModelRequest(2, 4, status=ModelRequestStatus.BLOCKED,
                               budget={"count_method": "exact", "prompt_tokens": 9000})
        for requests, count, expected_usage in (
            ([blocked], 0, dict.fromkeys(usage, 0)),
            ([completed, blocked], 1, usage),
        ):
            with self.subTest(count=count):
                result = TurnResult(messages=[], model_requests=requests)
                before = asdict(result)
                behavior = metrics(result)
                self.assertEqual(behavior["model_requests"], count)
                self.assertEqual(behavior["usage"], expected_usage)
                self.assertEqual(behavior["parse_errors"], 0)
                self.assertEqual(behavior["requests_without_usage"], 0)
                self.assertEqual(asdict(result), before)

    def test_metrics_separate_actor_and_summary_requests(self):
        result = TurnResult(messages=[], stop_reason=RunStopReason.FINAL_RESPONSE)
        result.model_requests = [
            ModelRequest(1, 4, purpose="compact", status=ModelRequestStatus.COMPLETED,
                         compact_after={"prompt_tokens": 300}),
            ModelRequest(1, 4, purpose="compact", status=ModelRequestStatus.COMPLETED,
                         error_message="Summary did not produce a smaller fitting actor input."),
            ModelRequest(1, 4, purpose="agent", status=ModelRequestStatus.COMPLETED,
                         budget={"prompt_tokens": 900}),
            ModelRequest(2, 6, purpose="agent", status=ModelRequestStatus.BLOCKED,
                         budget={"prompt_tokens": 9000}),
        ]
        report = metrics(result)
        self.assertEqual(report["model_requests"], 3)
        self.assertEqual(report["actor_requests"], 1)
        self.assertEqual(report["compact_requests"], 2)
        self.assertEqual(report["compactions_applied"], 1)
        self.assertEqual(report["max_actor_prompt_tokens"], 900)

    def test_request_budget_reads_both_schema_names(self):
        self.assertEqual(request_budget({"budget": {"prompt_tokens": 5}}), {"prompt_tokens": 5})
        self.assertEqual(request_budget({"context": {"prompt_tokens": 7}}), {"prompt_tokens": 7})
        self.assertIsNone(request_budget({}))

    def test_actual_request_with_missing_usage_remains_unknown(self):
        result = TurnResult(messages=[], model_requests=[
            ModelRequest(1, 2, status=ModelRequestStatus.MODEL_ERROR),
            ModelRequest(2, 2, status=ModelRequestStatus.BLOCKED),
        ])
        measured = metrics(result)
        self.assertEqual(measured["model_requests"], 1)
        self.assertEqual(measured["requests_without_usage"], 1)
        self.assertTrue(all(v is None for v in measured["usage"].values()))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.target = self.workspace / "a.txt"
        self.target.write_text("abcdefghij")
        self.registry = ToolRegistry([ReadFileTool(self.workspace)])

    def test_early_answer_ends_turn(self):
        model = Mock(spec=LLM)
        model.measure_context.return_value = {  # Scripted fitting budget; no real tokenizer.
            "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
            "response_reserve": 512, "remaining_tokens": 7388,
        }
        model.settings.return_value = {}
        model.generate.side_effect = [
            LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('read_file', {'path': 'a.txt'})]),
            LLMResponse("assistant", "Would you like me to continue?", ResponseType.direct),
        ]
        agent = Agent(model, registry=self.registry)
        result = agent.run_turn("Read a.txt")
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(result.stop_reason, "final_response")
        self.assertEqual(result.final_answer, "Would you like me to continue?")
        self.assertEqual([m["role"] for m in result.messages],
                         ["system", "user", "assistant", "tool", "assistant"])

    def test_repl_preserves_input_without_read_command_rewriting(self):
        model = Mock(spec=LLM)
        model.measure_context.return_value = {  # Scripted fitting budget; no real tokenizer.
            "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
            "response_reserve": 512, "remaining_tokens": 7388,
        }
        model.settings.return_value = {}
        model.generate.return_value = LLMResponse("assistant", "Answer", ResponseType.direct)
        agent = Agent(model, registry=self.registry)
        for text in ("Read a.txt", "/read a file.txt"):
            with patch("builtins.input", side_effect=[text, "exit"]), patch("builtins.print"):
                agent.run_repl()
            self.assertEqual(model.generate.call_args.args[0][1]["content"], text)


if __name__ == "__main__":
    unittest.main()
