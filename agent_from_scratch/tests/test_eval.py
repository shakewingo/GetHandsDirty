import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.evals.foundation import measure
from agent_from_scratch.evals.verify import metrics
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType
from agent_from_scratch.tools.base import ToolCall, ToolRegistry
from agent_from_scratch.evals.legacy_files import ReadFileTool
from agent_from_scratch.trace import ModelRequest, ModelRequestStatus, TurnResult


class ReadEvaluationTests(unittest.TestCase):
    def test_blocked_requests_do_not_count_as_model_calls_or_missing_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        completed = ModelRequest(1, 2, status=ModelRequestStatus.COMPLETED, usage=usage)
        blocked = ModelRequest(2, 4, status=ModelRequestStatus.BLOCKED,
                               context={"count_method": "exact", "prompt_tokens": 9000})
        for requests, count, expected_usage in (
            ([blocked], 0, dict.fromkeys(usage, 0)),
            ([completed, blocked], 1, usage),
        ):
            with self.subTest(count=count):
                result = TurnResult(messages=[], model_requests=requests)
                before = asdict(result)
                foundation = measure(result, b"", "a.txt", 0)
                behavior = metrics(result)
                for measured in (foundation, behavior):
                    self.assertEqual(measured["model_requests"], count)
                    self.assertEqual(measured["usage"], expected_usage)
                    self.assertEqual(measured["parse_errors"], 0)
                self.assertEqual(foundation["max_prompt_tokens"], 100 if count else 0)
                self.assertEqual(behavior["requests_without_usage"], 0)
                self.assertEqual(asdict(result), before)

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
        self.registry = ToolRegistry([ReadFileTool(self.workspace, max_bytes=4)])

    def observations(self, *offsets, path="a.txt"):
        messages = []
        for index, offset in enumerate(offsets):
            response = LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('read_file', {'path': path, 'offset': offset}, str(index))])
            result = self.registry.invoke("read_file", response.tool_calls[0].arguments, str(index))
            messages.extend([response.to_message(), {"role": "tool", "tool_call_id": str(index),
                                                      "content": json.dumps(asdict(result))}])
        return messages

    def score(self, messages, history=None):
        history = history or []
        result = TurnResult(messages=[{"role": "system", "content": "System"}, *history,
                                      {"role": "user", "content": "Read a.txt"}, *messages])
        return measure(result, self.target.read_bytes(), "a.txt", len(history))

    def test_first_or_last_chunk_is_not_full_coverage(self):
        for offset, count in ((0, 4), (8, 2)):
            result = self.score(self.observations(offset))
            self.assertFalse(result["read_coverage_passed"])
            self.assertEqual(result["covered_bytes"], count)

    def test_overlapping_out_of_order_reads_count_once(self):
        result = self.score(self.observations(8, 4, 0, 0, 2))
        self.assertTrue(result["read_coverage_passed"])
        self.assertEqual(result["covered_bytes"], 10)

    def test_empty_file_requires_a_real_read(self):
        self.target.write_bytes(b"")
        self.assertFalse(self.score([])["read_coverage_passed"])
        self.assertTrue(self.score(self.observations(0))["read_coverage_passed"])

    def test_utf8_counts_bytes_and_partial_task_need_not_reach_eof(self):
        self.target.write_text("你🙂好")
        self.assertTrue(self.score(self.observations(0, 3, 7))["read_coverage_passed"])
        self.target.write_bytes(b"x" * 1030)
        self.registry = ToolRegistry([ReadFileTool(self.workspace, max_bytes=1024)])
        result = self.score(self.observations(0))
        self.assertTrue(result["partial_read_passed"])
        self.assertFalse(result["read_coverage_passed"])

    def test_history_other_files_and_failed_calls_do_not_count(self):
        (self.workspace / "b.txt").write_text("abcdefghij")
        for messages in (self.observations(0, 4, 8, path="b.txt"), self.observations(20), []):
            result = self.score(messages, history=self.observations(0, 4, 8))
            self.assertEqual(result["covered_bytes"], 0)
            self.assertFalse(result["read_coverage_passed"])

    def test_mismatched_content_ranges_and_unpaired_reads_fail_scoring(self):
        for field, value in (("content", "WRONG"), ("next_offset", 7), ("eof", True), ("offset", 1)):
            messages = self.observations(0, 4, 8)
            observed = json.loads(messages[1]["content"])
            observed["output"][field] = value
            messages[1]["content"] = json.dumps(observed)
            with self.subTest(field=field):
                self.assertFalse(self.score(messages)["read_coverage_passed"])
        self.assertFalse(self.score(self.observations(0, 4, 8)[1:])["read_coverage_passed"])

    def test_early_answer_ends_turn_and_only_eval_reports_missing_coverage(self):
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
        before = asdict(result)
        self.assertFalse(measure(result, self.target.read_bytes(), "a.txt", 0)["read_coverage_passed"])
        self.assertEqual(asdict(result), before)

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
