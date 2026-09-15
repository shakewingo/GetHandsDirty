import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM
from agent_from_scratch.session import SessionStore
from agent_from_scratch.tools.base import Tool, ToolInterrupted
from agent_from_scratch.tools.files import ReadFileTool, WriteFileTool
from agent_from_scratch.tools.register import ToolRegistry
from agent_from_scratch.tools.shell import ShellTool


def block(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def raw(content, **fields):
    return {"choices": [{"message": {"role": "assistant", "content": content, **fields}, "finish_reason": "stop"}]}


class InterruptTool(Tool):
    name, description = "interrupt", "Interrupt with partial output."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self):
        raise ToolInterrupted({"stdout": "started"})


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.model = LLM.__new__(LLM)
        self.model.llm = Mock()
        self.model.model_path, self.model.temperature, self.model.max_tokens = "fake", 0, 512
        self.model.n_ctx, self.model.n_gpu_layers = 2048, 0
        self.agent = Agent(self.model, str(self.root / "state"), registry=ToolRegistry([
            ReadFileTool(self.workspace), WriteFileTool(self.workspace), ShellTool(self.workspace), InterruptTool(),
        ]))

    def run_script(self, *responses):
        self.model.llm.create_chat_completion.side_effect = responses
        return self.agent.run_turn("Complete the file task.", session_id="batch")

    def observations(self, result):
        calls = [call for message in result.messages for call in message.get("tool_calls", [])]
        observations = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
        self.assertEqual([c["id"] for c in calls], [o["call_id"] for o in observations])
        self.assertEqual(len({c["id"] for c in calls}), len(calls))
        return observations

    def test_write_rename_read_batch_is_ordered_and_replayable(self):
        content = "# preserve me\nprint('hello')\n"
        batch = "Creating.\n" + block("write_file", path="test.py", content=content)
        batch += "\nRenaming.\n" + block("shell", command="mv test.py test.text")
        batch += "\nChecking.\n" + block("read_file", path="test.text")
        result = self.run_script(raw(batch), raw(str(self.workspace / "test.text")))
        self.assertEqual(result.stop_reason, "final_response")
        self.assertEqual(len(result.model_requests), 2)
        self.assertFalse((self.workspace / "test.py").exists())
        self.assertEqual((self.workspace / "test.text").read_text(), content)
        observations = self.observations(result)
        self.assertEqual([o["ok"] for o in observations], [True] * 3)
        self.assertEqual(result.model_requests[0].call_ids, [o["call_id"] for o in observations])
        self.assertEqual(result.model_requests[0].raw_response, raw(batch))
        history = SessionStore(self.root / "state/sessions").load_history("batch")
        self.assertEqual(history, result.messages[1:])
        self.assertEqual(len(history[1]["tool_calls"]), 3)
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        from agent_from_scratch.llm import _QWEN_TEMPLATE
        formatter = Jinja2ChatFormatter(template=_QWEN_TEMPLATE.read_text(), eos_token="<|im_end|>", bos_token="<|endoftext|>")
        rendered = formatter(messages=result.messages, tools=list(self.agent.registry.schemas().values())).prompt
        self.assertEqual(rendered.count("<tool_response>"), 3)
        self.assertEqual(rendered.count('"name": "write_file"'), 2)  # Schema plus one executed call.

    def test_failure_skips_remainder_and_returns_all_results_for_replanning(self):
        result = self.run_script(
            raw(block("read_file", path="missing") + block("write_file", path="should-not-exist", content="bad")),
            raw(block("write_file", path="fixed", content="ok") + block("read_file", path="fixed")),
            raw("Recovered."),
        )
        observations = self.observations(result)
        self.assertEqual([o["error_code"] for o in observations], ["execution_error", "skipped", None, None])
        self.assertFalse((self.workspace / "should-not-exist").exists())
        self.assertEqual((self.workspace / "fixed").read_text(), "ok")
        self.assertEqual(result.stop_reason, "final_response")
        from agent_from_scratch.evals.verify import metrics
        counts = metrics(result)
        self.assertEqual((counts["tool_calls"], counts["tool_attempts"], counts["skipped_calls"]), (4, 3, 1))
        second = self.model.llm.create_chat_completion.call_args_list[1].kwargs["messages"]
        # Agent keeps one mutable transcript; request prefixes record the exact input boundary.
        prefix = second[:result.model_requests[1].input_message_count]
        self.assertEqual(json.loads(prefix[-1]["content"])["error_code"], "skipped")

    def test_repl_shows_narration_before_rename_and_final_path_after(self):
        original, renamed = self.workspace / "test.py", self.workspace / "test.text"
        self.model.llm.create_chat_completion.side_effect = [
            raw(block("write_file", path="test.py", content="keep")),
            raw(f"Created {original}.\n" + block("shell", command="mv test.py test.text")),
            raw(f"Renamed to {renamed}."),
        ]
        displayed = []
        with patch("builtins.input", side_effect=["Create then rename the file.", "/quit"]), \
             patch("builtins.print", side_effect=lambda text: displayed.append((text, original.exists(), renamed.exists()))):
            self.agent.run_repl()
        self.assertEqual(len(displayed), 2)
        self.assertIn(str(original), displayed[0][0])
        self.assertEqual(displayed[0][1:], (True, False))
        self.assertIn(str(renamed), displayed[1][0])
        self.assertEqual(displayed[1][1:], (False, True))

    def test_interruption_preserves_completed_effects_and_skips_remaining_calls(self):
        batch = block("write_file", path="before", content="keep") + block("interrupt")
        batch += block("write_file", path="after", content="never")
        result = self.run_script(raw(batch))
        self.assertEqual(result.stop_reason, "interrupted")
        observations = self.observations(result)
        self.assertEqual([o["error_code"] for o in observations], [None, "interrupted", "skipped"])
        self.assertEqual(observations[1]["output"], {"stdout": "started"})
        self.assertEqual((self.workspace / "before").read_text(), "keep")
        self.assertFalse((self.workspace / "after").exists())
        self.assertEqual(SessionStore(self.root / "state/sessions").load_history("batch"), [])

    def test_turn_budget_bounds_execution_within_a_batch(self):
        self.agent.max_tool_calls = 2
        result = self.run_script(raw("".join(block("write_file", path=str(i), content="x") for i in range(3))))
        self.assertEqual(result.stop_reason, "tool_limit")
        self.assertEqual([o["error_code"] for o in self.observations(result)], [None, None, "skipped"])
        self.assertEqual({p.name for p in self.workspace.iterdir()}, {"0", "1"})
        self.assertEqual(SessionStore(self.root / "state/sessions").load_history("batch"), [])

    def test_malformed_later_call_prevents_execution_of_whole_response(self):
        first = {"name": "write_file", "arguments": {"path": "bad", "content": "never"}}
        for response in (
            raw(block("write_file", **first["arguments"]) + '<tool_call>{broken}</tool_call>'),
            raw(None, tool_calls=[{"function": first}, {"function": {"name": "read_file", "arguments": "{broken}"}}]),
            raw(block("write_file", **first["arguments"]) * 9),
        ):
            with self.subTest(response=response):
                result = self.run_script(response, raw("No action executed."))
                self.assertEqual(result.model_requests[0].status, "parse_error")
                self.assertEqual(self.observations(result), [])
                self.assertEqual(list(self.workspace.iterdir()), [])

    def test_reused_native_ids_are_unique_across_model_requests(self):
        def response(path):
            return raw(None, tool_calls=[{"id": "provider-id", "function": {
                "name": "write_file", "arguments": json.dumps({"path": path, "content": "ok"})}}])
        result = self.run_script(response("one"), response("two"), raw("Done."))
        observations = self.observations(result)
        self.assertEqual(observations[0]["call_id"], "provider-id")
        self.assertNotEqual(observations[1]["call_id"], "provider-id")
        self.assertEqual(result.model_requests[1].raw_response, response("two"))

    def test_repeated_batch_failure_stops_without_executing_skipped_writes(self):
        batch = raw(block("read_file", path="missing") + block("write_file", path="bad", content="never"))
        result = self.run_script(batch, batch, batch)
        self.assertEqual(result.stop_reason, "no_progress")
        self.assertEqual([o["error_code"] for o in self.observations(result)], ["execution_error", "skipped"] * 3)
        self.assertEqual(list(self.workspace.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
