import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.agent import Agent
from agent_from_scratch.examples.tools_demo import demo_registry
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType
from agent_from_scratch.tools.base import ToolCall
from agent_from_scratch.trace import TraceStore


def call(name, **arguments):
    return LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall(name, arguments)])


class ToolWorkflowTests(unittest.TestCase):
    def test_failed_check_read_change_check_and_trace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "config.json").write_text('{"output":"old.txt","retries":3}')
            model = Mock(spec=LLM)
            model.measure_context.return_value = {  # Scripted fitting budget; no real tokenizer.
                "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
                "response_reserve": 512, "remaining_tokens": 7388,
            }
            model.settings.return_value = {}
            state = 0

            def generate(messages, tools, **kwargs):
                nonlocal state
                state += 1
                if state == 1:
                    return call("shell", command_id="check_fixture")
                observation = json.loads(messages[-1]["content"])
                if state == 2:
                    self.assertFalse(observation["ok"])
                    self.assertIn("Expected output", observation["output"]["stderr"])
                    return call("read_file", path="config.json")
                if state == 3:
                    data = json.loads(observation["output"]["content"])
                    data["output"] = "report.txt"
                    return call("write_file", path="config.json", content=json.dumps(data))
                if state == 4:
                    self.assertTrue(observation["ok"])
                    return call("shell", command_id="check_fixture")
                self.assertTrue(observation["ok"])
                self.assertEqual(observation["output"]["exit_code"], 0)
                return LLMResponse("assistant", "Configuration check passed.", ResponseType.direct)

            model.generate.side_effect = generate
            agent = Agent(model, str(root / "state"), registry=demo_registry(workspace, set()))
            result = agent.run_turn("Fix the configuration and check it", session_id="workflow")
            self.assertEqual(result.stop_reason, "final_response")
            self.assertEqual(json.loads((workspace / "config.json").read_text()),
                             {"output": "report.txt", "retries": 3})
            saved = TraceStore(root / "state/runs").load_run(result.run_id)
            assert saved is not None
            requests = [m["tool_calls"][0]["id"] for m in saved["messages"] if m.get("tool_calls")]
            observations = [m for m in saved["messages"] if m["role"] == "tool"]
            self.assertEqual(requests, [m["tool_call_id"] for m in observations])
            self.assertEqual(len(set(requests)), 4)
            self.assertEqual([json.loads(m.get("content") or "")["ok"] for m in observations], [False, True, True, True])

    def test_demo_refuses_to_put_trusted_command_inside_writable_workspace(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            demo_registry(Path(__file__).resolve().parents[1], set())


if __name__ == "__main__":
    unittest.main()
