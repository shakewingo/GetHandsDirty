"""The plan is re-injected every request and never enters raw history."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.config import AgentLimits
from agent_from_scratch.context import ContextState
from agent_from_scratch.llm import LLM
from agent_from_scratch.tests.test_turn import answer
from agent_from_scratch.tools.plan import PlanTool


def plan_call(status="in_progress"):
    return LLM.parse_response({"choices": [{"message": {"role": "assistant", "content":
        '<tool_call>{"name": "update_plan", "arguments": {"plan": '
        f'[{{"content": "Add numbers", "status": "{status}"}}]}}}}</tool_call>'}}]})


class PlanTests(unittest.TestCase):
    def test_tool_replaces_the_whole_plan_and_renders_it(self):
        tool = PlanTool()
        result = tool.invoke({"plan": [{"content": "Read config", "status": "in_progress"},
                                       {"content": "Report", "status": "pending"}]})
        self.assertTrue(result.ok)
        self.assertEqual(tool.render(), "[in_progress] Read config\n[pending] Report")
        self.assertFalse(tool.invoke({"plan": [{"content": "x", "status": "done"}]}).ok)

    def test_view_carries_one_trailing_reminder_and_raw_is_unchanged(self):
        raw = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
        state = ContextState(list(raw), turn_start=1, last_sent=1)
        self.assertEqual(state.messages(), raw)
        state.plan = ""
        self.assertIn("not created a plan", state.messages()[-1]["content"])
        state.plan = "[pending] Report"
        self.assertEqual(state.messages()[-1]["role"], "system")
        self.assertIn("[pending] Report", state.messages()[-1]["content"])
        self.assertEqual(len(state.messages()), 3)
        self.assertEqual(state.raw, raw)


class PlanTurnTests(unittest.TestCase):
    """Own copy of the TurnTests fixture; subclassing would rerun every inherited test here."""

    def setUp(self):
        self.model = Mock(spec=LLM)
        self.model.measure_context.return_value = {
            "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
            "response_reserve": 512, "remaining_tokens": 7388,
        }
        self.model.settings.return_value = {}
        self.model.read_usage.side_effect = LLM.read_usage
        self.agent = Agent(self.model)
        self.seen = []
        prompt_dir = Path(self.enterContext(TemporaryDirectory()))
        (prompt_dir / "system.md").write_text("System", encoding="utf-8")
        self.enterContext(patch("agent_from_scratch.context.PROMPTS_DIR", prompt_dir))

    def script(self, *responses):
        responses = iter(responses)

        def generate(messages, tools, **kwargs):
            self.seen.append(deepcopy(messages))
            return next(responses)

        self.model.generate.side_effect = generate

    def test_planning_turn_shows_the_plan_and_never_dispatches_it_to_the_registry(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(plan_call(), answer("4"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("2+2")
        execute.assert_not_called()
        self.assertIn("update_plan", result.model_requests[0].tools or {})
        self.assertIn("not created a plan", self.seen[0][-1]["content"])
        self.assertIn("[in_progress] Add numbers", self.seen[1][-1]["content"])
        self.assertEqual(sum(m["role"] == "system" for m in self.seen[1]), 2)
        self.assertFalse(any(m["role"] == "system" for m in result.messages[1:]))
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "4"))

    def test_planning_off_adds_no_tool_or_reminder(self):
        self.script(answer("4"))
        result = self.agent.run_turn("2+2")
        self.assertNotIn("update_plan", result.model_requests[0].tools or {})
        self.assertEqual(self.seen[0][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
