"""The plan is re-injected every request and never enters raw history."""

import unittest

from agent_from_scratch.context import ContextState
from agent_from_scratch.tools.plan import PlanTool


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


if __name__ == "__main__":
    unittest.main()
