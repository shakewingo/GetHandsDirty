"""FaultyTool fails exactly its configured call and delegates every other one."""

import unittest

from agent_from_scratch.evals.bench.faults import FaultyTool
from agent_from_scratch.tools.calculator import CalculatorTool


class FaultyToolTests(unittest.TestCase):
    def test_fails_only_the_configured_call(self):
        tool = FaultyTool(CalculatorTool(), on_call=2, message="Simulated failure.")
        first = tool.invoke({"operation": "add", "left": 1, "right": 1})
        second = tool.invoke({"operation": "add", "left": 1, "right": 1})
        third = tool.invoke({"operation": "add", "left": 1, "right": 1})
        self.assertEqual((first.ok, second.ok, third.ok), (True, False, True))
        self.assertIn("Simulated failure.", second.error_message)
        self.assertEqual(second.error_code, "execution_error")
        self.assertEqual((first.output, third.output), (2.0, 2.0))

    def test_copies_the_wrapped_tools_schema(self):
        wrapped = CalculatorTool()
        tool = FaultyTool(wrapped, on_call=1, message="x")
        self.assertEqual((tool.name, tool.description, tool.parameters),
                         (wrapped.name, wrapped.description, wrapped.parameters))

    def test_rejects_a_non_positive_call_number(self):
        with self.assertRaises(ValueError):
            FaultyTool(CalculatorTool(), on_call=0, message="x")


if __name__ == "__main__":
    unittest.main()
