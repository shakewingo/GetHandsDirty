import json
import unittest
from dataclasses import asdict

from agent_from_scratch.tools.base import ERROR_MESSAGES, ToolErrorCode, Tool
from agent_from_scratch.tools.calculator import CalculatorTool, OPERATION
from agent_from_scratch.tools.register import ToolRegistry


class RecordingCalculator(CalculatorTool):
    calls = 0

    def execute(self, operation: OPERATION, left: float, right: float) -> float:
        self.calls += 1
        return super().execute(operation, left, right)


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tool = RecordingCalculator()
        self.registry = ToolRegistry([self.tool])

    def test_success_preserves_identity_and_output(self):
        result = self.registry.invoke(
            "calculator", {"operation": "multiply", "left": 6, "right": 7}, "c1"
        )
        self.assertTrue(result.ok)
        self.assertEqual((result.call_id, result.tool_name, result.output),
                         ("c1", "calculator", 42))
        self.assertIsNone(result.error_code)

    def test_invalid_calls_never_execute(self):
        valid = {"operation": "add", "left": 1, "right": 2}
        cases = [
            (None, valid, ToolErrorCode.INVALID_TOOL_CALL),
            ([], valid, ToolErrorCode.INVALID_TOOL_CALL),
            (" ", valid, ToolErrorCode.INVALID_TOOL_CALL),
            ("missing", valid, ToolErrorCode.UNKNOWN_TOOL),
            ("calculator", None, ToolErrorCode.INVALID_ARGUMENTS),
            ("calculator", {}, ToolErrorCode.INVALID_ARGUMENTS),
            ("calculator", {**valid, "left": True}, ToolErrorCode.INVALID_ARGUMENTS),
            ("calculator", {**valid, "left": "1"}, ToolErrorCode.INVALID_ARGUMENTS),
            ("calculator", {**valid, "operation": "power"}, ToolErrorCode.INVALID_ARGUMENTS),
            ("calculator", {**valid, "extra": 0}, ToolErrorCode.INVALID_ARGUMENTS),
        ]
        for name, arguments, code in cases:
            with self.subTest(name=name, arguments=arguments):
                result = self.registry.invoke(name, arguments, "c2")
                self.assertFalse(result.ok)
                self.assertEqual(result.error_code, code)
                self.assertEqual(result.call_id, "c2")
                assert result.error_message is not None
                self.assertTrue(result.error_message.startswith(ERROR_MESSAGES[code]))
        self.assertEqual(self.tool.calls, 0)

    def test_execution_failure_is_serializable_and_next_call_works(self):
        result = self.registry.invoke(
            "calculator", {"operation": "divide", "left": 1, "right": 0}, "c3"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, ToolErrorCode.EXECUTION_ERROR)
        assert result.error_message is not None
        self.assertIn("ZeroDivisionError", result.error_message)
        self.assertEqual(json.loads(json.dumps(asdict(result)))["error_code"],
                         "execution_error")
        self.assertTrue(self.registry.invoke(
            "calculator", {"operation": "add", "left": 1, "right": 2}
        ).ok)

    def test_duplicate_registration_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate tool name"):
            ToolRegistry([self.tool, CalculatorTool()])

    def test_union_and_implicit_container_constraints(self):
        for schema, value in [
            ({"type": ["object", "null"], "required": ["x"]}, {}),
            ({"required": ["x"]}, {}),
            ({"type": ["array", "null"], "items": {"type": "number"}}, ["bad"]),
        ]:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                Tool._validate_value(value, schema, "arguments")
        Tool._validate_value(None, {"type": ["object", "null"], "required": ["x"]}, "arguments")

    def test_enum_separates_booleans_and_numbers(self):
        for value, option in [(True, 1), ([True], [1]), ({"x": False}, {"x": 0})]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Tool._validate_value(value, {"enum": [option]}, "arguments")
        Tool._validate_value(1.0, {"enum": [1]}, "arguments")


if __name__ == "__main__":
    unittest.main()
