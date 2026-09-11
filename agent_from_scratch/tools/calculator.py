from typing import Literal
from .base import Tool

OPERATION = Literal["add", "subtract", "multiply", "divide"]

class CalculatorTool(Tool):
    name = "calculator"
    description = "A calculator that can evaluate mathematical expressions."
    parameters = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
            "description": "The mathematical operation to perform.",
        },
        "left": {
            "type": "number",
            "description": "The left operand.",
        },
        "right": {
            "type": "number",
            "description": "The right operand.",
        }
    },
    "required": ["operation", "left", "right"],
    "additionalProperties": False
    }

    def execute(self, operation: OPERATION, left: float, right: float) -> float:
        if operation == "add":
            return left + right
        if operation == "subtract":
            return left - right
        if operation == "multiply":
            return left * right
        if operation == "divide":
            return left / right
        raise ValueError(f"Unsupported operation: {operation!r}")
