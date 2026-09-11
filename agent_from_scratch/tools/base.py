from __future__ import annotations # postpone evaluating type annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionTool


class ToolErrorCode(StrEnum):
    INVALID_TOOL_CALL = "invalid_tool_call"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_ERROR = "execution_error"


ERROR_MESSAGES = {
    ToolErrorCode.INVALID_TOOL_CALL: "Tool name must be a non-empty string.",
    ToolErrorCode.UNKNOWN_TOOL: "The requested tool is not registered.",
    ToolErrorCode.INVALID_ARGUMENTS: "Tool arguments do not match the schema.",
    ToolErrorCode.EXECUTION_ERROR: "Tool execution failed.",
}


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    ok: bool
    output: Any = None
    error_code: ToolErrorCode | None = None
    error_message: str | None = None

    @classmethod
    def failure(
        cls, code: ToolErrorCode, *, call_id: str = "",
        tool_name: str = "", detail: str = "",
    ) -> "ToolResult":
        message = ERROR_MESSAGES[code]
        if detail:
            message = f"{message} {detail}"
        return cls(call_id=call_id, tool_name=tool_name, ok=False,
                   error_code=code, error_message=message)


class Tool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Return the tool's output; invoke() owns result wrapping and errors."""
        pass

    def validate_arguments(self, arguments: Mapping[str, Any]) -> None:
        """Validate model-provided arguments against this tool's JSON schema.

        This intentionally supports the JSON Schema features used by the local
        tools: ``type``, ``properties``, ``required``, ``additionalProperties``,
        ``enum`` and array ``items``. Invalid arguments raise ``ValueError`` and
        must never reach ``execute``.
        """
        if not isinstance(arguments, Mapping):
            raise ValueError(
                f"arguments must be an object, got {type(arguments).__name__}"
            )

        self._validate_value(arguments, self.parameters, path="arguments")

    def invoke(self, arguments: Mapping[str, Any], call_id: str = "") -> ToolResult:
        """Validate and execute a tool call, returning failures as data."""
        try:
            self.validate_arguments(arguments)
        except ValueError as error:
            return ToolResult.failure(
                ToolErrorCode.INVALID_ARGUMENTS,
                call_id=call_id,
                tool_name=self.name,
                detail=str(error),
            )

        try:
            output = self.execute(**dict(arguments))
        except Exception as error:
            return ToolResult.failure(
                ToolErrorCode.EXECUTION_ERROR,
                call_id=call_id,
                tool_name=self.name,
                detail=f"{type(error).__name__}: {error}",
            )

        return ToolResult(call_id=call_id, tool_name=self.name, ok=True, output=output)

    @classmethod
    def _validate_value(
        cls,
        value: Any,
        schema: Mapping[str, Any],
        path: str,
    ) -> None:
        if not isinstance(schema, Mapping):
            raise TypeError(f"Invalid tool schema at {path}: expected an object")

        expected_type = schema.get("type")
        if expected_type is not None and not cls._matches_type(value, expected_type):
            raise ValueError(
                f"{path} must be {expected_type}, got {type(value).__name__}"
            )

        if "enum" in schema and not any(cls._json_equal(value, option) for option in schema["enum"]):
            raise ValueError(f"{path} must be one of {schema['enum']}, got {value!r}")

        if isinstance(value, Mapping):
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            if not isinstance(properties, Mapping):
                raise TypeError(
                    f"Invalid tool schema at {path}.properties: expected an object"
                )
            if not isinstance(required, list) or not all(
                isinstance(name, str) for name in required
            ):
                raise TypeError(
                    f"Invalid tool schema at {path}.required: expected a list of strings"
                )

            missing = [name for name in required if name not in value]
            if missing:
                raise ValueError(f"{path} is missing required fields: {missing}")

            additional = schema.get("additionalProperties", True)
            unknown = [name for name in value if name not in properties]
            if additional is False and unknown:
                raise ValueError(f"{path} contains unexpected fields: {unknown}")

            for name, item in value.items():
                if name in properties:
                    cls._validate_value(
                        item,
                        properties[name],
                        path=f"{path}.{name}",
                    )
                elif isinstance(additional, Mapping):
                    cls._validate_value(
                        item,
                        additional,
                        path=f"{path}.{name}",
                    )

        if isinstance(value, list):
            items = schema.get("items")
            if items is not None:
                for index, item in enumerate(value):
                    cls._validate_value(item, items, path=f"{path}[{index}]")

    @classmethod
    def _json_equal(cls, left: Any, right: Any) -> bool:
        """JSON booleans are distinct from numbers, including inside containers."""
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return left.keys() == right.keys() and all(
                cls._json_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                cls._json_equal(a, b) for a, b in zip(left, right)
            )
        return left == right

    @staticmethod
    def _matches_type(value: Any, expected_type: Any) -> bool:
        if isinstance(expected_type, list):
            return any(Tool._matches_type(value, item) for item in expected_type)

        checks = {
            "object": lambda item: isinstance(item, Mapping),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "number": lambda item: isinstance(item, (int, float))
            and not isinstance(item, bool),
            "integer": lambda item: isinstance(item, int)
            and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
        }

        check = checks.get(expected_type)
        if check is None:
            raise TypeError(f"Unsupported JSON Schema type: {expected_type!r}")
        return check(value)

    def to_schema(self) -> ChatCompletionTool:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
