from collections.abc import Iterable
from .calculator import CalculatorTool
from .base import Tool, ToolErrorCode, ToolResult
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionTool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool.name, str) or not tool.name.strip():
                raise ValueError("Registered tools must have non-empty string names.")
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool

    def schemas(self) -> Dict[str, "ChatCompletionTool"]:
        return {name: tool.to_schema() for name, tool in self._tools.items()}

    def invoke(self, tool_name: Any, arguments: Any, call_id: str = "") -> ToolResult:
        if not isinstance(tool_name, str) or not tool_name.strip():
            return ToolResult.failure(
                ToolErrorCode.INVALID_TOOL_CALL, call_id=call_id,
            )
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.failure(
                ToolErrorCode.UNKNOWN_TOOL, call_id=call_id, tool_name=tool_name,
                detail=f"Requested: {tool_name!r}.",
            )
        return tool.invoke(arguments, call_id=call_id)



tools = [CalculatorTool()]
registry = ToolRegistry(tools)
tool_schemas = registry.schemas()
