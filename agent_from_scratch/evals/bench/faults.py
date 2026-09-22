"""Eval-only tool wrapper that fails a wrapped tool's k-th call; never used by the runtime."""

from ...tools.base import Tool, ToolErrorCode, ToolExecutionError


class FaultyTool(Tool):
    """Delegate to `wrapped`, except call number `on_call` (1-based, across all its calls)."""

    def __init__(self, wrapped: Tool, on_call: int, message: str):
        if on_call < 1:
            raise ValueError("on_call must be a 1-based call number.")
        self.wrapped, self.on_call, self.message = wrapped, on_call, message
        self.name, self.description, self.parameters = wrapped.name, wrapped.description, wrapped.parameters
        self.calls = 0

    def execute(self, **kwargs):
        self.calls += 1
        if self.calls == self.on_call:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, self.message)
        return self.wrapped.execute(**kwargs)
