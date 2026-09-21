"""Default tool wiring: which tools this agent gets, and over which workspace."""

from pathlib import Path

from .base import ToolRegistry
from .calculator import CalculatorTool
from .files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from .search import GlobFilesTool, GrepTextTool
from .shell import ShellTool
from .web import WebFetchTool, WebSearchTool

workspace = Path(__file__).resolve().parents[1]
tools = [
    CalculatorTool(),
    ListFilesTool(workspace=workspace),
    GlobFilesTool(workspace=workspace),
    GrepTextTool(workspace=workspace),
    ReadFileTool(workspace=workspace),
    WriteFileTool(workspace=workspace),
    EditFileTool(workspace=workspace),
    ShellTool(workspace=workspace),
    WebFetchTool(),
    WebSearchTool(),
]
default_registry = ToolRegistry(tools)
default_tool_schemas = default_registry.schemas()
