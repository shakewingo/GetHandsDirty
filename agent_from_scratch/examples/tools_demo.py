"""REPL with an explicit workspace; choose fixture tools or general shell/search/fetch."""

import argparse
from pathlib import Path
import sys

from ..agent import Agent
from ..llm import LLM
from ..tools.calculator import CalculatorTool
from ..tools.files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from ..evals import legacy_files
from ..tools.base import ToolRegistry
from ..tools.shell import Command, ShellTool
from ..tools.web import WebFetchTool, WebSearchTool


def demo_registry(workspace: Path, hosts: set[str]) -> ToolRegistry:
    workspace = workspace.resolve(strict=True)
    script = Path(__file__).with_name("fixture_command.py").resolve()
    if script.is_relative_to(workspace):
        raise ValueError("The trusted fixture script must be outside the writable workspace.")
    commands = {
        "inspect_fixture": Command((sys.executable, "-I", str(script), "inspect"), "Print config.json"),
        "check_fixture": Command((sys.executable, "-I", str(script), "check"),
                                 'Check output="report.txt" and retries=3 in config.json'),
    }
    return ToolRegistry([CalculatorTool(), legacy_files.ListFilesTool(workspace), legacy_files.ReadFileTool(workspace),
                         legacy_files.WriteFileTool(workspace), WebFetchTool(hosts), ShellTool(workspace, commands)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Existing disposable directory")
    parser.add_argument("--state", type=Path, default=Path("outputs/tools-demo-state"))
    parser.add_argument("--allow-host", action="append", default=[], help="Exact HTTPS host; repeat to add more")
    parser.add_argument("--general-tools", action="store_true", help="General shell and web search/fetch instead of fixture tools")
    args = parser.parse_args()
    if args.state.resolve().is_relative_to(args.workspace.resolve()):
        parser.error("State must be outside the writable workspace.")
    if args.general_tools:
        registry = ToolRegistry([
            CalculatorTool(), ListFilesTool(args.workspace), ReadFileTool(args.workspace),
            WriteFileTool(args.workspace), EditFileTool(args.workspace), ShellTool(args.workspace),
            WebFetchTool(set(args.allow_host) if args.allow_host else None), WebSearchTool(),
        ])
    else:
        registry = demo_registry(args.workspace, set(args.allow_host))
    model = LLM()
    try:
        Agent(model, str(args.state), registry=registry).run_repl()
    finally:
        model.close()


if __name__ == "__main__":
    main()
