"""Stage 2A REPL with explicit workspace, HTTPS hosts, and trusted fixed commands."""

import argparse
from pathlib import Path
import sys

from ..agent import Agent
from ..llm import LLM, _QWEN_TEMPLATE
from ..tools.calculator import CalculatorTool
from ..tools.files import ListFilesTool, ReadFileTool, WriteFileTool
from ..tools.register import ToolRegistry
from ..tools.shell import Command, ShellTool
from ..tools.web import WebFetchTool


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
    return ToolRegistry([CalculatorTool(), ListFilesTool(workspace), ReadFileTool(workspace),
                         WriteFileTool(workspace), WebFetchTool(hosts), ShellTool(workspace, commands)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Existing disposable directory")
    parser.add_argument("--state", type=Path, default=Path("outputs/tools-demo-state"))
    parser.add_argument("--allow-host", action="append", default=[], help="Exact HTTPS host; repeat to add more")
    args = parser.parse_args()
    if args.state.resolve().is_relative_to(args.workspace.resolve()):
        parser.error("State must be outside the writable workspace.")
    registry = demo_registry(args.workspace, set(args.allow_host))
    model = LLM(temperature=0, max_tokens=2048, n_ctx=8000, chat_template_path=_QWEN_TEMPLATE)
    Agent(model, str(args.state), registry=registry).run_repl()


if __name__ == "__main__":
    main()
