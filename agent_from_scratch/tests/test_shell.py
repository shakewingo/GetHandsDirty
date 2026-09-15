import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import ToolCall, LLM, LLMResponse, ResponseType
from agent_from_scratch.session import SessionStore
from agent_from_scratch.tools.base import ToolInterrupted
from agent_from_scratch.tools.register import ToolRegistry
from agent_from_scratch.tools.shell import Command, ShellTool
from agent_from_scratch.trace import TraceStore


class ShellTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tool(self, code, *, timeout=2, max_bytes=128):
        # Test code is a trusted fixed command, never supplied as a tool argument.
        return ShellTool(self.workspace, {
            "check": Command((sys.executable, "-I", "-c", code), "Check the fixture", timeout),
        }, max_bytes=max_bytes)

    def test_fixed_argv_cwd_and_nonzero_output(self):
        for status in (0, 7):
            tool = self.tool(f'import os, sys; print(os.getcwd()); print("diagnostic", file=sys.stderr); sys.exit({status})')
            result = tool.invoke({"command_id": "check"}, call_id="shell-1")
            self.assertEqual(result.ok, status == 0)
            self.assertEqual(result.call_id, "shell-1")
            self.assertEqual(Path(result.output["stdout"].strip()), self.workspace.resolve())
            self.assertEqual(result.output["stderr"], "diagnostic\n")
            self.assertEqual(result.output["exit_code"], status)
            self.assertFalse(result.output["timed_out"])
            if status:
                self.assertEqual(result.error_code, "execution_error")

    def test_unknown_commands_and_extra_arguments_never_spawn(self):
        tool = self.tool('open("created", "w").write("x")')
        with patch("agent_from_scratch.tools.shell.subprocess.Popen") as spawn:
            for command_id in ("missing", "check; touch created", "../check"):
                self.assertEqual(tool.invoke({"command_id": command_id}).error_code, "denied")
            for arguments in ({"command": "touch created"}, {"command_id": "check", "cwd": "/"},
                              {"command_id": "check", "timeout": 100}, {"command_id": 5}):
                self.assertEqual(tool.invoke(arguments).error_code, "invalid_arguments")
            spawn.assert_not_called()
        self.assertFalse((self.workspace / "created").exists())

    def test_both_pipes_are_drained_with_bounded_memory_and_visible_truncation(self):
        tool = self.tool('import os;\nfor _ in range(64):\n os.write(1, b"x"*8192); os.write(2, b"y"*8192)')
        result = tool.invoke({"command_id": "check"})
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.output["stdout"], "x" * 128)
        self.assertEqual(result.output["stderr"], "y" * 128)
        self.assertEqual(result.output["truncated"], {"stdout": True, "stderr": True})

    def test_timeout_preserves_partial_output_and_next_call_can_run(self):
        tool = self.tool('import time; print("started", flush=True); time.sleep(10)', timeout=0.15)
        started = monotonic()
        result = tool.invoke({"command_id": "check"})
        self.assertEqual(result.error_code, "timeout")
        self.assertLess(monotonic() - started, 1)
        self.assertIn("started", result.output["stdout"])
        self.assertTrue(result.output["timed_out"])
        self.assertLess(result.output["exit_code"], 0)
        self.assertTrue(self.tool('print("recovered")').invoke({"command_id": "check"}).ok)

    def test_timeout_cleans_descendants_even_when_leader_has_exited(self):
        # The child inherits the output pipes and would create a file after the timeout.
        child_code = 'import time; time.sleep(.5); open("escaped", "w").write("bad")'
        tool = self.tool(f'import subprocess,sys; subprocess.Popen([sys.executable,"-I","-c",{child_code!r}]); print("spawned", flush=True)', timeout=0.12)
        result = tool.invoke({"command_id": "check"})
        self.assertEqual(result.error_code, "timeout")
        sleep(0.55)
        self.assertFalse((self.workspace / "escaped").exists())

    def test_ctrl_c_preserves_output_closes_process_and_repl_does_not_replay(self):
        tool = self.tool('import os,time; print(os.getpid(), flush=True); time.sleep(10)')
        model = Mock(spec=LLM)
        model.settings.return_value = {}
        seen = []
        responses = iter([
            LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('shell', {'command_id': 'check'})]),
            LLMResponse("assistant", "REPL recovered", ResponseType.direct),
        ])

        def generate(messages, tools):
            seen.append(list(messages))
            return next(responses)

        model.generate.side_effect = generate
        agent = Agent(model, str(self.root / "state"), registry=ToolRegistry([tool]))
        select = selectors.KqueueSelector.select if sys.platform == "darwin" else selectors.EpollSelector.select
        selections = 0

        def interrupt_after_read(selector, timeout=None):
            nonlocal selections
            selections += 1
            if selections == 3:
                raise KeyboardInterrupt()
            return select(selector, timeout)

        with patch.object(selectors.DefaultSelector, "select", interrupt_after_read), \
                patch("builtins.input", side_effect=["run check", "hello", "/quit"]), patch("builtins.print"):
            agent.run_repl()
        records = SessionStore(self.root / "state/sessions").load_records("default_session")
        self.assertEqual([r["stop_reason"] for r in records], ["interrupted", "final_response"])
        self.assertEqual(records[0]["messages"], [])
        trace = TraceStore(self.root / "state/runs").load_run(records[0]["run_id"])
        observation = json.loads(trace["messages"][-1]["content"])
        self.assertEqual(observation["error_code"], "interrupted")
        self.assertTrue(observation["output"]["interrupted"])
        pid = int(observation["output"]["stdout"].strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertEqual(trace["messages"][-2]["tool_calls"][0]["id"], observation["call_id"])
        self.assertEqual(seen[-1][-1]["content"], "hello")
        self.assertFalse(any(m.get("tool_calls") for m in seen[-1]))

    def test_real_sigint_cleans_process_group(self):
        code = 'import os,signal,time; os.kill(os.getppid(), signal.SIGINT); time.sleep(10)'
        tool = self.tool(code)
        with self.assertRaises(ToolInterrupted) as caught:
            tool.invoke({"command_id": "check"})
        self.assertTrue(caught.exception.output["interrupted"])
        self.assertLess(caught.exception.output["exit_code"], 0)

    def test_general_command_supports_pipes_redirects_and_environment(self):
        result = ShellTool(self.workspace).invoke({
            "command": "TINY_WORD=hello; printf '%s' \"$TINY_WORD\" | tr a-z A-Z > result.txt; cat result.txt",
        })
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.output["stdout"], "HELLO")
        self.assertEqual((self.workspace / "result.txt").read_text(), "HELLO")
        self.assertIn("command", result.output)

    def test_general_command_can_run_a_local_script_and_choose_cwd(self):
        directory = self.workspace / "subdirectory"
        directory.mkdir()
        (directory / "script.py").write_text('print("script executed")')
        tool = ShellTool(self.workspace)
        for cwd in ("subdirectory", str(directory.resolve())):
            result = tool.invoke({"command": "python script.py", "working_dir": cwd})
            self.assertTrue(result.ok, result.error_message)
            self.assertEqual(result.output["stdout"], "script executed\n")
            self.assertEqual(Path(result.output["cwd"]), directory.resolve())

    def test_general_nonzero_and_timeout_preserve_observations(self):
        result = ShellTool(self.workspace).invoke({"command": "printf diagnostic >&2; exit 7"})
        self.assertEqual(result.error_code, "execution_error")
        self.assertEqual(result.output["exit_code"], 7)
        self.assertEqual(result.output["stderr"], "diagnostic")
        code = 'import time; print("started", flush=True); time.sleep(10)'
        result = ShellTool(self.workspace, timeout=0.15).invoke({
            "command": shlex.join((sys.executable, "-I", "-c", code)),
        })
        self.assertEqual(result.error_code, "timeout")
        self.assertIn("started", result.output["stdout"])
        self.assertTrue(result.output["timed_out"])

    def test_general_timeout_cleans_the_shells_descendant(self):
        code = 'import time; time.sleep(.4); open("late-write", "w").write("bad")'
        result = ShellTool(self.workspace, timeout=0.1).invoke({
            "command": shlex.join((sys.executable, "-I", "-c", code)) + " & wait",
        })
        self.assertEqual(result.error_code, "timeout")
        sleep(0.45)
        self.assertFalse((self.workspace / "late-write").exists())

    def test_general_output_caps_still_apply(self):
        code = 'import os; os.write(1, b"a" * 100000); os.write(2, b"b" * 100000)'
        result = ShellTool(self.workspace, max_bytes=64).invoke({
            "command": shlex.join((sys.executable, "-I", "-c", code)),
        })
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.output["stdout"], "a" * 64)
        self.assertEqual(result.output["stderr"], "b" * 64)
        self.assertEqual(result.output["truncated"], {"stdout": True, "stderr": True})

    def test_general_invalid_arguments_never_spawn(self):
        with patch("agent_from_scratch.tools.shell.subprocess.Popen") as spawn:
            for args in ({"command": ""}, {"command": "\0"}, {"command_id": "pwd"},
                         {"command": "pwd", "timeout": 900}):
                self.assertEqual(ShellTool(self.workspace).invoke(args).error_code, "invalid_arguments")
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
