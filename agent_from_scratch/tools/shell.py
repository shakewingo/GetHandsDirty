"""Synchronous local shell, with an optional fixed-command mode for evaluations.

General command/working_dir semantics follow nanobot's ExecTool. A cwd is not a sandbox.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
from time import monotonic
from typing import Mapping

from .base import Tool, ToolErrorCode, ToolExecutionError, ToolInterrupted


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    description: str
    timeout: float = 10.0

    def __post_init__(self):
        if (not isinstance(self.argv, tuple) or not self.argv
                or any(not isinstance(arg, str) or "\0" in arg for arg in self.argv)
                or not Path(self.argv[0]).is_absolute() or not Path(self.argv[0]).is_file()
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("Use fixed argv with an absolute executable and a positive finite timeout.")


class ShellTool(Tool):
    name = "shell"
    parameters = {
        "type": "object",
        "properties": {"command_id": {"type": "string"}},
        "required": ["command_id"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | Path = ".", commands: Mapping[str, Command] | None = None,
                 max_bytes: int = 4096, *, timeout: float = 30):
        self.workspace = Path(workspace).resolve(strict=True)
        self.commands = dict(commands) if commands is not None else None
        if (os.name != "posix" or not self.workspace.is_dir() or max_bytes < 1
                or not math.isfinite(timeout) or timeout <= 0
                or any(not key.strip() or not isinstance(value, Command)
                       for key, value in (self.commands or {}).items())):
            raise ValueError("Shell requires POSIX, a workspace, valid Commands, and positive limits.")
        self.max_bytes, self.timeout = max_bytes, timeout
        if self.commands is None:
            self.parameters = {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command, including pipes or redirects."},
                    "working_dir": {"type": "string", "description": "Optional absolute path or path relative to the default working directory."},
                },
                "required": ["command"], "additionalProperties": False,
            }
            self.description = (
                "Execute a local shell command using /bin/sh. Supports installed programs, "
                "scripts, pipes, redirects and environment assignments. "
                "Use mv to rename or move files, preserving their contents. "
                f"Default working directory: {self.workspace}. Timeout: {timeout}s. "
                "Runs with your local user permissions; working_dir is not a sandbox. "
                "Inspect exit_code, stdout, stderr and truncation before claiming success."
            )
            return
        catalog = "; ".join(f"{key}: {value.description}" for key, value in self.commands.items())
        self.description = (
            "Run one configured command in the workspace. Only command_id is accepted. "
            "Inspect exit_code, stdout, stderr and truncation; a failed check is feedback. "
            f"Available commands: {catalog or '(none)'}."
        )

    def execute(self, command_id: str | None = None, *, command: str | None = None,
                working_dir: str | None = None) -> dict:
        cwd = self.workspace
        if self.commands is None:
            if not command or not command.strip() or "\0" in command:
                raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "Provide a nonempty shell command without NUL bytes.")
            if working_dir is not None:
                requested = Path(working_dir).expanduser()
                cwd = (self.workspace / requested).resolve(strict=True)
                if not cwd.is_dir():
                    raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "working_dir must be a directory.")
            argv = ("/bin/sh", "-c", command)
            timeout = self.timeout
            env = os.environ.copy()
            # Ensure `python` resolves to the active interpreter even when launched by absolute path.
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", os.defpath)
        else:
            configured = self.commands.get(command_id or "")
            if configured is None:
                raise ToolExecutionError(ToolErrorCode.DENIED,
                                         f"Unknown command_id. Available: {', '.join(self.commands)}.")
            argv, timeout = configured.argv, configured.timeout
            env = {"PATH": os.defpath, "LANG": "C.UTF-8"}
        started = monotonic()
        deadline = started + timeout
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}
        timed_out = interrupted = False
        process = subprocess.Popen(
            argv, cwd=cwd, shell=False, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        try:
            with selectors.DefaultSelector() as selector:
                for name in buffers:
                    stream = getattr(process, name)
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                while selector.get_map() or process.poll() is None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    for key, _ in selector.select(min(remaining, 0.05)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        name = key.data
                        room = self.max_bytes - len(buffers[name])
                        buffers[name].extend(chunk[:room])
                        truncated[name] |= len(chunk) > room
        except KeyboardInterrupt:
            interrupted = True
        finally:
            # Also reap descendants holding pipes after the leader exits. Detached processes
            # are outside this foreground-process contract.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        output = {
            "argv": list(argv), "cwd": str(cwd),
            "exit_code": process.returncode,
            **{name: data.decode("utf-8", errors="replace") for name, data in buffers.items()},
            "truncated": truncated, "timed_out": timed_out, "interrupted": interrupted,
            "elapsed_seconds": round(monotonic() - started, 3),
        }
        output["command_id" if self.commands is not None else "command"] = command_id if self.commands is not None else command
        if interrupted:
            raise ToolInterrupted(output)
        if timed_out:
            raise ToolExecutionError(ToolErrorCode.TIMEOUT, f"Limit: {timeout}s.", output)
        if process.returncode != 0:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"Command exited with code {process.returncode}; inspect stderr/stdout.", output)
        return output
