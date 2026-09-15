"""Synchronous fixed commands for a local POSIX workspace, not a shell sandbox."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
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

    def __init__(self, workspace: str | Path, commands: Mapping[str, Command], max_bytes: int = 4096):
        self.workspace = Path(workspace).resolve(strict=True)
        self.commands = dict(commands)
        if (os.name != "posix" or not self.workspace.is_dir() or max_bytes < 1
                or any(not key.strip() or not isinstance(value, Command)
                       for key, value in self.commands.items())):
            raise ValueError("Shell requires POSIX, a workspace, named Commands, and a positive output cap.")
        self.max_bytes = max_bytes
        catalog = "; ".join(f"{key}: {value.description}" for key, value in self.commands.items())
        self.description = (
            "Run one configured command in the workspace. Only command_id is accepted. "
            "Inspect exit_code, stdout, stderr and truncation; a failed check is feedback. "
            f"Available commands: {catalog or '(none)'}."
        )

    def execute(self, command_id: str) -> dict:
        command = self.commands.get(command_id)
        if command is None:
            raise ToolExecutionError(ToolErrorCode.DENIED,
                                     f"Unknown command_id. Available: {', '.join(self.commands)}.")
        started = monotonic()
        deadline = started + command.timeout
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}
        timed_out = interrupted = False
        process = subprocess.Popen(
            command.argv, cwd=self.workspace, shell=False, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # Trusted commands opt into capabilities through their fixed argv, not ambient Python/shell setup.
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
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
            # are outside this trusted-command contract; do not configure daemonizing scripts.
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
            "command_id": command_id, "argv": list(command.argv), "cwd": str(self.workspace),
            "exit_code": process.returncode,
            **{name: data.decode("utf-8", errors="replace") for name, data in buffers.items()},
            "truncated": truncated, "timed_out": timed_out, "interrupted": interrupted,
            "elapsed_seconds": round(monotonic() - started, 3),
        }
        if interrupted:
            raise ToolInterrupted(output)
        if timed_out:
            raise ToolExecutionError(ToolErrorCode.TIMEOUT, f"Limit: {command.timeout}s.", output)
        if process.returncode != 0:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"Command exited with code {process.returncode}; inspect stderr/stdout.", output)
        return output
