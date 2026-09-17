"""Load turn-scoped instructions and prepare independent model inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import stat
from typing import Any, TYPE_CHECKING

from .config import PROMPTS_DIR

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


class InstructionLoadError(ValueError):
    """Instruction setup failed before model generation or tool execution."""


@dataclass(frozen=True)
class InstructionConfig:
    workspace: Path | None = None
    user_path: Path | None = None
    max_source_bytes: int = 8192
    max_total_bytes: int = 16384

    def __post_init__(self):
        for name in ("max_source_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise InstructionLoadError(f"{name} must be a positive integer.")
        for name in ("workspace", "user_path"):
            path = getattr(self, name)
            if path is not None:
                try:
                    object.__setattr__(self, name, Path(path).expanduser().resolve())
                except (OSError, RuntimeError) as error:
                    raise InstructionLoadError(f"Cannot resolve {name} {path}: {error}") from error
        if self.workspace is not None and not self.workspace.is_dir():
            raise InstructionLoadError(f"Workspace must be an existing directory: {self.workspace}")


def _read_instruction(path: Path, limit: int) -> str:
    """Reject oversized/non-text sources instead of silently dropping rules."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise InstructionLoadError(f"Instruction source must be a regular file: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise InstructionLoadError(f"Instruction source exceeds {limit} bytes: {path}")
    return data.decode("utf-8")


def load_instructions(config: InstructionConfig) -> tuple[str, dict[str, Any]]:
    """Load once per turn; return one system message's text and its provenance.

    Files are literal Markdown. No imports, templates, ancestor search or truncation.
    Only absent root AGENTS.md is optional; a broken symlink is a loading error.
    """
    root = config.workspace
    if root is not None and not root.is_dir():
        raise InstructionLoadError(f"Workspace must be an existing directory: {root}")
    paths = [("system", PROMPTS_DIR / "system.md"), ("user", config.user_path),
             ("workspace", root / "AGENTS.md" if root is not None else None)]
    metadata: dict[str, Any] = {
        "version": 1, "order": [kind for kind, _ in paths],
        "workspace": str(root) if root is not None else None,
        "user_path": str(config.user_path) if config.user_path is not None else None,
        "max_source_bytes": config.max_source_bytes, "max_total_bytes": config.max_total_bytes,
        "sources": [],
    }
    blocks = []
    for kind, path in paths:
        source: dict[str, Any] = {"kind": kind, "path": str(path) if path is not None else None,
                                  "status": "disabled"}
        metadata["sources"].append(source)
        if path is None:
            continue
        try:
            if kind == "workspace":
                try:
                    path.lstat()
                except FileNotFoundError:
                    source["status"] = "missing"
                    continue
            resolved = path.resolve(strict=True)
            if kind == "workspace" and root is not None and not resolved.is_relative_to(root):
                raise InstructionLoadError(f"Workspace instruction source escapes {root}: {path}")
            text = _read_instruction(resolved, config.max_source_bytes)
        except (OSError, UnicodeError, RuntimeError) as error:
            raise InstructionLoadError(f"Cannot load {kind} instructions from {path}: {error}") from error
        if kind == "system":
            text = text.strip()  # Preserve the original static system prompt's outer-whitespace policy.
            if not text:
                raise InstructionLoadError(f"System instructions are empty: {path}")
            blocks.append(text)
        else:
            label = "User defaults" if kind == "user" else "Workspace rules"
            blocks.append(f"# {label}\n\n{text}")
        encoded = text.encode("utf-8")
        source.update(path=str(resolved), status="loaded", sha256=sha256(encoded).hexdigest(),
                      byte_count=len(encoded))
    assembled = "\n\n".join(blocks)
    if len(assembled.encode("utf-8")) > config.max_total_bytes:
        raise InstructionLoadError(f"Assembled instructions exceed {config.max_total_bytes} bytes.")
    return assembled, metadata


class ContextBuilder:
    def build_messages(
        self,
        *,
        instructions: list[ChatCompletionRequestMessage],
        history: list[ChatCompletionRequestMessage],
        current_turn: list[ChatCompletionRequestMessage],
    ) -> list[ChatCompletionRequestMessage]:
        """Assemble an independent view, including nested tool-call dictionaries."""
        return deepcopy([*instructions, *history, *current_turn])
