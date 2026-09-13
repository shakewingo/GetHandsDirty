from pathlib import Path
import json
from typing import Any
import os
import sys
from tempfile import NamedTemporaryFile

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def resolve_path(root: str | Path, path: str | Path) -> Path:
    """Resolve a path under root, rejecting empty paths and symlink escapes."""
    if not str(path).strip():
        raise ValueError("Path must not be empty; use '.' for the root directory.")
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path must stay inside the configured root directory.")
    return resolved


def color_label(label: str, color: int) -> str:
    """Bold and color a terminal label; preserve plain redirected output."""
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return label
    return f"\033[1;{color}m{label}\033[0m"


def file_version(stat: os.stat_result) -> str:
    """Cheap change detection for a single-writer workspace, not a content hash."""
    return ":".join(str(value) for value in (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
    ))


def render_prompt(name: str, **context) -> str:
    """Render a prompt template by filename (e.g. 'system.jinja')."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(_PROMPTS_DIR), trim_blocks=True,
        lstrip_blocks=True, undefined=StrictUndefined, keep_trailing_newline=False,
    )
    return env.get_template(name).render(**context).strip()

def decode_qwen_tool_call(content: str) -> dict[str, Any]:
    """Decode a whole response block; tags inside JSON strings remain data."""
    text = content.strip()
    if not text.startswith("<tool_call>"):
        raise ValueError("Expected a top-level tool-call block.")
    payload = text[len("<tool_call>"):].lstrip()
    try:
        call, end = json.JSONDecoder().raw_decode(payload)
        if payload[end:].strip() != "</tool_call>":
            raise ValueError("Expected exactly one complete tool-call block.")
        # Qwen can use the same JSON-string arguments as a native tool call.
        if isinstance(call, dict) and isinstance(call.get("arguments"), str):
            call["arguments"] = json.loads(call["arguments"])
    except json.JSONDecodeError as error:
        raise ValueError("Invalid tool-call JSON") from error
    if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
            or not call["name"].strip() or not isinstance(call.get("arguments"), dict)):
        raise ValueError("Tool call requires a string name and object arguments.")
    return call


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Replace a local file atomically; a failed write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
