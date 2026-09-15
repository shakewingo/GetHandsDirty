from pathlib import Path
import json
import re
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

_TOOL_OR_QUOTE = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*(?:\n|$)"
    r"|^ {0,3}>[^\n]*(?:\n|$)|(?P<ticks>`+)|(?P<tag></?tool_call>)",
    re.MULTILINE,
)


def _next_tool_tag(content: str, start: int = 0) -> re.Match[str] | None:
    """Find a protocol tag outside fenced/inline code and Markdown quote lines."""
    while match := _TOOL_OR_QUOTE.search(content, start):
        if match.group("tag"):
            return match
        if fence := match.group("fence"):
            closing = re.compile(
                r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*(?:\n|$)",
                re.MULTILINE,
            ).search(content, match.end())
            if closing is None:
                return None  # The rest of an unclosed fence is still quoted code.
            start = closing.end()
        elif ticks := match.group("ticks"):
            closing = re.compile(r"(?<!`)" + re.escape(ticks) + r"(?!`)").search(content, match.end())
            start = closing.end() if closing else match.end()
        else:
            start = match.end()  # Skip this blockquote line.
    return None


def extract_qwen_tool_calls(content: str) -> tuple[list[dict[str, Any]], str]:
    """Extract ordered unquoted calls; decode JSON before looking for closing tags."""
    calls, narration, cursor = [], [], 0
    opening = _next_tool_tag(content)
    while opening is not None:
        if opening.group("tag") != "<tool_call>":
            raise ValueError("Unexpected closing tool-call tag.")
        start = opening.end()
        while start < len(content) and content[start].isspace():
            start += 1
        try:
            call, end = json.JSONDecoder().raw_decode(content, start)
            if isinstance(call, dict) and isinstance(call.get("arguments"), str):
                call["arguments"] = json.loads(call["arguments"])
        except json.JSONDecodeError as error:
            raise ValueError("Invalid tool-call JSON") from error
        while end < len(content) and content[end].isspace():
            end += 1
        if not content.startswith("</tool_call>", end):
            raise ValueError("Expected a closing tag immediately after the tool-call JSON.")
        if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                or not call["name"].strip() or not isinstance(call.get("arguments"), dict)):
            raise ValueError("Tool call requires a string name and object arguments.")
        calls.append(call)
        narration.append(content[cursor:opening.start()])
        cursor = end + len("</tool_call>")
        opening = _next_tool_tag(content, cursor)
    narration.append(content[cursor:])
    return calls, "".join(narration).strip()


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
