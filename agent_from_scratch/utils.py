from pathlib import Path
import json
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def render_prompt(name: str, **context) -> str:
    """Render a prompt template by filename (e.g. 'system.jinja')."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(_PROMPTS_DIR), trim_blocks=True,
        lstrip_blocks=True, undefined=StrictUndefined, keep_trailing_newline=False,
    )
    return env.get_template(name).render(**context).strip()

def decode_qwen_tool_call(content: str) -> dict[str, Any]:
    # TODO; may later on extend to a structued json output helper
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"

    if content.count(start_tag) != 1 or content.count(end_tag) != 1:
        raise ValueError("Expected exactly one complete tool-call block.")
    start = content.index(start_tag) + len(start_tag)
    end = content.index(end_tag, start)
    payload = content[start:end].strip()

    # Compatibility with Qwen's extra outer braces: {{...}}
    if payload.startswith("{{") and payload.endswith("}}"):
        payload = payload[1:-1]
    try:
        call = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid tool-call JSON") from error
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
        raise ValueError("Tool call requires a string name and object arguments.")
    return call
