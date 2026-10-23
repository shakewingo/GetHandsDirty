"""Turn rows of a public tool-use dataset into (prompt, completion) SFT examples, as `export` does.

Usage: python -m agent_from_scratch.evals public --source {when2call-sft,when2call-pref,toucan}
           --input FILE [FILE ...] --revision SHA --student-gguf FILE --output DIR
           [--max-examples N] [--seed 0]

Each row becomes our conversation: our system prompt, the row's own tools in our template, its
tool results in our tool-result envelope, and one example per assistant turn. A row that cannot
be converted exactly is dropped and counted, never repaired or truncated. Rows are visited in a
seeded order and whole rows are kept until --max-examples is reached. Writes examples.jsonl and
metadata.json, which `train` and `mix` read. Design: docs/STAGE9_ITEM4_PLAN.md (D6, D7).
"""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re

from .export import render_turn
from ..run import save
from ...config import BENCH_N_CTX, CHAT_TEMPLATE_PATH, MAX_TOOL_CALLS_PER_RESPONSE, PROMPTS_DIR
from ...llm import compile_chat_template
from ...tools.base import ToolCall
from ...utils import write_jsonl

SOURCES = {"when2call-sft": ("nvidia/When2Call", "CC-BY-4.0"),
           "when2call-pref": ("nvidia/When2Call", "CC-BY-4.0"),
           "toucan": ("Agent-Ark/Toucan-1.5M", "Apache-2.0")}
# When2Call writes parameter types the BFCL way; our template expects JSON Schema names.
SCHEMA_TYPES = {"dict": "object", "Dict": "object", "str": "string", "int": "integer", "float": "number",
                "list": "array", "List": "array", "tuple": "array", "set": "array", "bool": "boolean",
                **{name: name for name in ("object", "string", "integer", "number", "array", "boolean", "null")}}
TOOLCALL = re.compile(r"<TOOLCALL>(.*)</TOOLCALL>", re.S)


class Dropped(Exception):
    """A row that cannot be converted exactly; `args[0]` is the reason counted in the funnel."""


def schema_type(name: str) -> str:
    """The JSON Schema name of a When2Call type such as "str", "int, optional" or "List[int]".

    The ", optional" and ", default=..." qualifiers are dropped (the schema's `required` list
    already says it); a generic container is an array whose items are left unstated. Union and
    Callable types have no faithful JSON Schema name here: Dropped.
    """
    base = re.sub(r",\s*(optional|default).*$", "", name).strip()
    if base in SCHEMA_TYPES:
        return SCHEMA_TYPES[base]
    if re.fullmatch(r"(List|Tuple|Set)\[.*\]", base) and not re.search(r"Union|Callable", base):
        return "array"
    raise Dropped("unknown_type")


def json_schema(schema):
    """`schema` with every When2Call type name mapped to its JSON Schema name; Dropped on an unknown one."""
    if isinstance(schema, list):
        return [json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    converted = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            value = schema_type(value)
        elif key == "properties" and isinstance(value, dict):
            value = {name: json_schema(prop) for name, prop in value.items()}
        elif isinstance(value, (dict, list)):
            value = json_schema(value)
        converted[key] = value
    return converted


def _tools(schemas: list[dict]) -> dict:
    tools = {}
    for schema in schemas:
        function = schema.get("function", schema)
        if not isinstance(function.get("name"), str) or function["name"] in tools:
            raise Dropped("bad_tools")
        tools[function["name"]] = {"type": "function", "function": {
            "name": function["name"], "description": function.get("description", ""),
            "parameters": json_schema(function.get("parameters", {"type": "object", "properties": {}}))}}
    return tools


def _envelope(call: ToolCall, output: str) -> dict:
    # The shape of ToolResult.to_message. Whether a public call succeeded is unknown, so `ok` is
    # true and any error text stays in `output`; tool results are context, never targets.
    content = {"call_id": call.call_id, "tool_name": call.name, "ok": True, "output": output,
               "error_code": None, "error_message": None}
    return {"role": "tool", "content": json.dumps(content), "tool_call_id": call.call_id}


def _assistant(content: str, calls: list[ToolCall]) -> dict:
    if "<think>" in content:
        raise Dropped("thinking")
    if len(calls) > MAX_TOOL_CALLS_PER_RESPONSE:
        raise Dropped("too_many_calls")
    if not content.strip() and not calls:
        raise Dropped("empty_turn")
    turn = {"role": "assistant", "content": content}
    if calls:
        turn["tool_calls"] = [call.to_dict() for call in calls]
    return turn


def when2call_row(row: dict, index: int, *, pref: bool) -> tuple[str, str, list[dict], dict]:
    """(row id, category, messages, tools) from a When2Call `train_sft` or `train_pref` row.

    For `train_pref` the chosen response is the assistant turn. The category is read from the
    reply: a call, a question (it contains "?"), or a decline.
    """
    messages = list(row["messages"]) + ([row["chosen_response"]] if pref else [])
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise Dropped("bad_roles")
    reply = messages[1]["content"]
    match = TOOLCALL.fullmatch(reply.strip())
    calls = []
    if match:
        try:
            calls = [ToolCall(item["name"], item["arguments"], f"call_1_{n}")
                     for n, item in enumerate(json.loads(match.group(1)), 1)]
        except (ValueError, KeyError, TypeError) as error:
            raise Dropped("bad_call") from error
        if not calls or any(not isinstance(call.arguments, dict) for call in calls):
            raise Dropped("bad_call")
    elif "TOOLCALL" in reply:
        raise Dropped("bad_call")
    category = "call" if calls else "question" if "?" in reply else "decline"
    turn = _assistant("" if calls else reply, calls)
    tools = _tools([json.loads(schema) for schema in row["tools"]])
    return f"row-{index}", category, [{"role": "user", "content": messages[0]["content"]}, turn], tools


def toucan_row(row: dict, index: int) -> tuple[str, str, list[dict], dict]:
    """(row id, category, messages, tools) from a Toucan SFT row.

    Toucan writes an assistant's narration, its calls and their results as separate messages
    (`assistant`, then `tool_call`s, then `tool_response`s); here they become one assistant turn
    with its calls, followed by one tool message per call, in order.
    """
    source = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    tools = _tools(json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"])
    messages, pending, narration, calls = [], [], None, []

    def close_turn():
        nonlocal narration, calls
        if narration is not None or calls:
            messages.append(_assistant(narration or "", calls))
            pending.extend(calls)
        narration, calls = None, []

    for message in source:
        role, content = message["role"], message.get("content") or ""
        if message.get("reasoning_content"):
            raise Dropped("thinking")
        if role == "tool_call":
            if pending:
                raise Dropped("unpaired")
            try:
                parsed = ast.literal_eval(content)
                arguments = parsed["arguments"]
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                name = parsed["name"]
            except (ValueError, SyntaxError, KeyError, TypeError) as error:
                raise Dropped("bad_call") from error
            if not isinstance(arguments, dict) or not isinstance(name, str):
                raise Dropped("bad_call")
            calls.append(ToolCall(name, arguments, f"call_{len(messages) + 1}_{len(calls) + 1}"))
        elif role == "tool_response":
            close_turn()
            if not pending:
                raise Dropped("unpaired")
            messages.append(_envelope(pending.pop(0), content))
        elif role == "assistant":
            close_turn()
            if pending:
                raise Dropped("unpaired")
            narration = content
        elif role == "user":
            close_turn()
            if pending:
                raise Dropped("unpaired")
            messages.append({"role": "user", "content": content})
        else:
            raise Dropped("bad_roles")
    close_turn()
    if pending or not messages or messages[-1]["role"] != "assistant":
        raise Dropped("unpaired")
    return str(row.get("uuid", f"row-{index}")), str(row.get("subset_name", "toucan")), messages, tools


def convert_row(row_id: str, messages: list[dict], tools: dict, template, system: str, count) -> list[dict]:
    """One (prompt, completion) per assistant turn of a converted row; Dropped if any is not exact.

    Args:
        messages: the row as our conversation, without the system message.
        count: text -> student tokens; every prompt plus completion must fit `BENCH_N_CTX`.
    """
    history, examples = [{"role": "system", "content": system}], []
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            try:
                prompt, completion = render_turn(template, history, message, tools)
            except ValueError as error:
                raise Dropped("round_trip") from error
            examples.append({"turn_index": index, "prompt": prompt, "completion": completion})
        history.append(message)
    if not examples:
        raise Dropped("empty_turn")
    # The last example holds the longest text of the row.
    if count(examples[-1]["prompt"] + examples[-1]["completion"]) > BENCH_N_CTX:
        raise Dropped("too_long")
    return examples


def read_rows(paths: list[Path]):
    """Rows from JSONL files, or from parquet files (needs pyarrow), in file order."""
    for path in paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet

            yield from pyarrow.parquet.read_table(path).to_pylist()
        else:
            with path.open(encoding="utf-8") as stream:
                yield from (json.loads(line) for line in stream if line.strip())


def collect(rows: list[dict], source: str, *, count, max_examples: int | None = None,
            seed: int = 0, caps: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    """(examples, funnel) from `rows` visited in a seeded order, whole rows at a time.

    `caps` limits the kept rows of a category (e.g. {"question": 200}); rows past it are skipped
    and counted as "capped".
    """
    text = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    template = compile_chat_template(text)
    system = (PROMPTS_DIR / "system.md").read_text(encoding="utf-8").strip()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    examples, dropped, categories, kept_rows, visited = [], Counter(), Counter(), 0, 0
    for index in order:
        if max_examples is not None and len(examples) >= max_examples:
            break
        visited += 1
        try:
            if source == "toucan":
                row_id, category, messages, tools = toucan_row(rows[index], index)
            else:
                row_id, category, messages, tools = when2call_row(rows[index], index,
                                                                  pref=source == "when2call-pref")
            turns = convert_row(row_id, messages, tools, template, system, count)
        except Dropped as reason:
            dropped[reason.args[0]] += 1
            continue
        except (KeyError, TypeError, ValueError, SyntaxError):
            dropped["bad_row"] += 1
            continue
        if caps and category in caps and categories[category] >= caps[category]:
            dropped["capped"] += 1
            continue
        kept_rows += 1
        categories[category] += 1
        for turn in turns:
            pair = turn["prompt"] + "\0" + turn["completion"]
            examples.append({"id": sha256(pair.encode("utf-8")).hexdigest()[:16],
                             "prompt": turn["prompt"], "completion": turn["completion"],
                             "provenance": {"track": "public", "pilot": False, "source": source,
                                            "skeleton": f"{source}:{category}", "family": "public",
                                            "condition": None, "task_id": row_id, "seed": None,
                                            "run_dir": source, "run_id": row_id,
                                            "request_index": turn["turn_index"], "dropped_turns": [],
                                            "raw_matches_canonical": None}})
    funnel = {"rows": len(rows), "rows_visited": visited, "dropped": dict(sorted(dropped.items())),
              "kept_rows": kept_rows, "kept_rows_by_category": dict(sorted(categories.items())),
              "examples": len(examples)}
    return examples, funnel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Downloaded JSONL or parquet files")
    parser.add_argument("--revision", required=True, help="The dataset commit the files were downloaded at")
    parser.add_argument("--student-gguf", type=Path, required=True,
                        help="A GGUF of the student; its tokenizer checks that every row fits the window")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, help="Stop after the row that reaches this many examples")
    parser.add_argument("--seed", type=int, default=0, help="Fixes the order in which rows are visited")
    parser.add_argument("--cap", action="append", default=[], metavar="CATEGORY=N",
                        help="Keep at most N rows of a category, e.g. question=200 (repeatable)")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Evidence must be outside the source package.")
    if not args.student_gguf.is_file():
        parser.error("--student-gguf must be an existing GGUF of the student model.")
    from .teacher import student_token_counter

    count, tokenizer = student_token_counter(args.student_gguf)
    rows = list(read_rows(args.input))
    try:
        caps = {name: int(n) for name, n in (cap.split("=") for cap in args.cap)}
    except ValueError:
        parser.error("--cap takes CATEGORY=N.")
    examples, funnel = collect(rows, args.source, count=count, max_examples=args.max_examples, seed=args.seed,
                               caps=caps)
    if not examples:
        parser.error(f"No example to write; funnel: {json.dumps(funnel)}")
    repository, license_name = SOURCES[args.source]
    text = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8").encode("utf-8")
    out.mkdir(parents=True)
    write_jsonl(out / "examples.jsonl", examples)
    # `train` and `mix` read track, pilot, flawed_turns, max_per_task and chat_template_sha256.
    save(out / "metadata.json", {
        "track": "public", "pilot": False, "flawed_turns": None, "max_per_task": None,
        "chat_template_sha256": sha256(text).hexdigest(), "target": "canonical",
        "system_prompt_sha256": sha256((PROMPTS_DIR / "system.md").read_bytes()).hexdigest(),
        "source": {"name": args.source, "repository": repository, "revision": args.revision,
                   "license": license_name, "files": [path.name for path in args.input],
                   "files_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in args.input}},
        "budget_tokenizer": tokenizer, "n_ctx": BENCH_N_CTX,
        "sampling": {"seed": args.seed, "max_examples": args.max_examples, "caps": caps}, "funnel": funnel})
    print(json.dumps(funnel, indent=2))


if __name__ == "__main__":
    main()
