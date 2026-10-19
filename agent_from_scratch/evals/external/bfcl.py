"""BFCL multi-turn base, run locally (Stage 9 Item 5). A local adaptation, never an official score.

Usage: python -m agent_from_scratch.evals bfcl --bfcl-root DIR --output DIR [--adapter NAME]
           [--limit N] [--workers 8] [--system-prompt {qwen-default,ours}]

DIR is `berkeley-function-call-leaderboard` from ShishirPatil/gorilla. Each entry's user turns are
given to the model in a plain chat loop with the entry's own tools (not our agent harness),
greedy, at most 20 steps per turn. Calls are rebuilt from the parsed JSON arguments and run on
BFCL's own simulated backends; BFCL's own `multi_turn_checker` scores the entry. Design:
docs/STAGE9_ITEM5_6_PLAN.md (E3).
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys

from .when2call import Server
from ..run import save
from ..sft.public import Dropped, json_schema
from ...config import BENCH_MODEL, CHAT_TEMPLATE_PATH, PROMPTS_DIR, VLLM_BASE_URL
from ...llm import LLM, ResponseError, compile_chat_template, render_chat
from ...tools.base import ToolCall
from ...utils import write_jsonl

CATEGORY = "multi_turn_base"
MAX_STEPS = 20
DOC_FILES = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
             "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
             "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control"}


def load(root: Path) -> tuple[list[dict], dict, dict]:
    """(entries, ground truth by id, function docs by class) of the base category."""
    data = root / "bfcl_eval" / "data"
    lines = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    entries = lines(data / f"BFCL_v4_{CATEGORY}.json")
    truth = {row["id"]: row["ground_truth"] for row in lines(data / "possible_answer" / f"BFCL_v4_{CATEGORY}.json")}
    docs = {name: lines(data / "multi_turn_func_doc" / f"{file}.json") for name, file in DOC_FILES.items()}
    return entries, truth, docs


def tools_for(entry: dict, docs: dict) -> dict:
    """The entry's tools in our template's shape: its classes' functions, minus the excluded ones."""
    excluded = set(entry.get("excluded_function") or [])
    tools = {}
    for name in entry["involved_classes"]:
        for doc in docs[name]:
            if doc["name"] in excluded:
                continue
            tools[doc["name"]] = {"type": "function", "function": {
                "name": doc["name"], "description": doc["description"],
                "parameters": json_schema(doc["parameters"])}}
    return tools


def call_string(call: ToolCall) -> str:
    """BFCL's call syntax, e.g. cd(folder='document'). Values are Python literals via repr, so
    evaluating the string can only call the named function."""
    return f"{call.name}({', '.join(f'{key}={value!r}' for key, value in call.arguments.items())})"


def run_entry(entry: dict, tools: dict, generate, execute, system: str | None = None
              ) -> tuple[list[list[list[str]]], list[dict], str | None]:
    """Play every user turn of one entry; returns (calls by turn and step, messages, stop reason).

    Args:
        generate: messages -> the model's raw text for the next assistant turn.
        execute: call strings -> their result strings, on this entry's live backends.
    """
    # Without a system message our template writes Qwen's default line, as BFCL's Qwen handler does.
    messages = [{"role": "system", "content": system}] if system else []
    decoded, stop = [], None
    for turn in entry["question"]:
        messages += copy.deepcopy(turn)
        steps = []
        for _ in range(MAX_STEPS):
            try:
                text = generate(messages)
            except Exception as error:  # an over-long prompt, a server error
                stop = f"generate_failed: {type(error).__name__}"
                break
            try:
                parsed = LLM.parse_response({"choices": [{"message": {"role": "assistant", "content": text}}]}, 64)
            except ResponseError:
                messages.append({"role": "assistant", "content": text})
                break
            if not parsed.tool_calls:
                messages.append({"role": "assistant", "content": parsed.content})
                break
            # A function that was not offered is never executed or passed to the checker; it
            # cannot change any backend state, and its error is reported to the model.
            offered = [call for call in parsed.tool_calls if call.name in tools]
            strings = [call_string(call) for call in offered]
            results = iter(execute(strings) if strings else [])
            for number, call in enumerate(parsed.tool_calls):
                call.call_id = f"call_{len(messages)}_{number}"
            messages.append({"role": "assistant", "content": parsed.content,
                             "tool_calls": [call.to_dict() for call in parsed.tool_calls]})
            for call in parsed.tool_calls:
                content = next(results) if call.name in tools else f"Error: {call.name} is not an available function."
                messages.append({"role": "tool", "content": content, "tool_call_id": call.call_id})
            if strings:
                steps.append(strings)
        else:
            stop = stop or "max_steps"
        decoded.append(steps)
        if stop and stop.startswith("generate_failed"):
            # The remaining turns count as empty, as an aborted inference would leave them.
            decoded += [[] for _ in entry["question"][len(decoded):]]
            break
    return decoded, messages, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bfcl-root", type=Path, required=True, help="gorilla/berkeley-function-call-leaderboard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", help="LoRA module name on the server; omit for the base model")
    parser.add_argument("--base-url", default=VLLM_BASE_URL)
    parser.add_argument("--served-model", default=BENCH_MODEL["id"])
    parser.add_argument("--limit", type=int, help="First N entries only (a pipeline check)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--system-prompt", choices=("qwen-default", "ours"), default="qwen-default",
                        help="ours: prompts/system.md as the system message (a diagnostic, Stage 9 Item 7)")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Evidence must be outside the source package.")
    if not (args.bfcl_root / "bfcl_eval" / "data").is_dir():
        parser.error("--bfcl-root must hold bfcl_eval/data.")
    try:
        server = Server(args.base_url, args.served_model, args.adapter)
    except Exception as error:
        parser.error(str(error))
    # BFCL's own executor and checker, imported from the checkout.
    sys.path.insert(0, str(args.bfcl_root.resolve()))
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    entries, truth, docs = load(args.bfcl_root)
    entries = entries[:args.limit]
    template = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    system = ((PROMPTS_DIR / "system.md").read_text(encoding="utf-8").strip()
              if args.system_prompt == "ours" else None)
    # Backend instances live in a module dictionary keyed by this name and the entry ID.
    run_name = f"local_{server.model}_{out.name}".replace("-", "_")

    def one(entry):
        try:
            tools = tools_for(entry, docs)
        except Dropped as reason:
            return {"id": entry["id"], "valid": False, "error_type": f"tools:{reason.args[0]}"}

        def generate(messages):
            return server.complete(render_chat(template, messages, tools))

        def execute(strings):
            return execute_multi_turn_func_call(strings, entry["initial_config"], entry["involved_classes"],
                                                run_name, entry["id"])[0]
        decoded, messages, stop = run_entry(entry, tools, generate, execute, system)
        verdict = multi_turn_checker(decoded, truth[entry["id"]], entry, CATEGORY, run_name)
        return {"id": entry["id"], "valid": verdict["valid"], "error_type": verdict.get("error_type"),
                "stop": stop, "decoded": decoded, "turns": len(entry["question"]),
                "messages": messages}

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, entries))
    out.mkdir(parents=True)
    write_jsonl(out / "results.jsonl", results)
    scores = {"entries": len(results), "accuracy": round(sum(r["valid"] for r in results) / len(results), 4),
              "errors": dict(sorted(Counter(r["error_type"] for r in results if not r["valid"]).items())),
              "stopped": dict(Counter(r.get("stop") for r in results if r.get("stop")))}
    save(out / "scores.json", scores)
    save(out / "metadata.json", {"category": CATEGORY, "model": server.model, "decoding": "greedy",
                                 "max_steps_per_turn": MAX_STEPS, "template": "prompts/qwen_chat.jinja", "system_prompt": args.system_prompt,
                                 "label": "local adaptation of BFCL multi-turn base; not an official score"})
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
