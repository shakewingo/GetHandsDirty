"""Turn verifier-passed train rollouts into (prompt, completion) SFT examples for one track.

Usage: python -m agent_from_scratch.evals export --track {teacher,rft} --output DIR
           [--max-per-task K] [--pilot] [--flawed-turns {reject,drop}] RUN_DIR [RUN_DIR ...]

Each example is one model request: `prompt` is the recorded input rendered by the inference
renderer, `completion` is the assistant turn that followed, as the template writes it. Only
`bench --split train` runs are accepted. A rollout is used when the verifier passed it and it
holds none of the model's own mistakes; per task, identical trajectories collapse and at most K
distinct ones are kept. Writes examples.jsonl and metadata.json (sources and the funnel).
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ..run import save
from ...config import CHAT_TEMPLATE_PATH
from ...llm import LLM, QWEN_END, ResponseError, compile_chat_template, render_chat
from ...utils import write_jsonl

# The backend whose rollouts each track is made of, as `settings()["backend"]` records it.
TRACK_BACKENDS = {"teacher": "deepseek", "rft": "vllm"}


def render_turn(template, messages: list[dict], turn: dict, tools) -> tuple[str, str]:
    """(prompt, completion) for one assistant `turn` that follows `messages`.

    The prompt is what the model saw; the completion is what the template adds when the turn
    joins the conversation, without the newline after its stop token. Raises ValueError unless
    the completion parses back to the turn's calls and content.
    """
    prompt = render_chat(template, messages, tools)
    full = render_chat(template, [*messages, turn], tools, add_generation_prompt=False)
    if not full.startswith(prompt) or not full.endswith(QWEN_END + "\n"):
        raise ValueError("the rendered turn does not extend its prompt.")
    # The server stops on <|im_end|>, so the newline the template writes after it is never generated.
    completion = full[len(prompt):].removesuffix("\n")
    # Round trip: the text the model is trained to write must parse to the action it took.
    calls = turn.get("tool_calls") or []
    try:
        parsed = LLM.parse_response({"choices": [{"message": {
            "role": "assistant", "content": completion.removesuffix(QWEN_END)}}]}, len(calls))
    except ResponseError as error:
        raise ValueError(f"the target does not parse: {error}") from error
    taken = [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in calls]
    if ([(call.name, call.arguments) for call in parsed.tool_calls] != taken
            or parsed.content != (turn.get("content") or "")):
        raise ValueError("the target does not parse back to the recorded turn.")
    return prompt, completion


def examples_from_trace(trace: dict, template) -> list[dict]:
    """One example per completed actor request, in request order.

    Args:
        trace: the JSON object in `state/runs/<run_id>.jsonl` (schema 5).
        template: `compile_chat_template(...)` of prompts/qwen_chat.jinja.

    Returns:
        list of {"request_index", "prompt", "completion", "raw_matches_canonical"}.
    """
    messages, examples = trace["messages"], []
    for index, request in enumerate(trace["model_requests"]):
        if request["purpose"] != "agent" or request["status"] != "completed":
            continue
        count = request["input_message_count"]
        turn = messages[count] if count < len(messages) else {}
        calls = turn.get("tool_calls") or []
        if turn.get("role") != "assistant" or [call["id"] for call in calls] != request["call_ids"]:
            raise ValueError(f"Request {index} does not line up with messages[{count}].")
        try:
            prompt, completion = render_turn(template, request["input_messages"], turn, request["tools"])
        except ValueError as error:
            raise ValueError(f"Request {index}: {error}") from error
        try:
            sampled = request["raw_response"]["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            sampled = None  # Not the vLLM completion shape: nothing was sampled as Qwen text.
        examples.append({"request_index": index, "prompt": prompt, "completion": completion,
                         "raw_matches_canonical": None if sampled is None
                         else sampled + QWEN_END == completion})
    return examples


def flawed_turns(record: dict, task: dict, trace: dict) -> dict[int, str]:
    """Request index -> why that turn must not be a target, in a rollout the verifier passed.

    `own_error`: the turn holds a failed tool call of the model's own making, which is one that
    is neither the injected fault nor a call skipped after an earlier failure in its batch.
    `loose_answer`: the final answer is not exactly the requested value. The verifier accepts
    that on tasks whose answer format is advisory.

    Args:
        record: the task's entry in results.json; `checks.format_exact` is read from it.
        task: the task's task.json, read for `fault_signal` only.
        trace: the run trace.
    """
    signal = task.get("fault_signal")
    failed = set()
    for message in trace["messages"]:
        if message["role"] != "tool":
            continue
        result = json.loads(message["content"])
        # Recovering from the injected fault is a target; the model's own slips are not.
        injected = (bool(signal) and result["tool_name"] == signal[0]
                    and signal[1] in (result["error_message"] or ""))
        if not result["ok"] and result["error_code"] != "skipped" and not injected:
            failed.add(message["tool_call_id"])
    requests = trace["model_requests"]
    flawed = {index: "own_error" for index, request in enumerate(requests)
              if failed.intersection(request["call_ids"])}
    if record["checks"].get("format_exact") is False and requests and not requests[-1]["call_ids"]:
        flawed.setdefault(len(requests) - 1, "loose_answer")
    return flawed


def rejection(record: dict, task: dict, trace: dict, *, flawed: str = "reject") -> str | None:
    """Rejection sampling - Why a rollout may not become training data; None when it may.

    Args:
        record: the task's entry in results.json (verifier outcome plus metrics()).
        task: the task's task.json, read for `fault_signal` only.
        trace: the run trace.
        flawed: "reject" refuses a rollout that has any flawed turn (see `flawed_turns`);
            "drop" accepts it, and the caller leaves those turns out of the examples.
    """
    if not record["passed"]:
        return "failed"
    reasons = set(flawed_turns(record, task, trace).values()) if flawed == "reject" else set()
    # A parse error or a stuck reminder puts runtime feedback into the history, so neither
    # mode keeps such a rollout.
    if record["parse_errors"] or record["stuck_reminders"] or "own_error" in reasons:
        return "own_error"
    requests = trace["model_requests"]
    if any(request["purpose"] != "agent" or request["status"] != "completed"
           or request["input_messages"] != trace["messages"][:request["input_message_count"]]
           for request in requests):
        return "edited_view"
    if "loose_answer" in reasons:
        return "loose_answer"
    return None


def select(trajectories: list[dict], max_per_task: int) -> list[dict]:
    """Distinct trajectories per task, at most `max_per_task` each: fewest own-error turns first
    (those stay in the history of later examples), then fewest dropped turns, then shortest.

    Each trajectory has at least `task_id`, `seed`, `run_dir` and `examples`.
    """
    def rank(trajectory: dict) -> tuple:
        dropped = [example.get("dropped") for example in trajectory["examples"]]
        return (trajectory["task_id"], dropped.count("own_error"), sum(map(bool, dropped)),
                len(dropped), trajectory["seed"], trajectory["run_dir"])
    kept, distinct = [], {}
    for trajectory in sorted(trajectories, key=rank):
        actions = tuple(example["completion"] for example in trajectory["examples"])
        seen = distinct.setdefault(trajectory["task_id"], set())
        if actions in seen or len(seen) >= max_per_task:
            continue
        seen.add(actions)
        kept.append(trajectory)
    return kept


def collect(run_dirs: list[Path], track: str, *, max_per_task: int = 2, pilot: bool = False,
            flawed: str = "reject") -> tuple[list[dict], dict]:
    """Read bench run directories and return (examples, metadata) for one track.

    `flawed` is "reject" or "drop": what to do with a verified rollout that has flawed turns
    (see `flawed_turns`), refuse the rollout or leave only those turns out.
    """
    text = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    template, template_sha256 = compile_chat_template(text), sha256(text.encode("utf-8")).hexdigest()
    names = [run.name for run in run_dirs]
    if len(set(names)) != len(names):
        raise ValueError("Run directories must have distinct names; provenance records the name.")
    sources, trajectories, rollouts = [], [], set()
    rejected = dict.fromkeys(("failed", "own_error", "edited_view", "loose_answer"), 0)
    dropped = dict.fromkeys(("own_error", "loose_answer"), 0)
    for run in run_dirs:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        records = json.loads((run / "results.json").read_text(encoding="utf-8"))
        settings, seed = metadata["settings"], metadata["seed"]
        if metadata["split"] != "train" or any(record["split"] != "train" for record in records):
            raise ValueError(f"{run.name}: only the train split may become training data.")
        if settings["backend"] == "scripted":
            raise ValueError(f"{run.name}: scripted gate fixtures are never training data.")
        if settings["backend"] != TRACK_BACKENDS[track] and not pilot:
            raise ValueError(f"{run.name}: backend {settings['backend']!r} is not the {track} track's "
                             f"({TRACK_BACKENDS[track]!r}); --pilot marks a pipeline check.")
        if settings.get("chat_template_sha256") not in (None, template_sha256):
            raise ValueError(f"{run.name}: sampled under a different chat template than the one on disk.")
        sources.append({"run_dir": run.name, "seed": seed, "settings": settings,
                        "git_revision": metadata["git_revision"]})
        for record in records:
            if (record["id"], seed) in rollouts:
                raise ValueError(f"Duplicate rollout: {record['id']} at seed {seed}.")
            rollouts.add((record["id"], seed))
            case = run / record["id"]
            task = json.loads((case / "task.json").read_text(encoding="utf-8"))
            trace = json.loads((case / "state" / "runs" / f"{record['run_id']}.jsonl")
                               .read_text(encoding="utf-8"))
            if trace["schema_version"] != 5:
                raise ValueError(f"{run.name}/{record['id']}: trace schema {trace['schema_version']}, not 5.")
            reason = rejection(record, task, trace, flawed=flawed)
            if reason is not None:
                rejected[reason] += 1
                continue
            drop = flawed_turns(record, task, trace) if flawed == "drop" else {}
            trajectories.append({"task_id": record["id"], "skeleton": record["skeleton"],
                                 "family": record["family"], "condition": record["condition"],
                                 "seed": seed, "run_dir": run.name, "run_id": record["run_id"],
                                 "examples": [{**example, "dropped": drop.get(example["request_index"])}
                                              for example in examples_from_trace(trace, template)]})
    kept = select(trajectories, max_per_task)
    examples, before, histories, by_skeleton = [], 0, set(), {}
    for trajectory in kept:
        group = by_skeleton.setdefault(trajectory["skeleton"], {"trajectories": 0, "tasks": set()})
        group["trajectories"] += 1
        group["tasks"].add(trajectory["task_id"])
        provenance = {"track": track, "pilot": pilot,
                      **{key: trajectory[key] for key in ("task_id", "skeleton", "family", "condition",
                                                          "seed", "run_dir", "run_id")},
                      # Turns of this rollout that are left out. An own_error turn is still in
                      # the prompt of every later example, as history.
                      "dropped_turns": [example["request_index"] for example in trajectory["examples"]
                                        if example["dropped"]]}
        history = ()
        for example in trajectory["examples"]:
            # Prompts never repeat across rollouts (each names its own temporary workspace), so
            # a shared prefix is recognised by the task and the actions taken so far. A dropped
            # turn stays in that history: what follows it was decided after it.
            history += (example["completion"],)
            if example["dropped"]:
                dropped[example["dropped"]] += 1
                continue
            before += 1
            if (trajectory["task_id"], history) in histories:
                continue
            histories.add((trajectory["task_id"], history))
            pair = example["prompt"] + "\0" + example["completion"]
            examples.append({"id": sha256(pair.encode("utf-8")).hexdigest()[:16],
                             "prompt": example["prompt"], "completion": example["completion"],
                             "provenance": {**provenance, "request_index": example["request_index"],
                                            "raw_matches_canonical": example["raw_matches_canonical"]}})
    funnel = {"rollouts": len(rollouts), "rejected": rejected, "eligible": len(trajectories),
              "kept_trajectories": len(kept), "tasks_covered": len({t["task_id"] for t in kept}),
              "dropped_turns": dropped, "examples_before_dedup": before, "examples": len(examples),
              "by_skeleton": {name: {"trajectories": group["trajectories"], "tasks": len(group["tasks"])}
                              for name, group in sorted(by_skeleton.items())}}
    return examples, {"track": track, "pilot": pilot, "max_per_task": max_per_task,
                      "flawed_turns": flawed, "chat_template_sha256": template_sha256, "target": "canonical",
                      "sources": sources, "funnel": funnel}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="bench --split train output directories")
    parser.add_argument("--track", choices=tuple(TRACK_BACKENDS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the dataset")
    parser.add_argument("--max-per-task", type=int, default=2,
                        help="Distinct trajectories kept per task, shortest first")
    parser.add_argument("--pilot", action="store_true",
                        help="Accept another backend's runs and mark every example as a pilot, if pilot==True, this dataset is a pipeline check not a real training data, it allows to run using llama.cpp as backend locally")
    parser.add_argument("--flawed-turns", choices=("reject", "drop"), default="reject",
                        help="A verified rollout with the model's own failed tool call, or a final answer "
                             "that is not exactly the requested value: reject the rollout, or drop only "
                             "those turns")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Evidence must be outside the source package.")
    try:
        examples, metadata = collect(args.runs, args.track, max_per_task=args.max_per_task,
                                     pilot=args.pilot, flawed=args.flawed_turns)
    except (ValueError, OSError, KeyError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    if not examples:
        parser.error(f"No example to write; funnel: {json.dumps(metadata['funnel'])}")
    out.mkdir(parents=True)
    write_jsonl(out / "examples.jsonl", examples)
    save(out / "metadata.json", metadata)
    print(json.dumps(metadata["funnel"], indent=2))


if __name__ == "__main__":
    main()