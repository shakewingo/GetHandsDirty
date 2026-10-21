"""Build Stage 8 benchmark tasks from seeds, run them, and score them.

Usage: python -m agent_from_scratch.evals bench --output DIR --split {train,val,test,calib,hardtest} [--tasks a,b]
           [--seed N] [--backend {llama_cpp,vllm,teacher}] [--base-url URL] [--served-model NAME]
           [--adapter NAME] [--limits '{"planning": true}'] [--final --manifest FILE]
           [--teacher-model ID] [--teacher-effort {none,low,high,max}] [--student-gguf FILE]

Every run uses the frozen sampled protocol (config.BENCH_DECODING); `--seed` picks the sample.
Run once per seed in config.BENCH_SEEDS, then `aggregate` the runs. Val is where every choice is
made; test is run once per final candidate, with --final and a versioned dataset --manifest.
Stage 10 adds calib (every shape at every level, to choose each shape's level) and hardtest
(the hard shapes at their chosen level, under the same rule as test).
"""

import argparse
from dataclasses import asdict, replace
import json
import re
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

from .faults import FaultyTool
from .skeletons import Skeleton, registered
from .spec import BuildContext, Task
from .verify import EVIDENCE_TOOLS, bench_summary, check
from ..run import open_run, parse_limits, save, seed_every_request
from ..trajectory import profile
from ..verify import exchanges, metrics, snapshot


# Replies that stop to ask or to decline; a task that expects one of these tokens asks correctly.
ASK_TOKENS = {"NEED_INPUT", "NEED_UNIT", "UNSURE", "MISSING", "NOT_FOUND", "LOCKED", "CONFLICT", "BLOCKED"}
ASKING = re.compile(r"\?\s*$|(?i:\b(unable|cannot|can't|could you|please (provide|specify|clarify))\b)")


def asked_instead(task: Task, record: dict) -> bool:
    """The run failed with a final reply that asks or declines, where acting was expected."""
    expected = task.expect.answer.value if task.expect.answer else ""
    reply = (record.get("final_answer") or "").strip()
    return not record["passed"] and expected.split(" ")[0] not in ASK_TOKENS and bool(ASKING.search(reply))


REQUEST_COUNTS = ("model_requests", "actor_requests", "compact_requests", "compactions_applied",
                  "parse_errors", "requests_without_usage", "elapsed_seconds", "stuck_reminders")
INVALID_REQUEST_CODES = {"invalid_tool_call", "multiple_tool_calls", "too_many_tool_calls"}


def turn_metrics(results: list) -> dict:
    """`metrics` of a task played over several turns, one TurnResult per turn.

    Each turn's messages hold every earlier turn, so the last result alone gives the tool rows;
    request counts, usage and time add up over the turns.
    """
    merged = metrics(results[-1])
    for earlier in results[:-1]:
        part = metrics(earlier)
        for key in REQUEST_COUNTS:
            merged[key] = round(merged[key] + part[key], 2) if key == "elapsed_seconds" else merged[key] + part[key]
        merged["invalid_calls"] += sum(q.status != "blocked" and q.error_code in INVALID_REQUEST_CODES
                                       for q in earlier.model_requests)
        merged["max_actor_prompt_tokens"] = max(merged["max_actor_prompt_tokens"], part["max_actor_prompt_tokens"])
        merged["usage"] = {key: None if merged["usage"][key] is None or part["usage"][key] is None
                           else merged["usage"][key] + part["usage"][key] for key in merged["usage"]}
        merged["actions"] = part["actions"] + merged["actions"]
    # A turn that did not end normally ends the task; report the first such reason.
    merged["stop_reason"] = next((r.stop_reason for r in results if r.stop_reason != "final_response"),
                                 results[-1].stop_reason)
    merged["run_ids"] = [r.run_id for r in results]
    return merged
from ...agent import Agent
from ...config import BENCH_DECODING, BENCH_MODEL, BENCH_N_CTX, MODEL_PATH, VLLM_BASE_URL
from ...llm import LLM, VLLMClient
from ...tools.base import ToolRegistry
from ...tools.calculator import CalculatorTool
from ...tools.files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from ...tools.search import GlobFilesTool, GrepTextTool
from ...tools.shell import Command, ShellTool

HERE = Path(__file__).resolve().parent
CHECK_SCRIPT = HERE / "check.py"


def specs(split: str | None = None) -> list[tuple[Skeleton, BuildContext]]:
    """(skeleton, BuildContext) pairs in registration order; reads only levels.json.

    `split` None is every split but calibration; "calib" is only calibration (`registered`).
    """
    entries = []
    for skeleton in registered(split):
        for seed in skeleton.seeds:
            for condition in (("clean", "fault") if skeleton.recovery else (None,)):
                entries.append((skeleton, BuildContext(skeleton.name, skeleton.family,
                                                        skeleton.split, seed, condition)))
    return entries


def registry_for(task: Task, workspace: Path, private: Path) -> ToolRegistry:
    """The confined general registry, narrowed to `task.tools`, with fault/shell wiring."""
    available = {"calculator": CalculatorTool(),
                "list_files": ListFilesTool(workspace, restrict_to_workspace=True),
                "glob_files": GlobFilesTool(workspace, restrict_to_workspace=True),
                "grep_text": GrepTextTool(workspace, restrict_to_workspace=True),
                "read_file": ReadFileTool(workspace, restrict_to_workspace=True),
                "write_file": WriteFileTool(workspace, restrict_to_workspace=True),
                "edit_file": EditFileTool(workspace, restrict_to_workspace=True)}
    if "shell" in task.tools:
        spec_path = private / "check-spec.json"
        spec_path.write_text(json.dumps({"path": "config.json", "value": task.debug["correct"]}))
        available["shell"] = ShellTool(workspace, {"check_fixture": Command(
            (sys.executable, "-I", str(CHECK_SCRIPT), str(spec_path)),
            "Check whether config.json matches the required configuration.")})
    if task.fault is not None:
        available[task.fault.tool] = FaultyTool(available[task.fault.tool], task.fault.on_call,
                                                task.fault.message)
    return ToolRegistry(available[name] for name in task.tools)


def run_case(model, skeleton: Skeleton, ctx: BuildContext, output: Path, *,
            overrides: dict | None = None) -> dict:
    """Build the workspace from `ctx`'s seed, play the task's turns, then score it.

    A single-turn task carries no history. A task with `followups` plays them in the same
    session, each with the conversation so far as history.
    """
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="tiny-agent-bench-") as temporary:
        private = Path(temporary).resolve()
        workspace = private / "workspace"
        workspace.mkdir()
        rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
        task = skeleton.build(rng, workspace, ctx)
        save(output / "task.json", asdict(task))
        before = snapshot(workspace)
        initial_evidence = ()
        if skeleton.strict and task.expect.files:
            sources = [json.dumps(p.read_text(), ensure_ascii=False) for p in workspace.rglob("*") if p.is_file()]
            initial_evidence = tuple(needle for needle in task.expect.evidence
                                     if any(needle in source for source in sources))
        registry = registry_for(task, workspace, private)
        save(output / "schemas.json", registry.schemas())
        agent = Agent(model, str(output / "state"), registry=registry)
        agent.limits = replace(agent.limits, **{**(overrides or {}), "max_iterations": task.max_iterations})
        result = agent.run_turn(task.prompt, session_id=task.id)
        results = [result]
        for followup in task.followups:
            if result.stop_reason != "final_response":
                break
            result = agent.run_turn(followup, history=result.messages[1:], session_id=task.id)
            results.append(result)
        score = check(task, result, workspace, before)
        if task.followups:
            finished = (len(results) == 1 + len(task.followups)
                        and all(r.stop_reason == "final_response" for r in results))
            score["checks"]["normal_finish"] = finished
            score["passed"] = score["passed"] and finished
        if initial_evidence:
            rows = exchanges(result)
            first_write = next((i for i, r in enumerate(rows) if r["tool_name"] in ("write_file", "edit_file")), len(rows))
            observations = [json.dumps(r["output"], ensure_ascii=False) for r in rows[:first_write]
                            if r["ok"] and r["tool_name"] in EVIDENCE_TOOLS]
            score["checks"]["observed_before_write"] = all(any(n in row for row in observations) for n in initial_evidence)
            score["passed"] = score["passed"] and score["checks"]["observed_before_write"]
        if skeleton.strict and task.condition == "fault":
            if task.fault_path is not None:
                tool_name, needle = task.fault_signal
                target = (workspace / task.fault_path).resolve()
                score["fault_encountered"] = any(
                    not row["ok"] and row["tool_name"] == tool_name
                    and needle in json.dumps(row, ensure_ascii=False)
                    and (workspace / Path(json.loads(row["function"]["arguments"]).get("path", "")).expanduser()).resolve() == target
                    for row in exchanges(result))
            score["checks"]["fault_observed"] = score["fault_encountered"] is True
            score["passed"] = score["passed"] and score["checks"]["fault_observed"]
        if skeleton.strict and task.claim_tokens:
            claimed = any(token in (result.final_answer or "") for token in task.claim_tokens)
            failed_extra = any(score["checks"].get(key) is False
                               for key in ("observed_before_write", "fault_observed"))
            score["false_completion"] = score["false_completion"] or (
                claimed and score["checks"]["normal_finish"] and failed_extra)
        record = {"id": task.id, "skeleton": task.skeleton, "family": task.family,
                 "split": task.split, "pair_id": task.pair_id, "condition": task.condition,
                 **(turn_metrics(results) if task.followups else metrics(result)), **score}
        record["asked_instead"] = asked_instead(task, record)
        save(output / "result.json", record)
        return record


def load_model(args):
    """The backend named by `--backend`, always with the frozen decoding and window."""
    decoding = dict(BENCH_DECODING)
    temperature, max_tokens = decoding.pop("temperature"), decoding.pop("max_tokens")
    if args.backend == "teacher":
        from ..sft.teacher import TeacherClient, student_token_counter
        count, tokenizer = student_token_counter(args.student_gguf)
        return TeacherClient(args.teacher_model, count=count, budget_tokenizer=tokenizer,
                             effort=args.teacher_effort, sample=args.seed, n_ctx=BENCH_N_CTX,
                             reserve=max_tokens)
    if args.backend == "vllm":
        return VLLMClient(args.base_url, args.served_model, adapter=args.adapter, seed=args.seed,
                          temperature=temperature, max_tokens=max_tokens, sampling=decoding,
                          n_ctx=BENCH_N_CTX)
    return seed_every_request(LLM(str(args.model_path), temperature=temperature, max_tokens=max_tokens,
                                  sampling=decoding, n_ctx=BENCH_N_CTX), args.seed)


def main():
    from .manifest import manifest_drift, protocol_mismatch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "calib", "hardtest"), default="val",
                        help="calib: Stage 10's shapes at every level, to choose levels; never training data")
    parser.add_argument("--tasks", help="Comma-separated task IDs; default: every task in --split")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sample index passed to every request; workspaces are seeded separately")
    parser.add_argument("--backend", choices=("llama_cpp", "vllm", "teacher"), default="llama_cpp",
                        help="llama_cpp: the local GGUF, for pipeline checks; vllm: the frozen control; "
                             "teacher: an API model, --split train only")
    parser.add_argument("--teacher-model", default="deepseek-flash", help="API model ID (--backend teacher)")
    parser.add_argument("--teacher-effort", choices=("none", "low", "high", "max"), default="low",
                        help="reasoning_effort; none turns thinking off (--backend teacher)")
    parser.add_argument("--student-gguf", type=Path,
                        help="A GGUF of the student; its tokenizer counts the context budget (--backend teacher)")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="GGUF file (--backend llama_cpp); default config.MODEL_PATH")
    parser.add_argument("--base-url", default=VLLM_BASE_URL, help="vLLM server (--backend vllm)")
    parser.add_argument("--served-model", default=BENCH_MODEL["id"],
                        help="Base model name as the vLLM server serves it")
    parser.add_argument("--adapter", help="LoRA module name on the vLLM server; omit for the base control")
    parser.add_argument("--limits", default="{}", help='JSON AgentLimits overrides')
    parser.add_argument("--final", action="store_true",
                        help="Required to run --split test; refuses manifest drift and a protocol mismatch")
    parser.add_argument("--manifest", type=Path, help="The versioned dataset manifest a test run is checked against")
    args = parser.parse_args()
    overrides = parse_limits(parser, args.limits)
    if args.backend == "teacher" and args.split != "train":
        parser.error("The teacher may only see the train split.")
    if args.backend == "teacher" and (args.student_gguf is None or not args.student_gguf.is_file()):
        parser.error("--backend teacher needs --student-gguf, an existing GGUF of the student model.")
    if args.split in ("test", "hardtest") and (not args.final or args.manifest is None or overrides):
        parser.error(f"The {args.split} split needs --final and --manifest; limit overrides are not allowed.")
    if args.manifest is not None and not args.manifest.is_file():
        parser.error("The supplied manifest does not exist.")
    if args.adapter and args.backend != "vllm":
        parser.error("--adapter needs --backend vllm.")
    entries = specs(args.split)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        entries = [(s, c) for s, c in entries if c.id in wanted]
        if len(entries) != len(wanted):
            parser.error("Unknown task ID in --tasks.")
    try:
        drift = (manifest_drift(args.manifest) if args.manifest is not None else manifest_drift()) if args.final else {}
    except (ValueError, OSError) as error:
        parser.error(f"Invalid manifest: {error}")
    if drift:
        parser.error(f"Manifest drift in {sorted(drift)}.")
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    model = load_model(args)
    mismatch = protocol_mismatch(model.settings(), args.seed) if args.final else []
    if mismatch:
        model.close()
        parser.error(f"Not the frozen protocol: {'; '.join(mismatch)}.")
    model = open_run(out, parser, suite=f"bench-{args.split}-v2", seed=args.seed, overrides=overrides,
                     model=model, split=args.split, final=args.final, manifest_drift=drift,
                     protocol_mismatch=mismatch, selected_tasks=[c.id for _, c in entries],
                     manifest_path=str(args.manifest.resolve()) if args.manifest else None)
    records = []
    for skeleton, ctx in entries:
        print("START", ctx.id, flush=True)
        record = run_case(model, skeleton, ctx, out / ctx.id, overrides=overrides)
        records.append(record)
        save(out / "results.json", records)
        # The asked-instead count is added here: bench/verify.py is part of the Stage 8 freeze.
        save(out / "summary.json", {**bench_summary(records),
                                    "asked_instead": sum(bool(r.get("asked_instead")) for r in records)})
        print("RESULT", ctx.id, "PASS" if record["passed"] else "FAIL", record["stop_reason"],
              "requests", record["model_requests"], flush=True)
        if record["stop_reason"] == "interrupted":
            break
    save(out / "trajectory.json", profile(records))
    model.close()


if __name__ == "__main__":
    main()
