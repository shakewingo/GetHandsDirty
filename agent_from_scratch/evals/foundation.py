"""Small real-model dev suite. Run as a module; never point writable runs at source.

Example: python -m agent_from_scratch.evals.foundation --workspace outputs/fixture
         --output outputs/foundation --cases exact,guarded --seeds 11,22,33
Answer relevance is deliberately left for human review, separate from read coverage.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

from loguru import logger
from ..agent import Agent
from ..llm import LLM, _QWEN_TEMPLATE
from ..tools.base import ToolErrorCode, ToolResult
from ..tools.calculator import CalculatorTool
from ..tools.files import ListFilesTool, ReadFileTool, WriteFileTool
from ..tools.register import ToolRegistry


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def measure(result, original: bytes, relative_path: str, history_length: int) -> dict:
    observations = [json.loads(m["content"]) for m in result.messages[2 + history_length:]
                    if m["role"] == "tool"]
    chunks = [o["output"] for o in observations if o["ok"] and o["tool_name"] == "read_file"
              and o["output"]["path"] == relative_path]
    covered = bytearray(len(original))
    matches = True
    for chunk in chunks:
        data = chunk["content"].encode("utf-8")
        offset = chunk["offset"]
        matches &= original[offset:offset + len(data)] == data
        if 0 <= offset <= offset + len(data) <= len(original):
            covered[offset:offset + len(data)] = b"\1" * len(data)
    return {
        "run_id": result.run_id, "stop_reason": result.stop_reason,
        "completion_check": getattr(result, "completion_check", None),
        "model_requests": len(result.model_requests),
        "parse_errors": sum(q.status == "parse_error" for q in result.model_requests),
        "tool_errors": [o for o in observations if not o["ok"]],
        "tool_names": [o["tool_name"] for o in observations],
        "read_calls": len(chunks), "offsets": [c["offset"] for c in chunks],
        "covered_bytes": sum(covered), "target_bytes": len(original),
        "read_coverage_passed": bool(chunks) and all(covered) and matches and any(c["eof"] for c in chunks),
        "content_matches_fixture": matches,
        "partial_read_passed": bool(chunks) and matches and sum(covered) == min(1024, len(original))
        and all(covered[:1024]),
        "elapsed_seconds": result.elapsed_seconds, "error_message": result.error_message,
        "final_answer": result.final_answer, "answer_relevant": None,
        "usage": {key: sum((q.usage or {}).get(key) or 0 for q in result.model_requests)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "max_prompt_tokens": max(((q.usage or {}).get("prompt_tokens") or 0
                                  for q in result.model_requests), default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="llm.py")
    parser.add_argument("--cases", default="exact,guarded,guided,partial")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--read-bytes", type=int, default=8192)
    parser.add_argument("--readonly", action="store_true", help="Block writes; record every attempt.")
    args = parser.parse_args()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    target = (workspace / args.target).resolve()
    if not target.is_relative_to(workspace):
        parser.error("Target must be inside workspace.")
    if out.exists() and (out / "results.json").exists():
        parser.error("Use a fresh output directory to preserve earlier evidence.")
    out.mkdir(parents=True, exist_ok=True)
    original = target.read_bytes()
    cases = args.cases.split(",")
    if not set(cases) <= {"exact", "guarded", "guided", "partial", "history", "edit"}:
        parser.error("Unknown case.")
    if "edit" in cases and args.readonly:
        parser.error("edit requires a disposable writable workspace.")
    logger.remove()
    llm = LLM(temperature=0, max_tokens=2048, n_ctx=8000, chat_template_path=_QWEN_TEMPLATE)
    reader = ReadFileTool(workspace)
    # Before/after comparison without changing the protocol's 8192-byte ceiling.
    execute_read = reader.execute
    reader.execute = lambda path, offset=0, chunk_size=None: execute_read(
        path, offset, args.read_bytes if chunk_size is None else chunk_size)
    reader.parameters = {**reader.parameters, "properties": {
        **reader.parameters["properties"], "chunk_size": {
            **reader.parameters["properties"]["chunk_size"], "default": args.read_bytes,
            "description": f"Maximum bytes to read; defaults to {args.read_bytes} bytes.",
        },
    }}
    registry = ToolRegistry([CalculatorTool(), ListFilesTool(workspace), reader, WriteFileTool(workspace)])
    root = Path(__file__).resolve().parents[1]
    metadata = {"settings": llm.settings(), "schemas": registry.schemas(),
                "target": str(target), "target_sha256": digest(original),
                "runtime_hashes": {str(p.relative_to(root)): digest(p.read_bytes())
                                   for p in root.rglob("*.py") if "__pycache__" not in p.parts},
                "system_prompt": (root / "prompts/system.md").read_text(),
                "cases": cases, "seeds": args.seeds, "read_bytes": args.read_bytes,
                "readonly": args.readonly}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    backend = llm.llm.create_chat_completion
    summaries = []
    for case in cases:
        for seed in map(int, args.seeds.split(",")):
            name = f"{case}_{seed}"
            count = 0

            def observed_backend(**kwargs):
                nonlocal count
                count += 1
                started = monotonic()
                raw = backend(**kwargs, seed=seed)
                choice = raw["choices"][0]
                print(name, "request", count, "seconds", round(monotonic() - started, 1),
                      "finish", choice["finish_reason"],
                      "preview", repr(str(choice["message"].get("content"))[:120]), flush=True)
                return raw

            llm.llm.create_chat_completion = observed_backend
            agent = Agent(llm, state_dir=str(out / "state"), registry=registry)
            if args.readonly:
                invoke = agent.execute_tool

                def readonly(name, arguments, call_id=""):
                    if name == "write_file":
                        return ToolResult.failure(ToolErrorCode.EXECUTION_ERROR, call_id=call_id,
                                                  tool_name=name, detail="Read-only diagnostic.")
                    return invoke(name, arguments, call_id)

                agent.execute_tool = readonly
            prompt = f"Read file {target}"
            history = []
            check = None
            if case == "guarded":
                from ..verification import full_file_check
                check = full_file_check(workspace, target)
            elif case == "guided":
                prompt += " in full, continuing with next_offset until eof is true. Do not ask me questions; finish by briefly summarizing the file."
            elif case == "partial":
                prompt = f"Read only the first 1024 bytes of {target}, then briefly summarize that part."
            elif case == "history":
                history = [{"role": "user", "content": "Please keep reports concise."},
                           {"role": "assistant", "content": "I will keep reports concise."}]
            elif case == "edit":
                (workspace / "config.json").write_text('{"output": "old.txt", "retries": 3}\n')
                prompt = ("Read config.json, change only output to new.txt, save it, "
                          "then read it back to verify and report the result.")
            print("START", name, flush=True)
            before = {str(p.relative_to(workspace)): digest(p.read_bytes())
                      for p in workspace.rglob("*") if p.is_file()}
            options = {"completion_check": check} if check is not None else {}
            result = agent.run_turn(prompt, history, session_id=name, **options)
            summary = {"case": name, "prompt": prompt, "history_messages": len(history),
                       **measure(result, original, str(target.relative_to(workspace)), len(history))}
            summary["target_unchanged"] = target.read_bytes() == original
            after = {str(p.relative_to(workspace)): digest(p.read_bytes())
                     for p in workspace.rglob("*") if p.is_file()}
            summary["changed_files"] = sorted(p for p in before.keys() | after.keys()
                                               if before.get(p) != after.get(p))
            if case == "edit":
                actual = (workspace / "config.json").read_text()
                try:
                    summary["artifact_correct"] = json.loads(actual) == {"output": "new.txt", "retries": 3}
                except ValueError:
                    summary["artifact_correct"] = False
                summary["artifact"] = actual
                observations = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
                wrote = False
                summary["read_back_verified"] = False
                for observation in observations:
                    if not observation["ok"]:
                        continue
                    if observation["tool_name"] == "write_file" and observation["output"]["path"] == "config.json":
                        wrote = True
                    if wrote and observation["tool_name"] == "read_file" and observation["output"]["path"] == "config.json":
                        summary["read_back_verified"] = observation["output"]["content"] == actual
            summaries.append(summary)
            (out / "results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
            print("RESULT", name, summary["stop_reason"], "coverage", summary["covered_bytes"],
                  "/", len(original), "requests", summary["model_requests"], flush=True)
            if not summary["target_unchanged"]:
                raise RuntimeError("Fixture changed; stopping the comparison.")


if __name__ == "__main__":
    main()
