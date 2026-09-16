"""Three real-model Stage 2A demonstrations; not the Stage 2B behavioral benchmark."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from loguru import logger
from ..agent import Agent
from ..examples.tools_demo import demo_registry
from ..llm import LLM
from ..session import SessionStore
from ..tools.web import WebFetchTool


URL = "https://docs.python.org/3/library/pathlib.html"
PROMPTS = {
    "files_shell": "Read config.json. Change only output to report.txt, save it, run the available configuration check, and report what the check actually returned.",
    "recovery": "Run the configuration check for config.json first. If it fails, inspect and fix the configuration, check it again, and report the result.",
    "web_file": f"Fetch {URL}. Save a short note in pathlib-note.txt explaining the difference between pure and concrete paths, with the source URL. Tell me if the fetched page was truncated.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new evidence directory")
    parser.add_argument("--conversation", action="store_true", help="User-steered tool turns; separate from autonomous tasks")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logger.remove()
    model = LLM()
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "settings": model.settings(), "prompts": PROMPTS,
        "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob("*.py") if "__pycache__" not in p.parts},
        "system_prompt": (root / "prompts/system.md").read_text(),
        "scope": "Live tool/loop smoke, not a capability success-rate estimate",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    # Keep direct TLS evidence separate from the model's decision to fetch.
    fetched = WebFetchTool({"docs.python.org"}).invoke({"url": URL}, "live-tls-smoke")
    (out / "live-fetch.json").write_text(json.dumps(asdict(fetched), ensure_ascii=False, indent=2))
    if args.conversation:
        run_conversation(model, out)
        return
    results = []
    for name, prompt in PROMPTS.items():
        workspace = out / name / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "config.json").write_text('{"output":"old.txt","retries":3}\n')
        (workspace / "untouched.txt").write_text("keep this unchanged\n")
        registry = demo_registry(workspace, {"docs.python.org"})
        (out / name / "schemas.json").write_text(json.dumps(registry.schemas(), indent=2))
        agent = Agent(model, str(out / name / "state"), registry=registry)
        agent.limits = replace(agent.limits, max_iterations=12)
        print(f"START {name}", flush=True)
        result = agent.run_turn(prompt, session_id=name)
        observations = [json.loads(m["content"] or "") for m in result.messages if m["role"] == "tool"]
        expected = {"output": "report.txt", "retries": 3}
        try:
            config_correct = json.loads((workspace / "config.json").read_text()) == expected
        except (ValueError, OSError):
            config_correct = False
        checks = [o for o in observations if o["tool_name"] == "shell"
                  and (o.get("output") or {}).get("command_id") == "check_fixture"]
        note = workspace / "pathlib-note.txt"
        note_text = note.read_text() if note.is_file() else ""
        summary = {
            "case": name, "run_id": result.run_id, "stop_reason": result.stop_reason,
            "model_requests": len(result.model_requests), "elapsed_seconds": result.elapsed_seconds,
            "tool_names": [o["tool_name"] for o in observations],
            "tool_errors": [o for o in observations if not o["ok"]],
            "config_correct": config_correct,
            "check_passed": bool(checks) and checks[-1]["ok"],
            "recovered_failed_check": bool(checks) and not checks[0]["ok"] and checks[-1]["ok"],
            "note_has_source_and_terms": URL in note_text and "pure" in note_text.lower() and "concrete" in note_text.lower(),
            "web_fetch_succeeded": any(o["tool_name"] == "web_fetch" and o["ok"] for o in observations),
            "untouched_preserved": (workspace / "untouched.txt").read_text() == "keep this unchanged\n",
            "final_answer": result.final_answer,
            "usage": {key: sum((q.usage or {}).get(key) or 0 for q in result.model_requests)
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        }
        # Term/source checks are only artifact smoke; review the actual note and answer too.
        results.append(summary)
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def run_conversation(model, out):
    """Explicit user turns diagnose tool usability; they do not establish autonomous completion."""
    workspace = out / "workspace"
    workspace.mkdir()
    (workspace / "config.json").write_text('{"output":"old.txt","retries":3}\n')
    registry = demo_registry(workspace, {"docs.python.org"})
    state = out / "state"
    store = SessionStore(state / "sessions")
    prompts = [
        "Read config.json.",
        "Change output to report.txt in config.json. Preserve the other values.",
        "Run check_fixture and report its result.",
        f"Fetch {URL} and briefly explain pure versus concrete paths.",
        "Save that explanation and its source URL in pathlib-note.txt.",
    ]
    records = []
    for prompt in prompts:
        print("TURN", prompt, flush=True)
        # A fresh Agent reloads disk history between turns; the model instance is reused.
        agent = Agent(model, str(state), registry=registry)
        agent.limits = replace(agent.limits, max_iterations=8)
        result = agent.run_turn(prompt, store.load_history("conversation"), session_id="conversation")
        record = {"prompt": prompt, "run_id": result.run_id, "stop_reason": result.stop_reason,
                  "model_requests": len(result.model_requests), "final_answer": result.final_answer}
        records.append(record)
        (out / "conversation.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (out / "artifacts.json").write_text(json.dumps({p.name: p.read_text() for p in workspace.iterdir()
                                                   if p.is_file()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
