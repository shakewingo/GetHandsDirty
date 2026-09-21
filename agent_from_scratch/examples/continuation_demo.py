"""Compact a pressured turn, checkpoint it, then continue after a simulated restart.

Run: python -m agent_from_scratch.examples.continuation_demo --output outputs/continuation-demo
One session directory carries four cases: a first turn that compacts and checkpoints, a
restart that replays summary + uncovered suffix, the same restart against full raw history
as a control, and a restart whose checkpoint no longer matches an edited history.
The fixture is synthetic; no workspace tools or file edits are exposed to the model.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..agent import Agent
from ..context import InstructionConfig, load_instructions
from ..llm import LLM
from ..session import SessionStore
from ..tools.base import ToolRegistry

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


def pressured_history(model: LLM, instructions: str,
                      request: str) -> list[ChatCompletionRequestMessage]:
    """Grow repetitive filler until the next request sits in the early-compaction band."""
    history: list[ChatCompletionRequestMessage] = [
        {"role": "user", "content": "Remember: project code is CEDAR-42; limit is 6 kg."}]
    for count in range(1, 200):
        history[1:] = [{"role": "assistant", "content": "Acknowledged.\n" + "\n".join(
            f"Archived note {i}: routine inspection completed; no further action requested."
            for i in range(count))}]
        budget = model.measure_context(
            [{"role": "system", "content": instructions}, *history,
             {"role": "user", "content": request}], {})
        if 256 <= budget["remaining_tokens"] < 768:
            return history
    raise RuntimeError("Could not construct the measured pressure fixture.")


def actor_prompt_tokens(result) -> list[int | None]:
    """Prompt sizes of the actor's own requests, excluding summary calls."""
    return [(q.budget or {}).get("prompt_tokens") for q in result.model_requests
            if q.purpose == "agent"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = LLM(n_ctx=4096, max_tokens=512)
    try:
        instructions, _ = load_instructions(InstructionConfig())
        first = ("Correction: the limit is now 7 kg. "
                 "Reply only with the project code and corrected limit.")
        state_dir = str(args.output / "state")
        store = SessionStore(Path(state_dir, "sessions"))
        # A checkpoint may only cover messages the session holds, so seed it like a REPL would.
        store.append("continuation", "seed", pressured_history(model, instructions, first))
        report = {"cases": {}}

        opening = Agent(model, state_dir, registry=ToolRegistry([])).run_turn(
            first, store.load_history("continuation"), session_id="continuation")
        report["cases"]["first_turn"] = asdict(opening)

        # A fresh store and agent stand in for a restarted process.
        reopened = SessionStore(Path(state_dir, "sessions"))
        saved = reopened.load_history("continuation")
        checkpoint = reopened.load_checkpoint("continuation", saved)
        follow_up = "Without rereading anything, state the project code and the limit again."

        replayed = Agent(model, state_dir, registry=ToolRegistry([])).run_turn(
            follow_up, saved, session_id="continuation", checkpoint=checkpoint)
        report["cases"]["restart_with_checkpoint"] = asdict(replayed)

        control = Agent(model, None, registry=ToolRegistry([])).run_turn(follow_up, saved)
        report["cases"]["restart_raw_control"] = asdict(control)

        edited: list[ChatCompletionRequestMessage] = [
            {"role": "user", "content": "EDITED"}, *saved[1:]]
        report["cases"]["stale_checkpoint_ignored"] = {
            "checkpoint_found": reopened.load_checkpoint("continuation", edited) is not None}

        report["comparison"] = {
            "checkpoint_covered": None if checkpoint is None else checkpoint["covered"],
            "saved_messages": len(saved),
            "first_turn_purposes": [q.purpose for q in opening.model_requests],
            "first_turn_answer": opening.final_answer,
            "restart_with_checkpoint_prompt_tokens": actor_prompt_tokens(replayed),
            "restart_raw_prompt_tokens": actor_prompt_tokens(control),
            "checkpoint_answer": replayed.final_answer,
            "raw_answer": control.final_answer,
        }
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report["comparison"], ensure_ascii=False, indent=2), flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
