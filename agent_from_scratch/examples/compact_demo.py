"""Compare full/automatic/manual context with local Qwen, plus a deliberate failure.

Run: python -m agent_from_scratch.examples.compact_demo --output outputs/compact-demo
The fixture is synthetic; no workspace tools or file edits are exposed to the model.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from ..agent import Agent
from ..config import AgentLimits
from ..context import InstructionConfig, load_instructions
from ..llm import LLM
from ..tools.base import ToolRegistry

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = LLM(n_ctx=4096, max_tokens=512)
    try:
        instructions, _ = load_instructions(InstructionConfig())
        request = "Correction: the limit is now 7 kg. Reply only with the project code and corrected limit."
        history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Remember: project code is CEDAR-42; limit is 6 kg."}]
        # Place the fixture in the default early-compaction band using actual token counts.
        for count in range(1, 200):
            history[1:] = [{"role": "assistant", "content": "Acknowledged.\n" + "\n".join(
                f"Archived note {i}: routine inspection completed; no further action requested."
                for i in range(count))}]
            budget = model.measure_context([
                {"role": "system", "content": instructions}, *history,
                {"role": "user", "content": request}], {})
            if 256 <= budget["remaining_tokens"] < 768:
                break
        else:
            raise RuntimeError("Could not construct the measured pressure fixture.")
        report = {"initial_budget": budget, "limits": asdict(AgentLimits()), "cases": {}}
        for name in ("full", "automatic", "manual", "oversized"):
            limits = replace(AgentLimits(), max_compact_calls=0) if name == "full" else AgentLimits()
            agent = Agent(model, str(args.output / name), registry=ToolRegistry([]), limits=limits)
            result = agent.run_turn("token " * 6000 if name == "oversized" else request,
                                    [] if name == "oversized" else history, compact=name == "manual")
            report["cases"][name] = asdict(result)
            print(name, result.stop_reason, repr(result.final_answer),
                  [(q.purpose, q.status, (q.budget or {}).get("prompt_tokens"), q.error_message)
                   for q in result.model_requests], flush=True)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
