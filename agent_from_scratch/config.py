"""Model paths, decoding defaults and per-turn budgets.

Each value has one definition site here so that a run's settings can be snapshotted
into `TurnResult.settings` and frozen for evaluation.
"""

from dataclasses import dataclass
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

PROMPTS_DIR = _PACKAGE_DIR / "prompts"
CHAT_TEMPLATE_PATH = PROMPTS_DIR / "qwen_chat.jinja"
MODEL_PATH = (
    _PACKAGE_DIR.parent
    / "gz-data"
    / "hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f"
    / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
)

# Decoding defaults, matching what every entry point passes today.
TEMPERATURE = 0.0
MAX_TOKENS = 2048
N_CTX = 8000
N_GPU_LAYERS = -1

MAX_TOOL_CALLS_PER_RESPONSE = 8


@dataclass(frozen=True)
class AgentLimits:
    """Per-turn budgets recorded in `TurnResult.settings`; override with `replace`."""

    max_iterations: int = 20  # total model calls per turn, including summaries
    max_same_failures: int = 3
    max_tool_calls: int = 40  # max number of tool calls per turn
    max_tool_calls_per_response: int = MAX_TOOL_CALLS_PER_RESPONSE

    context_margin_tokens: int = 256  # token margin to reserve in the context window
    compact_headroom_tokens: int = 512  # try before the actor reaches the hard fit gate
    max_compact_calls: int = 4  # also charged against max_iterations; one call per attempt
    summary_max_tokens: int = 512  # summarizer output reserve, independent of the actor's
    max_summary_calls_per_attempt: int = 2  # one corrective retry at the same cut
