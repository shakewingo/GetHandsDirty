"""Model paths, decoding defaults and per-turn budgets.

Each value has one definition site here so that a run's settings can be snapshotted
into `TurnResult.settings` and frozen for evaluation.
"""

from dataclasses import dataclass
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

PROMPTS_DIR = _PACKAGE_DIR / "prompts"
CHAT_TEMPLATE_PATH = PROMPTS_DIR / "qwen_chat.jinja"
# local model path
MODEL_PATH = (
    _PACKAGE_DIR.parent
    / "gz-data"
    / "hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f"
    / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
)

# Decoding defaults, matching what every entry point passes today.
TEMPERATURE = 0.0
MAX_TOKENS = 2048
# Qwen2.5-7B's native window. The GGUF advertises 131072, which needs YaRN scaling
# that is not configured here, and would also allocate a 7 GiB KV cache up front.
N_CTX = 32768
N_GPU_LAYERS = -1

MAX_TOOL_CALLS_PER_RESPONSE = 8

# Stage 8 benchmark protocol, frozen in evals/bench/manifest.json. The control checkpoint is
# served by vLLM in bf16 so that a LoRA adapter can later be compared on the same server with
# only the adapter toggled. Sampling uses the model card's recommended values
# (generation_config.json at this revision); each task is run once per seed in BENCH_SEEDS.
BENCH_MODEL = {"id": "Qwen/Qwen3-4B-Instruct-2507",
               "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
               "dtype": "bfloat16", "backend": "vllm"}
BENCH_DECODING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                  "max_tokens": MAX_TOKENS}
BENCH_SEEDS = (0, 1, 2)
BENCH_N_CTX = 32768
VLLM_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class AgentLimits:
    """Per-turn budgets recorded in `TurnResult.settings`; override with `replace`."""

    max_iterations: int = 20  # total model calls per turn, including summaries
    max_same_failures: int = 3 # reminder ingestion after this many identical failed calls in a row
    stuck_reminder_calls: int = 3  # reminder ingestion after this many identical successful calls in a row
    max_tool_calls: int = 40  # max number of tool calls per turn
    max_tool_calls_per_response: int = MAX_TOOL_CALLS_PER_RESPONSE

    context_margin_tokens: int = 256  # token margin to reserve in the context window
    compact_ratio: float = 0.85  # hard-threshold in compact mechanism, summarize at this share of the usable window, before the hard fit gate
    max_compact_calls: int = 4  # also charged against max_iterations; one call per attempt
    summary_max_tokens: int = 512  # summarizer output reserve, independent of the actor's
    max_summary_calls_per_attempt: int = 2  # one corrective retry at the same cut
    elide_ratio: float | None = 0.6  # soft-threhold in compact mechanism, stub old tool outputs at this share of the usable window; None disables
    elide_min_chars: int = 400  # only outputs longer than this are elided
    planning: bool = False  # update_plan tool plus a per-request plan reminder
