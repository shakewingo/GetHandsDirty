"""The Stage 8 skeleton builders: one function pair per skeleton, registered in SKELETONS.

Each builder is deterministic in (skeleton name, seed, condition): `bench/run.py` seeds every
workspace from `random.Random(f"bench:{name}:{seed}:{condition}")` before calling it, so the
same triple always writes the same files. `solution()` returns the scripted LLM responses a
correct run would produce, in `evals.llm.LLMResponse` shape, for this module's own gate test
(`tests/test_bench_skeletons.py`) — nowhere else.
"""

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import random

from .spec import BuildContext, Task
from ...llm import LLMResponse, ResponseType
from ...tools.base import ToolCall


def call(name: str, **arguments) -> LLMResponse:
    return LLMResponse("assistant", "", ResponseType.tool_call, tool_calls=[ToolCall(name, arguments)])


def answer(text: str) -> LLMResponse:
    return LLMResponse("assistant", text, ResponseType.direct)


def _write(root: Path, name: str, content: str) -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _json(value) -> str:
    return json.dumps(value, indent=2) + "\n"


@dataclass(frozen=True)
class Skeleton:
    """A generator: `build` makes one Task from a seed and an optional clean/fault condition;
    `solution` scripts the LLM responses a correct run produces, used only by the gate test.
    """

    name: str
    family: str
    split: str
    seeds: tuple[int, ...]
    recovery: bool
    build: "Callable[[random.Random, Path, BuildContext], Task]"
    solution: "Callable[[Task], list[LLMResponse]]"


SKELETONS: list[Skeleton] = []
