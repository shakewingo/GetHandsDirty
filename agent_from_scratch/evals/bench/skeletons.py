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

from .spec import Answer, BuildContext, Expect, Task
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


def _build_pointer_lookup(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    profiles = [f"profile_{c}" for c in "abcdefgh"]
    rng.shuffle(profiles)
    active, decoys = profiles[0], profiles[1:1 + rng.randint(2, 3)]
    outputs = {name: f"{name}-report.json" for name in [active, *decoys]}
    _write(root, "manifest.json", _json({"active_profile": active}))
    for name in [active, *decoys]:
        _write(root, f"profiles/{name}.json", _json({"output": outputs[name]}))
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=("Use manifest.json to find the active profile, then read that profile "
                      "file under profiles/ and report only its output filename. Do not "
                      "change any files."),
               tools=("list_files", "read_file"),
               expect=Expect(answer=Answer(value=outputs[active]),
                            evidence=(active, outputs[active]), process=("no_write_attempts",)),
               debug={"active_profile": active})


def _solution_pointer_lookup(task: Task) -> list:
    active = task.debug["active_profile"]
    return [call("read_file", path="manifest.json"),
           call("read_file", path=f"profiles/{active}.json"),
           answer(task.expect.answer.value)]


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

SKELETONS.append(Skeleton(
    name="pointer_lookup", family="inspection", split="dev", seeds=(0, 1, 2), recovery=False,
    build=_build_pointer_lookup, solution=_solution_pointer_lookup))
