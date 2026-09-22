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


def _build_single_field_edit(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    retries = rng.randint(1, 5)
    new_output = rng.choice(["report.json", "summary.json", "result.json"])
    config = {"output": rng.choice(["draft.json", "old.json", "pending.json"]),
             "retries": retries, "format": "json"}
    _write(root, "config.json", _json(config))
    expected = {**config, "output": new_output}
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=(f"Read config.json and change only its output field to {new_output!r}. "
                      "Keep every other field exactly as it is, save the file, then reply DONE."),
               tools=("read_file", "write_file"),
               expect=Expect(answer=Answer(value="DONE", format="required"),
                            files={"config.json": _json(expected)}),
               claim_tokens=("DONE",))


def _solution_single_field_edit(task: Task) -> list:
    expected = json.loads(task.expect.files["config.json"])
    return [call("read_file", path="config.json"),
           call("write_file", path="config.json", content=json.dumps(expected)),
           answer("DONE")]


def _build_check_fix_recheck(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    correct = {"output": "report.json", "retries": rng.randint(1, 5), "enabled": True}
    if ctx.condition == "fault":
        _write(root, "config.json", _json({**correct, "output": "old.json"}))
        process = ("check_before_write", "passing_check_after_last_write")
    else:
        _write(root, "config.json", _json(correct))
        process = ("check_before_write", "no_write_attempts")
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=8,
               prompt=("Run check_fixture before making any change. If it fails because "
                      "config.json's output field is wrong, fix only that field and run "
                      "check_fixture again. If it already passes, do not rewrite the file. "
                      "Reply CHECKED only after a passing check."),
               tools=("read_file", "write_file", "shell"),
               expect=Expect(answer=Answer(value="CHECKED", format="required"),
                            files={"config.json": _json(correct)}, process=process),
               claim_tokens=("CHECKED",), fault_signal=("shell", "FAIL"),
               debug={"correct": correct})


def _solution_check_fix_recheck(task: Task) -> list:
    correct = task.debug["correct"]
    if task.condition == "fault":
        return [call("shell", command_id="check_fixture"),
               call("read_file", path="config.json"),
               call("write_file", path="config.json", content=json.dumps(correct)),
               call("shell", command_id="check_fixture"), answer("CHECKED")]
    return [call("shell", command_id="check_fixture"), answer("CHECKED")]


def _build_no_op_correct_config(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    config = {"output": "report.json", "retries": rng.randint(1, 5)}
    _write(root, "config.json", _json(config))
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=5,
               prompt=("Inspect config.json. Its output should be report.json. If it already "
                      "is, reply only UNCHANGED and do not call write_file. Otherwise correct "
                      "it and reply UPDATED."),
               tools=("read_file", "write_file"),
               expect=Expect(answer=Answer(value="UNCHANGED", format="required"),
                            evidence=("report.json",), process=("no_write_attempts",)))


def _solution_no_op_correct_config(task: Task) -> list:
    return [call("read_file", path="config.json"), answer("UNCHANGED")]


def _build_deep_chain_lookup(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    depth = rng.randint(3, 5)
    names = [f"node_{i}_{rng.choice('abcdefgh')}.json" for i in range(depth + 1)]
    final_value = f"VALUE-{rng.randint(1000, 9999)}"
    for i in range(depth):
        decoy = f"node_{i}_{rng.choice('xyzqrs')}.json"
        _write(root, decoy, _json({"note": "not the path to follow"}))
        _write(root, names[i], _json({"next": names[i + 1]}))
    _write(root, names[depth], _json({"value": final_value}))
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=depth + 3,
               prompt=(f"Start at {names[0]}. Each file names the next one under its 'next' "
                      "field; some other files exist but are not part of the chain. Follow "
                      "the chain until a file has a 'value' field instead, then report only "
                      "that value."),
               tools=("list_files", "read_file"),
               expect=Expect(answer=Answer(value=final_value), evidence=(final_value,),
                            process=("no_write_attempts",)),
               debug={"chain": names})


def _solution_deep_chain_lookup(task: Task) -> list:
    return [call("read_file", path=name) for name in task.debug["chain"]] + [
        answer(task.expect.answer.value)]


SERVICES = ("billing", "search", "auth", "ingest")
TEAMS = ("atlas", "vega", "orion", "lyra")


def _build_grep_locate(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    service = rng.choice(SERVICES)
    owner = f"team-{rng.choice(TEAMS)}"
    count = rng.randint(8, 14)
    target = rng.randrange(count)
    for i in range(count):
        path = f"services/group_{i // 4}/svc_{i}.conf"
        if i == target:
            _write(root, path, f"service = {service}\nowner[{service}] = {owner}\n")
        else:
            _write(root, path, f"service = {rng.choice(SERVICES)}\n"
                               f"owner[other] = team-{rng.choice(TEAMS)}\n")
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=(f"Somewhere under services/ a config line reads 'owner[{service}] = "
                      "<name>'. Find it and report only <name>."),
               tools=("grep_text", "read_file"),
               expect=Expect(answer=Answer(value=owner), evidence=(owner,),
                            process=("no_write_attempts",)),
               debug={"service": service, "path": f"services/group_{target // 4}/svc_{target}.conf"})


def _solution_grep_locate(task: Task) -> list:
    return [call("grep_text", query=f"owner\\[{task.debug['service']}\\]"),
           call("read_file", path=task.debug["path"]), answer(task.expect.answer.value)]


def _build_sum_across_files(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    count = rng.randint(3, 5)
    amounts = [rng.randint(10, 500) for _ in range(count)]
    for i, amount in enumerate(amounts):
        _write(root, f"invoices/inv_{i:02d}.json", _json({"amount": amount, "status": "final"}))
    for i in range(rng.randint(1, 2)):
        _write(root, f"invoices/void_{i:02d}.json", _json({"amount": rng.randint(10, 500), "status": "void"}))
    total = sum(amounts)
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=count + 3,
               prompt=("Under invoices/, files named inv_NN.json count; files named "
                      "void_NN.json do not. Read every inv_NN.json file and reply with only "
                      "the total of their amount fields, as a plain integer."),
               tools=("list_files", "read_file", "calculator"),
               expect=Expect(answer=Answer(value=str(total)),
                            evidence=tuple(str(a) for a in amounts), process=("no_write_attempts",)),
               debug={"count": count})


def _solution_sum_across_files(task: Task) -> list:
    n = task.debug["count"]
    return [call("read_file", path=f"invoices/inv_{i:02d}.json") for i in range(n)] + [
        answer(task.expect.answer.value)]


def _build_pointer_nested_edit(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    active, spare = ("settings_a", "settings_b") if rng.random() < 0.5 else ("settings_b", "settings_a")
    _write(root, "manifest.json", _json({"active": active}))
    base = {"name": active, "output": {"filename": "old.json", "format": "json"}, "enabled": True}
    _write(root, f"{active}.json", _json(base))
    _write(root, f"{spare}.json", _json({"name": spare,
        "output": {"filename": "keep.json", "format": "json"}, "enabled": True}))
    new_filename = "analysis.json"
    expected = {**base, "output": {**base["output"], "filename": new_filename}}
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=(f"Read manifest.json to find the active settings file, then change "
                      f"only its output.filename field to {new_filename!r}, keeping every "
                      "other field. Do not touch the other settings file. Reply DONE."),
               tools=("read_file", "write_file"),
               expect=Expect(answer=Answer(value="DONE", format="required"),
                            files={f"{active}.json": _json(expected)}, evidence=(active,)),
               claim_tokens=("DONE",), debug={"active": active})


def _solution_pointer_nested_edit(task: Task) -> list:
    active = task.debug["active"]
    expected = json.loads(task.expect.files[f"{active}.json"])
    return [call("read_file", path="manifest.json"), call("read_file", path=f"{active}.json"),
           call("write_file", path=f"{active}.json", content=json.dumps(expected)), answer("DONE")]


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

SKELETONS.append(Skeleton(
    name="single_field_edit", family="updates", split="dev", seeds=(0, 1, 2), recovery=False,
    build=_build_single_field_edit, solution=_solution_single_field_edit))

SKELETONS.append(Skeleton(
    name="check_fix_recheck", family="recovery", split="dev", seeds=(0, 1, 2), recovery=True,
    build=_build_check_fix_recheck, solution=_solution_check_fix_recheck))

SKELETONS.append(Skeleton(
    name="no_op_correct_config", family="stopping", split="dev", seeds=(0, 1, 2), recovery=False,
    build=_build_no_op_correct_config, solution=_solution_no_op_correct_config))

SKELETONS.append(Skeleton(name="deep_chain_lookup", family="inspection", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_deep_chain_lookup, solution=_solution_deep_chain_lookup))

SKELETONS.append(Skeleton(name="grep_locate", family="inspection", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_grep_locate, solution=_solution_grep_locate))

SKELETONS.append(Skeleton(name="sum_across_files", family="inspection", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_sum_across_files, solution=_solution_sum_across_files))

SKELETONS.append(Skeleton(name="pointer_nested_edit", family="updates", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_pointer_nested_edit, solution=_solution_pointer_nested_edit))
