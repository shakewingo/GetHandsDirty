# Stage 8 generated benchmark implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task by task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build the Stage 8 generated benchmark: 14 procedurally generated task skeletons (4
dev, 10 test × 6 variants = 60) on the current tool registry, scored by one declarative
verifier, frozen with a manifest that gates the test split.

**Architecture:** a new `evals/bench/` subpackage. `spec.py` holds the declarative shapes
(`Task`, `Expect`, `Answer`, `Fault`, `BuildContext`); `skeletons.py` holds one pure builder
function per skeleton plus a scripted `solution()`; `verify.py` holds the one verifier
(`check()`); `faults.py` wraps a tool to fail its k-th call; `run.py` builds tasks from seeds,
runs them through the ordinary `Agent`, and scores them; `manifest.py` freezes everything that
must not move once the test split runs. Every runner reuses the existing `open_run`,
`parse_limits`, `metrics()`, `profile()` and `compare` unchanged.

**Tech stack:** Python standard library, the existing `agent_from_scratch` package
(`Agent`, `ToolRegistry`, the file/search/shell/calculator tools), `unittest`. No new
dependencies.

**Spec:** [STAGE8_DESIGN.md](STAGE8_DESIGN.md) (approved). This plan implements it in full;
read the design for the *why* behind each shape.

## Global Constraints

- Content-first scoring: `passed` needs correct files, kept-untouched files, evidence, process
  rules and the answer value present as a whole token; exact format (`format_exact`) is
  reported separately and only folded into `passed` when `Answer.format == "required"`.
- Any valid tool path passes: the verifier checks final state and observed evidence, never a
  specific call sequence.
- Raw file content and workspace generation are the only mutable state; `Task` instances are
  frozen dataclasses.
- 10 test skeletons × 6 variants = 60; 4 dev skeletons = 15 tasks (recovery pairs count as 2).
  Listed in `evals/bench/splits.json` before implementation of any test skeleton continues.
- `--split test` requires `--final` and refuses on manifest drift unless `--allow-drift` is
  passed (recorded either way).
- Memory is off; no memory tool is ever offered by `registry_for`.
- Web tasks are out of scope for this plan (see the design's "Out of scope").
- Coding rules from `CLAUDE.md`/`STAGE.md`: standard library only, no helper abstractions for
  one-time operations, Google-style docstrings for larger functions, comments only where logic
  is not self-evident.
- Test command, from the repository root:
  `python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3`
- Commits: conventional prefixes (`feat:`, `test:`, `docs:`); `docs/` is git-ignored but
  tracked, so new doc files need `git add -f`.
- Every task's test run must leave the full suite green before moving to the next task.

---

## File map

| File | Change |
|---|---|
| `evals/bench/__init__.py` | new, empty |
| `evals/bench/spec.py` | new: `Task`, `Expect`, `Answer`, `Fault`, `BuildContext` |
| `evals/bench/faults.py` | new: `FaultyTool` |
| `evals/bench/verify.py` | new: `check()`, `bench_summary()` |
| `evals/bench/skeletons.py` | new: `Skeleton`, `SKELETONS`, 14 builder/solution pairs |
| `evals/bench/splits.json` | new: dev/test/train skeleton names |
| `evals/bench/run.py` | new: `specs()`, `registry_for()`, `run_case()`, `main()` |
| `evals/bench/manifest.py` | new: `build_manifest()`, `save_manifest()`, `manifest_drift()` |
| `evals/__main__.py` | modify: add `bench`, `freeze` commands |
| `evals/README.md` | modify: document `bench`/`freeze` |
| `docs/STAGE.md` | modify: defer Stage 4A–4B, tick satisfied Stage 8 boxes |
| `docs/benchmark.md` | new: dev pilot and test-run results |
| `docs/context-memory.md` | modify: retained/lost facts from the real-model demos |
| `tests/test_bench_spec.py` | new |
| `tests/test_bench_faults.py` | new |
| `tests/test_bench_verify.py` | new |
| `tests/test_bench_run.py` | new |
| `tests/test_bench_skeletons.py` | new |
| `tests/test_bench_manifest.py` | new |

---

## Task 1: Declarative shapes (`spec.py`)

**Files:**
- Create: `evals/bench/spec.py`
- Test: `tests/test_bench_spec.py`

**Interfaces — produces:**
- `Fault(tool: str, on_call: int, message: str)`
- `Answer(value: str, format: str = "advisory", reject: tuple[str, ...] = ())`
- `Expect(answer: Answer | None = None, files: dict[str, str] = {}, may_change: tuple[str, ...] = (), evidence: tuple[str, ...] = (), process: tuple[str, ...] = ())`
- `Task(id, skeleton, family, split, prompt, tools, expect, max_iterations=12, fault=None, fault_signal=None, claim_tokens=(), pair_id=None, condition=None, debug={})`
- `BuildContext(name, family, split, seed, condition)` with computed properties `.id` and `.pair_id`

- [ ] **Step 1: write the failing test**

```python
"""Declarative task shapes are frozen and JSON-serializable."""

from dataclasses import FrozenInstanceError, asdict
import json
import unittest

from agent_from_scratch.evals.bench.spec import Answer, BuildContext, Expect, Fault, Task


class SpecTests(unittest.TestCase):
    def test_answer_rejects_an_unknown_format(self):
        Answer(value="X")  # default "advisory" is fine
        with self.assertRaises(ValueError):
            Answer(value="X", format="exact")

    def test_task_is_frozen_and_asdict_is_json_serializable(self):
        task = Task(id="t-0", skeleton="t", family="inspection", split="dev",
                    prompt="p", tools=("read_file",), expect=Expect(answer=Answer(value="X")))
        with self.assertRaises(FrozenInstanceError):
            task.id = "other"
        json.dumps(asdict(task))  # must not raise

    def test_build_context_computes_id_and_pair_id(self):
        plain = BuildContext(name="s", family="f", split="dev", seed=2, condition=None)
        self.assertEqual((plain.id, plain.pair_id), ("s-2", None))
        paired = BuildContext(name="s", family="f", split="dev", seed=2, condition="fault")
        self.assertEqual((paired.id, paired.pair_id), ("s-2-fault", "s-2"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_spec -q`; expect
  `ModuleNotFoundError: No module named 'agent_from_scratch.evals.bench'`.

- [ ] **Step 3: implementation.** Create `evals/bench/__init__.py` (empty). Create
  `evals/bench/spec.py`:

```python
"""Declarative shapes for the Stage 8 generated benchmark: a Task's contract, not its content.

A skeleton builder returns a Task; `bench/verify.py` reads only `Task.expect` to score a run.
`debug` carries values a skeleton's own `solution()` needs to script a correct run (e.g. which
of several generated names is the answer); the verifier never reads it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fault:
    """Make the tool named `tool` fail its `on_call`-th invocation (1-based) with `message`."""

    tool: str
    on_call: int
    message: str


@dataclass(frozen=True)
class Answer:
    """What a passing final reply must contain.

    `format`: "advisory" means `value` only has to appear in the reply as a whole token (an
    extracted fact may be wrapped in a sentence); "required" means the stripped reply must
    equal `value` exactly, for tasks whose deliverable IS the reply token.
    """

    value: str
    format: str = "advisory"
    reject: tuple[str, ...] = ()

    def __post_init__(self):
        if self.format not in ("advisory", "required"):
            raise ValueError("Answer.format must be 'advisory' or 'required'.")


@dataclass(frozen=True)
class Expect:
    """Everything a passing run must satisfy; each field is checked and reported independently."""

    answer: Answer | None = None
    files: dict[str, str] = field(default_factory=dict)  # path -> expected final content
    may_change: tuple[str, ...] = ()  # paths allowed to change beyond `files`
    evidence: tuple[str, ...] = ()  # each must appear in some successful observation's output
    process: tuple[str, ...] = ()  # rule names; see bench/verify.py PROCESS_RULES


@dataclass(frozen=True)
class Task:
    """One generated benchmark task, already carrying its own workspace's expected shape."""

    id: str
    skeleton: str
    family: str
    split: str
    prompt: str
    tools: tuple[str, ...]
    expect: Expect
    max_iterations: int = 12
    fault: Fault | None = None
    fault_signal: tuple[str, str] | None = None  # (tool, substring to find in a failed row)
    claim_tokens: tuple[str, ...] = ()  # replies that claim completion, for false_completion
    pair_id: str | None = None
    condition: str | None = None  # "clean" | "fault" | None
    debug: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BuildContext:
    """Identity for one (skeleton, seed, condition) triple, computed before any file is written."""

    name: str
    family: str
    split: str
    seed: int
    condition: str | None

    @property
    def id(self) -> str:
        return f"{self.name}-{self.seed}" if self.condition is None else f"{self.name}-{self.seed}-{self.condition}"

    @property
    def pair_id(self) -> str | None:
        return f"{self.name}-{self.seed}" if self.condition is not None else None
```

- [ ] **Step 4:** rerun; expect OK, 3 tests.
- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/__init__.py agent_from_scratch/evals/bench/spec.py \
           agent_from_scratch/tests/test_bench_spec.py
git commit -m "feat: add the Stage 8 benchmark's declarative task shapes"
```

---

## Task 2: Fault injection (`faults.py`)

**Files:**
- Create: `evals/bench/faults.py`
- Test: `tests/test_bench_faults.py`

**Interfaces — consumes:** `tools.base.Tool`, `ToolExecutionError`, `ToolErrorCode`.
**Produces:** `FaultyTool(wrapped: Tool, on_call: int, message: str)`, a `Tool`.

- [ ] **Step 1: write the failing test**

```python
"""FaultyTool fails exactly its configured call and delegates every other one."""

import unittest

from agent_from_scratch.evals.bench.faults import FaultyTool
from agent_from_scratch.tools.calculator import CalculatorTool


class FaultyToolTests(unittest.TestCase):
    def test_fails_only_the_configured_call(self):
        tool = FaultyTool(CalculatorTool(), on_call=2, message="Simulated failure.")
        first = tool.invoke({"operation": "add", "left": 1, "right": 1})
        second = tool.invoke({"operation": "add", "left": 1, "right": 1})
        third = tool.invoke({"operation": "add", "left": 1, "right": 1})
        self.assertEqual((first.ok, second.ok, third.ok), (True, False, True))
        self.assertIn("Simulated failure.", second.error_message)
        self.assertEqual(second.error_code, "execution_error")
        self.assertEqual((first.output, third.output), (2.0, 2.0))

    def test_copies_the_wrapped_tools_schema(self):
        wrapped = CalculatorTool()
        tool = FaultyTool(wrapped, on_call=1, message="x")
        self.assertEqual((tool.name, tool.description, tool.parameters),
                         (wrapped.name, wrapped.description, wrapped.parameters))

    def test_rejects_a_non_positive_call_number(self):
        with self.assertRaises(ValueError):
            FaultyTool(CalculatorTool(), on_call=0, message="x")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_faults -q`;
  expect `ModuleNotFoundError`.

- [ ] **Step 3: implementation.** Create `evals/bench/faults.py`:

```python
"""Eval-only tool wrapper that fails a wrapped tool's k-th call; never used by the runtime."""

from ...tools.base import Tool, ToolErrorCode, ToolExecutionError


class FaultyTool(Tool):
    """Delegate to `wrapped`, except call number `on_call` (1-based, across all its calls)."""

    def __init__(self, wrapped: Tool, on_call: int, message: str):
        if on_call < 1:
            raise ValueError("on_call must be a 1-based call number.")
        self.wrapped, self.on_call, self.message = wrapped, on_call, message
        self.name, self.description, self.parameters = wrapped.name, wrapped.description, wrapped.parameters
        self.calls = 0

    def execute(self, **kwargs):
        self.calls += 1
        if self.calls == self.on_call:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, self.message)
        return self.wrapped.execute(**kwargs)
```

- [ ] **Step 4:** rerun; expect OK, 3 tests. Run the full suite; expect all green.
- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/faults.py agent_from_scratch/tests/test_bench_faults.py
git commit -m "feat: add FaultyTool, an eval-only k-th-call failure wrapper"
```

---

## Task 3: The verifier (`verify.py`)

**Files:**
- Create: `evals/bench/verify.py`
- Test: `tests/test_bench_verify.py`

**Interfaces — consumes:** `spec.Task`, `spec.Expect`, `spec.Answer`; `evals.verify.exchanges`,
`evals.verify.snapshot`. **Produces:** `check(task: Task, result, workspace: Path, before: dict) -> dict`
with keys `passed: bool`, `checks: dict[str, bool]`, `changed_paths: list[str]`,
`false_completion: bool | None`, `fault_encountered: bool | None`; `bench_summary(records: list[dict]) -> dict`.

- [ ] **Step 1: write the failing tests**

```python
"""The Stage 8 verifier: content-first scoring, automatic false-completion and fault checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.bench.spec import Answer, Expect, Task
from agent_from_scratch.evals.bench.verify import bench_summary, check
from agent_from_scratch.tools.base import ToolResult
from agent_from_scratch.trace import RunStopReason, TurnResult


def _row(name, output, *, ok=True, error_message=None, arguments=None, call_id="c1"):
    """One assistant tool-call message plus its tool-result message, in production shape."""
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments or {})}}]}
    result = ToolResult(call_id=call_id, tool_name=name, ok=ok, output=output,
                        error_code=None if ok else "execution_error",
                        error_message=error_message)
    return [assistant, result.to_message()]


def _turn(rows, final_answer, stop_reason=RunStopReason.FINAL_RESPONSE):
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
    for row in rows:
        messages.extend(row)
    return TurnResult(messages=messages, final_answer=final_answer, stop_reason=stop_reason)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        (self.workspace / "config.json").write_text('{"output": "old.json"}\n')

    def test_a_correct_advisory_answer_wrapped_in_prose_still_passes(self):
        task = Task(id="t", skeleton="s", family="inspection", split="dev", prompt="p",
                    tools=("read_file",), expect=Expect(answer=Answer(value="FINISH-LIME")))
        before = {"config.json": "h"}
        result = _turn([_row("read_file", {"content": "..."})], "The value is FINISH-LIME.")
        score = check(task, result, self.workspace, before)
        self.assertTrue(score["passed"], score["checks"])
        self.assertFalse(score["checks"]["format_exact"])

    def test_a_required_answer_needs_the_exact_format(self):
        task = Task(id="t", skeleton="s", family="stopping", split="dev", prompt="p",
                    tools=("read_file",),
                    expect=Expect(answer=Answer(value="UNCHANGED", format="required")))
        before = {"config.json": "h"}
        wrapped = _turn([], "The answer is UNCHANGED.")
        exact = _turn([], "UNCHANGED")
        self.assertFalse(check(task, wrapped, self.workspace, before)["passed"])
        self.assertTrue(check(task, exact, self.workspace, before)["passed"])

    def test_answer_match_is_whole_token_not_substring(self):
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(answer=Answer(value="DONE", reject=("UNDONE",))))
        before = {}
        self.assertFalse(check(task, _turn([], "UNDONE"), self.workspace, before)["passed"])
        self.assertTrue(check(task, _turn([], "DONE"), self.workspace, before)["passed"])

    def test_files_check_compares_json_by_value_and_text_after_stripping_newline(self):
        (self.workspace / "config.json").write_text('{"a": 1, "b": 2}\n')
        (self.workspace / "note.txt").write_text("hello\n")
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(files={"config.json": '{"b": 2, "a": 1}',
                                        "note.txt": "hello"}))
        before = {"config.json": "h", "note.txt": "h2"}
        self.assertTrue(check(task, _turn([], ""), self.workspace, before)["checks"]["files"])

    def test_unchanged_fails_on_an_unexpected_write(self):
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(files={"config.json": '{"output": "old.json"}'}))
        before = {"config.json": "OLDHASH", "other.txt": "SAMEHASH"}
        after_snapshot_note = "other.txt changed relative to `before`"  # documents the scenario
        # `check` re-snapshots the live workspace, so simulate drift by writing a new file.
        (self.workspace / "other.txt").write_text("changed")
        result = _turn([], "")
        score = check(task, result, self.workspace, before)
        self.assertFalse(score["checks"]["unchanged"], after_snapshot_note)
        self.assertIn("other.txt", score["changed_paths"])

    def test_evidence_requires_the_value_in_a_successful_observation(self):
        task = Task(id="t", skeleton="s", family="inspection", split="dev", prompt="p",
                    tools=("read_file",), expect=Expect(answer=Answer(value="X"),
                                                        evidence=("secret-42",)))
        before = {}
        missing = _turn([_row("read_file", {"content": "no clue here"})], "X")
        present = _turn([_row("read_file", {"content": "secret-42 is the value"})], "X")
        self.assertFalse(check(task, missing, self.workspace, before)["checks"]["evidence"])
        self.assertTrue(check(task, present, self.workspace, before)["checks"]["evidence"])

    def test_process_rules_no_write_attempts_and_check_before_write(self):
        task = Task(id="t", skeleton="s", family="stopping", split="dev", prompt="p", tools=(),
                    expect=Expect(process=("no_write_attempts",)))
        before = {}
        clean = _turn([_row("read_file", {})], "")
        wrote = _turn([_row("read_file", {}), _row("write_file", {}, call_id="c2")], "")
        self.assertTrue(check(task, clean, self.workspace, before)["checks"]["process"])
        self.assertFalse(check(task, wrote, self.workspace, before)["checks"]["process"])

        recovery = Task(id="t2", skeleton="s", family="recovery", split="dev", prompt="p",
                        tools=(), expect=Expect(process=("check_before_write", "passing_check_after_last_write")))
        bad_order = _turn([_row("write_file", {}, call_id="c1"),
                           _row("shell", {"exit_code": 0}, call_id="c2")], "")
        good_order = _turn([_row("shell", {"exit_code": 1}, ok=False,
                                 error_message="Command exited with code 1", call_id="c1"),
                            _row("write_file", {}, call_id="c2"),
                            _row("shell", {"exit_code": 0}, call_id="c3")], "")
        self.assertFalse(check(recovery, bad_order, self.workspace, before)["checks"]["process"])
        self.assertTrue(check(recovery, good_order, self.workspace, before)["checks"]["process"])

    def test_false_completion_flags_a_claimed_but_unmet_task(self):
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(files={"config.json": '{"output": "new.json"}'}),
                    claim_tokens=("DONE",))
        before = {"config.json": "h"}
        lied = _turn([], "DONE")  # config.json was never changed to new.json
        self.assertTrue(check(task, lied, self.workspace, before)["false_completion"])
        no_claim = Task(id="t2", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                        expect=Expect())
        self.assertIsNone(check(no_claim, _turn([], "anything"), self.workspace, before)["false_completion"])

    def test_fault_encountered_searches_the_whole_failed_row(self):
        task = Task(id="t", skeleton="s", family="recovery", split="dev", prompt="p",
                    tools=("read_file",), expect=Expect(answer=Answer(value="X")),
                    fault_signal=("read_file", "Simulated transient read failure."))
        before = {}
        hit = _turn([_row("read_file", {}, ok=False,
                          error_message="Simulated transient read failure."),
                    _row("read_file", {"content": "X"}, call_id="c2")], "X")
        miss = _turn([_row("read_file", {"content": "X"})], "X")
        self.assertTrue(check(task, hit, self.workspace, before)["fault_encountered"])
        self.assertFalse(check(task, miss, self.workspace, before)["fault_encountered"])

    def test_bench_summary_reports_by_skeleton_matched_recovery_and_tokens(self):
        usage_a = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        usage_b1 = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
        usage_b2 = {"prompt_tokens": 80, "completion_tokens": 8, "total_tokens": 88}
        records = [
            {"skeleton": "a", "family": "inspection", "split": "dev", "passed": True,
             "checks": {"format_exact": True}, "false_completion": None, "fault_encountered": None,
             "pair_id": None, "condition": None, "invalid_calls": 0, "model_requests": 2,
             "elapsed_seconds": 1.0, "usage": usage_a},
            {"skeleton": "b", "family": "recovery", "split": "dev", "passed": True,
             "checks": {}, "false_completion": False, "fault_encountered": False,
             "pair_id": "b-0", "condition": "clean", "invalid_calls": 0, "model_requests": 2,
             "elapsed_seconds": 1.0, "usage": usage_b1},
            {"skeleton": "b", "family": "recovery", "split": "dev", "passed": True,
             "checks": {}, "false_completion": False, "fault_encountered": True,
             "pair_id": "b-0", "condition": "fault", "invalid_calls": 1, "model_requests": 3,
             "elapsed_seconds": 2.0, "usage": usage_b2},
        ]
        summary = bench_summary(records)
        self.assertEqual(summary["by_skeleton"]["a"], {"passed": 1, "total": 1, "format_exact": 1})
        self.assertEqual(summary["matched_recovery"],
                         [{"pair_id": "b-0", "complete_pair": True, "clean_passed": True,
                           "fault_passed": True, "fault_encountered": True, "recovered": True}])
        self.assertEqual(summary["false_completion"], {"count": 0, "claimed": 2, "total": 3})
        self.assertEqual(summary["usage"], {"prompt_tokens": 230, "completion_tokens": 23,
                                            "total_tokens": 253})

    def test_bench_summary_usage_is_none_when_any_record_is_missing_it(self):
        records = [{"skeleton": "a", "family": "inspection", "split": "dev", "passed": True,
                   "checks": {}, "false_completion": None, "fault_encountered": None,
                   "pair_id": None, "condition": None, "invalid_calls": 0, "model_requests": 1,
                   "elapsed_seconds": 1.0, "usage": {"prompt_tokens": None,
                                                     "completion_tokens": None, "total_tokens": None}}]
        self.assertIsNone(bench_summary(records)["usage"]["total_tokens"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_verify -q`;
  expect `ModuleNotFoundError`.

- [ ] **Step 3: implementation.** Create `evals/bench/verify.py`:

```python
"""The Stage 8 benchmark's one verifier: interprets `Task.expect`, never a specific call path."""

import json
import re

from .spec import Answer, Task
from ..verify import exchanges, snapshot

EVIDENCE_TOOLS = {"read_file", "grep_text", "glob_files", "list_files", "shell"}
PROCESS_RULES = {}


def _process_rule(name):
    def register(fn):
        PROCESS_RULES[name] = fn
        return fn
    return register


@_process_rule("no_write_attempts")
def _no_write_attempts(rows: list[dict]) -> bool:
    return not any(r["tool_name"] in ("write_file", "edit_file") for r in rows)


@_process_rule("check_before_write")
def _check_before_write(rows: list[dict]) -> bool:
    checks = [i for i, r in enumerate(rows) if r["tool_name"] == "shell"]
    writes = [i for i, r in enumerate(rows) if r["tool_name"] in ("write_file", "edit_file")]
    return bool(checks) and (not writes or checks[0] < writes[0])


@_process_rule("passing_check_after_last_write")
def _passing_check_after_last_write(rows: list[dict]) -> bool:
    checks = [(i, r) for i, r in enumerate(rows) if r["tool_name"] == "shell"]
    writes = [i for i, r in enumerate(rows) if r["tool_name"] in ("write_file", "edit_file")]
    return bool(checks) and checks[-1][1]["ok"] and (not writes or checks[-1][0] > writes[-1])


def _content_matches(actual: str, expected: str) -> bool:
    try:
        return json.loads(actual) == json.loads(expected)
    except ValueError:
        return actual.rstrip("\n") == expected.rstrip("\n")


def _files_ok(task: Task, workspace) -> bool:
    for path, expected in task.expect.files.items():
        target = workspace / path
        if not target.is_file() or not _content_matches(target.read_text(encoding="utf-8"), expected):
            return False
    return True


def _unchanged_ok(task: Task, before: dict, after: dict) -> bool:
    protected = set(task.expect.files) | set(task.expect.may_change)
    return all(before.get(path) == after.get(path)
              for path in before.keys() | after.keys() if path not in protected)


def _evidence_ok(task: Task, rows: list[dict]) -> bool:
    haystacks = [json.dumps(r["output"], ensure_ascii=False) for r in rows
                if r["ok"] and r["tool_name"] in EVIDENCE_TOOLS]
    return all(any(needle in text for text in haystacks) for needle in task.expect.evidence)


def _process_ok(task: Task, rows: list[dict]) -> bool:
    return all(PROCESS_RULES[name](rows) for name in task.expect.process)


def _whole_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", text) is not None


def _answer_content_ok(answer: Answer, reply: str) -> bool:
    return _whole_token(reply, answer.value) and not any(_whole_token(reply, bad) for bad in answer.reject)


def check(task: Task, result, workspace, before: dict) -> dict:
    """Score a completed run against `task.expect`.

    Args:
        task: the generated task, including its expectation.
        result: a `TurnResult` (or anything with the same `.messages`, `.stop_reason`,
            `.final_answer` shape).
        workspace: the task's live workspace, already run.
        before: `evals.verify.snapshot(workspace)` taken before the run.

    Returns:
        dict: `passed`, `checks` (one bool per independent rule), `changed_paths`,
            `false_completion` (None when the task has no `claim_tokens`), and
            `fault_encountered` (None when the task has no `fault_signal`).
    """
    after = snapshot(workspace)
    rows = exchanges(result)
    answer = (result.final_answer or "").strip()
    checks = {"normal_finish": result.stop_reason == "final_response",
             "files": _files_ok(task, workspace), "unchanged": _unchanged_ok(task, before, after),
             "evidence": _evidence_ok(task, rows), "process": _process_ok(task, rows)}
    passed_keys = ["normal_finish", "files", "unchanged", "evidence", "process"]
    if task.expect.answer is not None:
        checks["answer_content"] = _answer_content_ok(task.expect.answer, answer)
        checks["format_exact"] = answer == task.expect.answer.value
        passed_keys.append("answer_content")
        if task.expect.answer.format == "required":
            passed_keys.append("format_exact")
    false_completion = None
    if task.claim_tokens:
        claimed = any(token in answer for token in task.claim_tokens)
        state_ok = checks["files"] and checks["unchanged"] and checks["process"] and checks["evidence"]
        false_completion = claimed and checks["normal_finish"] and not state_ok
    fault_encountered = None
    if task.fault_signal is not None:
        tool_name, needle = task.fault_signal
        fault_encountered = any(not r["ok"] and r["tool_name"] == tool_name
                                and needle in json.dumps(r, ensure_ascii=False) for r in rows)
    changed_paths = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    return {"passed": all(checks[k] for k in passed_keys), "checks": checks,
            "changed_paths": changed_paths, "false_completion": false_completion,
            "fault_encountered": fault_encountered}


def bench_summary(records: list[dict]) -> dict:
    """Aggregate `check()` records: pass rates by skeleton/family, false completion, recovery."""
    by_skeleton, by_family = {}, {}
    for record in records:
        for bucket, key in ((by_skeleton, record["skeleton"]), (by_family, record["family"])):
            row = bucket.setdefault(key, {"passed": 0, "total": 0, "format_exact": 0})
            row["total"] += 1
            row["passed"] += record["passed"]
            row["format_exact"] += record["checks"].get("format_exact") is True
    claimed = [r for r in records if r["false_completion"] is not None]
    pairs: dict[str, dict] = {}
    for record in records:
        if record["pair_id"]:
            pairs.setdefault(record["pair_id"], {})[record["condition"]] = record
    recovery = []
    for pair_id, conditions in pairs.items():
        clean, fault = conditions.get("clean"), conditions.get("fault")
        recovery.append({"pair_id": pair_id, "complete_pair": clean is not None and fault is not None,
            "clean_passed": clean["passed"] if clean else None,
            "fault_passed": fault["passed"] if fault else None,
            "fault_encountered": fault["fault_encountered"] if fault else None,
            "recovered": (fault["passed"] and fault["fault_encountered"]) if fault else None})
    return {"passed": sum(r["passed"] for r in records), "total": len(records),
            "by_skeleton": by_skeleton, "by_family": by_family,
            "false_completion": {"count": sum(r["false_completion"] for r in claimed),
                                 "claimed": len(claimed), "total": len(records)},
            "matched_recovery": recovery,
            "invalid_calls": sum(r["invalid_calls"] for r in records),
            "model_requests": sum(r["model_requests"] for r in records),
            "elapsed_seconds": round(sum(r["elapsed_seconds"] for r in records), 2),
            # Unknown usage in any record leaves the total unknown rather than silently smaller.
            "usage": {key: sum(r["usage"][key] for r in records) if records
                     and all(r["usage"][key] is not None for r in records) else None
                     for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
```

- [ ] **Step 4:** rerun; expect OK, 10 tests. Run the full suite; expect all green.
- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/verify.py agent_from_scratch/tests/test_bench_verify.py
git commit -m "feat: add the Stage 8 content-first verifier and its summary"
```

---

## Task 4: The runner framework (`run.py`, `skeletons.py` scaffolding, `splits.json`)

This is the biggest task: it proves the whole pipeline (build workspace → confined registry →
`Agent.run_turn` → `check()` → record) with one hand-built synthetic skeleton, before any of
the 14 real skeletons exist.

**Files:**
- Create: `evals/bench/run.py`, `evals/bench/skeletons.py`, `evals/bench/splits.json`
- Test: `tests/test_bench_run.py`

**Interfaces — consumes:** `spec.Task`, `spec.BuildContext`, `faults.FaultyTool`,
`verify.check`, `verify.bench_summary`, `evals.run.open_run`, `evals.run.parse_limits`,
`evals.run.save`, `evals.verify.metrics`, `evals.verify.snapshot`, `evals.trajectory.profile`.
**Produces:**
- `Skeleton(name, family, split, seeds: tuple[int, ...], recovery: bool, build, solution)` —
  `build(rng: random.Random, workspace: Path, ctx: BuildContext) -> Task`,
  `solution(task: Task) -> list[LLMResponse]`.
- `SKELETONS: list[Skeleton]` (starts empty; Tasks 5–18 append to it).
- `call(name: str, **arguments) -> LLMResponse`, `answer(text: str) -> LLMResponse`
  (in `skeletons.py`, for every skeleton's `solution()`).
- `specs(split: str | None = None) -> list[tuple[Skeleton, BuildContext]]`
- `registry_for(task: Task, workspace: Path, private: Path) -> ToolRegistry`
- `run_case(model, skeleton: Skeleton, ctx: BuildContext, output: Path, *, overrides: dict | None = None) -> dict`
- `main()`

- [ ] **Step 1: write the failing test**

```python
"""The bench runner works end to end for a hand-built skeleton, before any real one exists."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.evals.bench.faults import FaultyTool
from agent_from_scratch.evals.bench.run import registry_for, run_case, specs
from agent_from_scratch.evals.bench.skeletons import Skeleton, answer, call
from agent_from_scratch.evals.bench.spec import Answer, BuildContext, Expect, Fault, Task
from agent_from_scratch.llm import LLM


def _write_config(rng, workspace, ctx):
    correct = {"output": "report.json", "retries": 3}
    if ctx.condition == "fault":
        (workspace / "config.json").write_text('{"output": "old.json", "retries": 3}\n')
    else:
        (workspace / "config.json").write_text('{"output": "report.json", "retries": 3}\n')
    fault = Fault("shell", 1, "forced check failure") if ctx.condition == "fault" else None
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
                pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=5,
                prompt="Run check_fixture, fix config.json's output if needed, reply CHECKED.",
                tools=("read_file", "write_file", "shell"),
                expect=Expect(answer=Answer(value="CHECKED", format="required"),
                             files={"config.json": '{"output": "report.json", "retries": 3}'},
                             process=(("check_before_write", "passing_check_after_last_write")
                                      if ctx.condition == "fault" else ("no_write_attempts",))),
                claim_tokens=("CHECKED",), fault=fault,
                fault_signal=("shell", "forced") if fault else None,
                debug={"correct": correct})


def _solve(task):
    if task.condition == "fault":
        return [call("read_file", path="config.json"),
                call("write_file", path="config.json", content='{"output": "report.json", "retries": 3}'),
                call("shell", command_id="check_fixture"), answer("CHECKED")]
    return [call("shell", command_id="check_fixture"), answer("CHECKED")]


PROBE = Skeleton(name="probe_check", family="recovery", split="dev", seeds=(0,),
                 recovery=True, build=_write_config, solution=_solve)


def _model(responses):
    model = Mock(spec=LLM)
    model.settings.return_value = {}
    model.read_usage.side_effect = LLM.read_usage
    model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 1000,
        "window_tokens": 32768, "response_reserve": 2048, "remaining_tokens": 29720}
    model.generate.side_effect = list(responses)
    return model


class RunFrameworkTests(unittest.TestCase):
    def test_registry_for_restricts_to_the_tasks_declared_tools(self):
        with TemporaryDirectory() as private:
            workspace = Path(private) / "workspace"
            workspace.mkdir()
            ctx = BuildContext("probe_check", "recovery", "dev", 0, None)
            task = _write_config(None, workspace, ctx)
            registry = registry_for(task, workspace, Path(private))
            self.assertEqual(set(registry.schemas()), {"read_file", "write_file", "shell"})

    def test_run_case_scores_a_clean_recovery_task(self):
        ctx = BuildContext("probe_check", "recovery", "dev", 0, "clean")
        with TemporaryDirectory() as directory:
            record = run_case(_model(_solve(_write_config(None, Path(directory), ctx))),
                              PROBE, ctx, Path(directory) / "run")
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(record["id"], "probe_check-0-clean")

    def test_run_case_exercises_the_shell_fault_and_recovers(self):
        ctx = BuildContext("probe_check", "recovery", "dev", 0, "fault")
        with TemporaryDirectory() as directory:
            probe_workspace = Path(directory) / "probe"
            probe_workspace.mkdir()
            task = _write_config(None, probe_workspace, ctx)
            record = run_case(_model(_solve(task)), PROBE, ctx, Path(directory) / "run")
        self.assertTrue(record["passed"], record["checks"])
        self.assertTrue(record["fault_encountered"])

    def test_specs_lists_every_seed_and_condition_pair(self):
        from agent_from_scratch.evals.bench import skeletons
        original = list(skeletons.SKELETONS)
        skeletons.SKELETONS[:] = [PROBE]
        try:
            entries = specs()
            self.assertEqual([(s.name, c.seed, c.condition) for s, c in entries],
                             [("probe_check", 0, "clean"), ("probe_check", 0, "fault")])
        finally:
            skeletons.SKELETONS[:] = original


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_run -q`; expect
  `ModuleNotFoundError`.

- [ ] **Step 3: implementation.**

Create `evals/bench/skeletons.py` (framework only; the 14 real entries land in Tasks 5–18):

```python
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
```

Create `evals/bench/splits.json` (the full, final list; test skeletons 5–18 fill in as the
plan proceeds — the names below are fixed now, so this file does not change again):

```json
{
  "version": 1,
  "dev": ["pointer_lookup", "single_field_edit", "check_fix_recheck", "no_op_correct_config"],
  "test": ["deep_chain_lookup", "grep_locate", "sum_across_files", "pointer_nested_edit",
           "rename_key_all_files", "append_list_item", "transient_read_failure",
           "flaky_write_retry", "ambiguous_choice_stop", "missing_file_report"],
  "train": []
}
```

Create `evals/bench/run.py`:

```python
"""Build Stage 8 benchmark tasks from seeds, run them, and score them.

Usage: python -m agent_from_scratch.evals bench --output DIR --split {dev,test} [--tasks a,b]
           [--seed N] [--limits '{"planning": true}'] [--final] [--allow-drift]
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

from .faults import FaultyTool
from .skeletons import SKELETONS, Skeleton
from .spec import BuildContext, Task
from .verify import bench_summary, check
from ..run import open_run, parse_limits, save
from ..trajectory import profile
from ..verify import metrics, snapshot
from ...agent import Agent
from ...tools.base import ToolRegistry
from ...tools.calculator import CalculatorTool
from ...tools.files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from ...tools.search import GlobFilesTool, GrepTextTool
from ...tools.shell import Command, ShellTool

HERE = Path(__file__).resolve().parent
CHECK_SCRIPT = HERE.parent / "check.py"


def specs(split: str | None = None) -> list[tuple[Skeleton, BuildContext]]:
    """(skeleton, BuildContext) pairs in registration order; touches no disk."""
    entries = []
    for skeleton in SKELETONS:
        if split is not None and skeleton.split != split:
            continue
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
    """Build the workspace from `ctx`'s seed, run one turn, then score it. No history is carried."""
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="tiny-agent-bench-") as temporary:
        private = Path(temporary).resolve()
        workspace = private / "workspace"
        workspace.mkdir()
        rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
        task = skeleton.build(rng, workspace, ctx)
        save(output / "task.json", asdict(task))
        before = snapshot(workspace)
        registry = registry_for(task, workspace, private)
        save(output / "schemas.json", registry.schemas())
        agent = Agent(model, str(output / "state"), registry=registry)
        agent.limits = replace(agent.limits, **{**(overrides or {}), "max_iterations": task.max_iterations})
        result = agent.run_turn(task.prompt, session_id=task.id)
        score = check(task, result, workspace, before)
        record = {"id": task.id, "skeleton": task.skeleton, "family": task.family,
                 "split": task.split, "pair_id": task.pair_id, "condition": task.condition,
                 **metrics(result), **score}
        save(output / "result.json", record)
        return record


def main():
    from .manifest import manifest_drift
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--tasks", help="Comma-separated task IDs; default: every task in --split")
    parser.add_argument("--seed", type=int, default=11, help="Decoding seed; workspaces are seeded separately")
    parser.add_argument("--limits", default="{}", help='JSON AgentLimits overrides')
    parser.add_argument("--final", action="store_true", help="Required to run --split test")
    parser.add_argument("--allow-drift", action="store_true", help="Run --final despite manifest drift")
    args = parser.parse_args()
    overrides = parse_limits(parser, args.limits)
    if args.split == "test" and not args.final:
        parser.error("Running the test split needs --final (see docs/STAGE8_DESIGN.md).")
    entries = specs(args.split)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        entries = [(s, c) for s, c in entries if c.id in wanted]
        if len(entries) != len(wanted):
            parser.error("Unknown task ID in --tasks.")
    drift = manifest_drift() if args.final else {}
    if drift and not args.allow_drift:
        parser.error(f"Manifest drift in {sorted(drift)}; rerun with --allow-drift if intended.")
    out = args.output.resolve()
    model = open_run(out, parser, suite=f"bench-{args.split}-v1", seed=args.seed, overrides=overrides,
                     split=args.split, final=args.final, manifest_drift=drift,
                     selected_tasks=[c.id for _, c in entries])
    records = []
    for skeleton, ctx in entries:
        print("START", ctx.id, flush=True)
        record = run_case(model, skeleton, ctx, out / ctx.id, overrides=overrides)
        records.append(record)
        save(out / "results.json", records)
        save(out / "summary.json", bench_summary(records))
        print("RESULT", ctx.id, "PASS" if record["passed"] else "FAIL", record["stop_reason"],
              "requests", record["model_requests"], flush=True)
        if record["stop_reason"] == "interrupted":
            break
    save(out / "trajectory.json", profile(records))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** rerun; expect OK, 4 tests. Run the full suite; expect all green (the
  `manifest.py` import inside `main()` is deferred to call time, so it does not need to exist
  yet for the tests above, which call `run_case`/`registry_for`/`specs` directly, not `main`).

- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/run.py agent_from_scratch/evals/bench/skeletons.py \
           agent_from_scratch/evals/bench/splits.json agent_from_scratch/tests/test_bench_run.py
git commit -m "feat: add the Stage 8 benchmark runner framework"
```

---

## Task 5: Dev skeleton `pointer_lookup`, and the shared skeleton gate test

**Files:**
- Modify: `evals/bench/skeletons.py`
- Create: `tests/test_bench_skeletons.py`

**Interfaces — produces:** `_build_pointer_lookup`, `_solution_pointer_lookup`; the
`pointer_lookup` entry in `SKELETONS`. From here on, `tests/test_bench_skeletons.py` is not
edited again — it iterates `SKELETONS` and Tasks 6–18 each add one entry to that list.

- [ ] **Step 1: append the entry, referencing not-yet-defined names.** At the bottom of
  `evals/bench/skeletons.py`:

```python
SKELETONS.append(Skeleton(
    name="pointer_lookup", family="inspection", split="dev", seeds=(0, 1, 2), recovery=False,
    build=_build_pointer_lookup, solution=_solution_pointer_lookup))
```

Create `tests/test_bench_skeletons.py`:

```python
"""Every registered skeleton's scripted solution passes; a fake claim fails; seeds reproduce."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import SKELETONS, answer
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.llm import LLM

SPLITS_PATH = Path(__file__).resolve().parents[1] / "evals" / "bench" / "splits.json"


def _model(responses):
    model = Mock(spec=LLM)
    model.settings.return_value = {}
    model.read_usage.side_effect = LLM.read_usage
    model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 1000,
        "window_tokens": 32768, "response_reserve": 2048, "remaining_tokens": 29720}
    model.generate.side_effect = list(responses)
    return model


class SkeletonGateTests(unittest.TestCase):
    def _task(self, skeleton, ctx):
        directory = Path(self.enterContext(TemporaryDirectory()))
        rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
        return skeleton.build(rng, directory, ctx)

    def test_every_scripted_solution_passes(self):
        for index, (skeleton, ctx) in enumerate(specs()):
            with self.subTest(id=ctx.id):
                task = self._task(skeleton, ctx)
                output = Path(self.enterContext(TemporaryDirectory())) / f"run-{index}"
                record = run_case(_model(skeleton.solution(task)), skeleton, ctx, output)
                self.assertTrue(record["passed"], record["checks"])

    def test_a_fake_claim_with_no_tool_calls_fails(self):
        for index, (skeleton, ctx) in enumerate(specs()):
            with self.subTest(id=ctx.id):
                task = self._task(skeleton, ctx)
                claim = (task.claim_tokens[0] if task.claim_tokens else
                        task.expect.answer.value if task.expect.answer else "done")
                output = Path(self.enterContext(TemporaryDirectory())) / f"fake-{index}"
                record = run_case(_model([answer(claim)]), skeleton, ctx, output)
                self.assertFalse(record["passed"], record["checks"])

    def test_the_same_seed_rebuilds_an_identical_workspace(self):
        for skeleton, ctx in specs():
            with self.subTest(id=ctx.id):
                a = Path(self.enterContext(TemporaryDirectory()))
                b = Path(self.enterContext(TemporaryDirectory()))
                skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), a, ctx)
                skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), b, ctx)
                self.assertEqual(snapshot(a), snapshot(b))

    def test_registered_skeletons_are_unique_and_declare_the_right_split(self):
        names = [skeleton.name for skeleton in SKELETONS]
        self.assertEqual(len(names), len(set(names)), "duplicate skeleton name")
        splits = json.loads(SPLITS_PATH.read_text())
        for skeleton in SKELETONS:
            self.assertIn(skeleton.name, splits[skeleton.split])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_skeletons -q`;
  expect `NameError: name '_build_pointer_lookup' is not defined`.

- [ ] **Step 3: implementation.** Above the `SKELETONS` list in `evals/bench/skeletons.py`,
  add (importing `Answer`, `Expect` alongside the existing `BuildContext`, `Task` import):

```python
from .spec import Answer, BuildContext, Expect, Task
```

```python
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
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/skeletons.py agent_from_scratch/tests/test_bench_skeletons.py
git commit -m "feat: add the pointer_lookup dev skeleton and the shared skeleton gate test"
```

---

## Task 6: Dev skeleton `single_field_edit`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="single_field_edit", family="updates", split="dev", seeds=(0, 1, 2), recovery=False, build=_build_single_field_edit, solution=_solution_single_field_edit))`.
- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_skeletons -q`; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
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
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the single_field_edit dev skeleton"`

---

## Task 7: Dev skeleton `check_fix_recheck` (recovery pair)

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="check_fix_recheck", family="recovery", split="dev", seeds=(0, 1, 2), recovery=True, build=_build_check_fix_recheck, solution=_solution_check_fix_recheck))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_check_fix_recheck(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    correct = {"output": "report.json", "retries": rng.randint(1, 5), "enabled": True}
    if ctx.condition == "fault":
        _write(root, "config.json", _json({**correct, "output": "old.json"}))
        process = ("check_before_write", "passing_check_after_last_write")
    else:
        _write(root, "config.json", _json(correct))
        process = ("no_write_attempts",)
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
```

Note: `registry_for` reads `task.debug["correct"]` to write `check-spec.json` for any task
that offers `shell` — this key name must match what Task 4's `registry_for` already expects.

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the check_fix_recheck dev recovery skeleton"`

---

## Task 8: Dev skeleton `no_op_correct_config`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="no_op_correct_config", family="stopping", split="dev", seeds=(0, 1, 2), recovery=False, build=_build_no_op_correct_config, solution=_solution_no_op_correct_config))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
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
```

- [ ] **Step 4:** rerun; expect OK, dev set complete (15 tasks). Run the full suite; expect
  all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the no_op_correct_config dev skeleton, completing the dev set"`

---

## Task 9: Test skeleton `deep_chain_lookup`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="deep_chain_lookup", family="inspection", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_deep_chain_lookup, solution=_solution_deep_chain_lookup))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
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
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the deep_chain_lookup test skeleton"`

---

## Task 10: Test skeleton `grep_locate`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="grep_locate", family="inspection", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_grep_locate, solution=_solution_grep_locate))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
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
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the grep_locate test skeleton"`

---

## Task 11: Test skeleton `sum_across_files`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="sum_across_files", family="inspection", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_sum_across_files, solution=_solution_sum_across_files))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
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
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the sum_across_files test skeleton"`

---

## Task 12: Test skeleton `pointer_nested_edit`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="pointer_nested_edit", family="updates", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_pointer_nested_edit, solution=_solution_pointer_nested_edit))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
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
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the pointer_nested_edit test skeleton"`

---

## Task 13: Test skeleton `rename_key_all_files`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="rename_key_all_files", family="updates", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_rename_key_all_files, solution=_solution_rename_key_all_files))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_rename_key_all_files(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    count = rng.randint(3, 5)
    old_key, new_key = "endpoint", "url"
    contents = {}
    for i in range(count):
        content = {old_key: f"https://svc-{i}.internal/api", "timeout": rng.randint(1, 10)}
        path = f"services/svc_{i:02d}.json"
        contents[path] = content
        _write(root, path, _json(content))
    _write(root, "legacy/svc_00.json", _json({old_key: "https://legacy.internal/api", "timeout": 5}))
    expected = {path: _json({new_key: content[old_key], "timeout": content["timeout"]})
               for path, content in contents.items()}
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=count * 2 + 2,
               prompt=(f"Under services/, every file has a {old_key!r} field. Rename that "
                      f"field to {new_key!r} in every file under services/ (keep its value "
                      "and every other field), but do not touch anything under legacy/. "
                      "Reply DONE."),
               tools=("list_files", "read_file", "write_file"),
               expect=Expect(answer=Answer(value="DONE", format="required"), files=expected),
               claim_tokens=("DONE",), debug={"paths": list(contents)})


def _solution_rename_key_all_files(task: Task) -> list:
    calls = [call("list_files", path="services", recursive=True)]
    for path in task.debug["paths"]:
        calls.append(call("read_file", path=path))
        calls.append(call("write_file", path=path, content=json.dumps(json.loads(task.expect.files[path]))))
    return calls + [answer("DONE")]
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the rename_key_all_files test skeleton"`

---

## Task 14: Test skeleton `append_list_item`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="append_list_item", family="updates", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_append_list_item, solution=_solution_append_list_item))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_append_list_item(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    registries = [f"registry_{c}" for c in "ab"[:rng.randint(2, 3)]]
    target = rng.choice(registries)
    items = {name: [f"item-{name}-{i}" for i in range(rng.randint(1, 3))] for name in registries}
    for name in registries:
        _write(root, f"{name}.json", _json({"items": items[name]}))
    new_item = f"item-new-{rng.randint(100, 999)}"
    expected_items = [*items[target], new_item]
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=(f"Read {target}.json and append {new_item!r} to the end of its items "
                      "list, keeping the existing items in order. Do not change any other "
                      "registry file. Reply DONE."),
               tools=("read_file", "write_file"),
               expect=Expect(answer=Answer(value="DONE", format="required"),
                            files={f"{target}.json": _json({"items": expected_items})}),
               claim_tokens=("DONE",), debug={"target": target})


def _solution_append_list_item(task: Task) -> list:
    target = task.debug["target"]
    expected = json.loads(task.expect.files[f"{target}.json"])
    return [call("read_file", path=f"{target}.json"),
           call("write_file", path=f"{target}.json", content=json.dumps(expected)), answer("DONE")]
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the append_list_item test skeleton"`

---

## Task 15: Test skeleton `transient_read_failure` (recovery pair)

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="transient_read_failure", family="recovery", split="test", seeds=(0, 1, 2), recovery=True, build=_build_transient_read_failure, solution=_solution_transient_read_failure))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_transient_read_failure(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    value = f"CODE-{rng.randint(1000, 9999)}"
    _write(root, "pointer.json", _json({"next": "detail.json"}))
    _write(root, "detail.json", _json({"value": value}))
    fault = fault_signal = None
    if ctx.condition == "fault":
        # call #2 on the shared read_file tool is the first attempt at detail.json.
        fault = Fault(tool="read_file", on_call=2, message="Simulated transient read failure.")
        fault_signal = ("read_file", "Simulated transient read failure.")
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=("Read pointer.json, then read the file it names under 'next', then "
                      "report only the value field of that file. If a read fails, try it "
                      "again."),
               tools=("read_file",),
               expect=Expect(answer=Answer(value=value), evidence=(value,),
                            process=("no_write_attempts",)),
               fault=fault, fault_signal=fault_signal, debug={"value": value})


def _solution_transient_read_failure(task: Task) -> list:
    calls = [call("read_file", path="pointer.json")]
    if task.condition == "fault":
        calls.append(call("read_file", path="detail.json"))  # this attempt is made to fail
    calls.append(call("read_file", path="detail.json"))
    return calls + [answer(task.expect.answer.value)]
```

Add `from .spec import Answer, BuildContext, Expect, Fault, Task` at the top of
`skeletons.py` (extends the existing import with `Fault`).

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the transient_read_failure test recovery skeleton"`

---

## Task 16: Test skeleton `flaky_write_retry` (recovery pair)

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="flaky_write_retry", family="recovery", split="test", seeds=(0, 1, 2), recovery=True, build=_build_flaky_write_retry, solution=_solution_flaky_write_retry))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_flaky_write_retry(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    config = {"mode": rng.choice(["draft", "staged"]), "count": rng.randint(1, 9)}
    _write(root, "state.json", _json(config))
    expected = {**config, "mode": "final"}
    fault = fault_signal = None
    if ctx.condition == "fault":
        fault = Fault(tool="write_file", on_call=1, message="Simulated transient write failure.")
        fault_signal = ("write_file", "Simulated transient write failure.")
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=6,
               prompt=("Change state.json's mode field to 'final', keeping every other field, "
                      "and save it. If the write fails, try it again. Reply DONE only after "
                      "the file is saved."),
               tools=("write_file",),
               expect=Expect(answer=Answer(value="DONE", format="required"),
                            files={"state.json": _json(expected)}),
               claim_tokens=("DONE",), fault=fault, fault_signal=fault_signal,
               debug={"expected": expected})


def _solution_flaky_write_retry(task: Task) -> list:
    expected = task.debug["expected"]
    calls = [call("write_file", path="state.json", content=json.dumps(expected))]
    if task.condition == "fault":
        calls.append(call("write_file", path="state.json", content=json.dumps(expected)))
    return calls + [answer("DONE")]
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the flaky_write_retry test recovery skeleton"`

---

## Task 17: Test skeleton `ambiguous_choice_stop`

**Files:** Modify `evals/bench/skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="ambiguous_choice_stop", family="stopping", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_ambiguous_choice_stop, solution=_solution_ambiguous_choice_stop))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_ambiguous_choice_stop(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    a, b = "profile_north", "profile_south"
    _write(root, f"{a}.json", _json({"output": "north.json"}))
    _write(root, f"{b}.json", _json({"output": "south.json"}))
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=5,
               prompt=(f"Two profile files exist, {a}.json and {b}.json, and nothing marks "
                      "either one active. Update the active profile's output to "
                      "'current.json'. If you cannot tell which profile is active, change "
                      "nothing and reply only NEED_INPUT."),
               tools=("list_files", "read_file", "write_file"),
               expect=Expect(answer=Answer(value="NEED_INPUT", format="required"),
                            evidence=(a, b), process=("no_write_attempts",)))


def _solution_ambiguous_choice_stop(task: Task) -> list:
    return [call("list_files", path="."), answer("NEED_INPUT")]
```

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/bench/skeletons.py && git commit -m "feat: add the ambiguous_choice_stop test skeleton"`

---

## Task 18: Test skeleton `missing_file_report`, completing the test set

**Files:** Modify `evals/bench/skeletons.py`, `tests/test_bench_skeletons.py`.

- [ ] **Step 1:** append `SKELETONS.append(Skeleton(name="missing_file_report", family="stopping", split="test", seeds=(0, 1, 2, 3, 4, 5), recovery=False, build=_build_missing_file_report, solution=_solution_missing_file_report))`.
- [ ] **Step 2:** run the gate test; expect `NameError`.
- [ ] **Step 3: implementation.**

```python
def _build_missing_file_report(rng: random.Random, root: Path, ctx: BuildContext) -> Task:
    decoy_value = f"STALE-{rng.randint(1000, 9999)}"
    _write(root, "config.json.bak", _json({"output": decoy_value}))
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
               pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=5,
               prompt=("Read config.json and report its output field. Only config.json.bak, "
                      "not config.json, may exist here; do not use the backup as if it were "
                      "current. If config.json is missing, reply only MISSING."),
               tools=("list_files", "read_file"),
               expect=Expect(answer=Answer(value="MISSING", format="required", reject=(decoy_value,)),
                            process=("no_write_attempts",)),
               debug={"decoy": decoy_value})


def _solution_missing_file_report(task: Task) -> list:
    return [call("list_files", path="."), answer("MISSING")]
```

Then extend `tests/test_bench_skeletons.py`'s completeness test to check both directions, now
that all 14 skeletons exist. Replace
`test_registered_skeletons_are_unique_and_declare_the_right_split` with:

```python
    def test_registered_skeletons_match_splits_json_exactly(self):
        names = [skeleton.name for skeleton in SKELETONS]
        self.assertEqual(len(names), len(set(names)), "duplicate skeleton name")
        splits = json.loads(SPLITS_PATH.read_text())
        listed = {name for group in (splits["dev"], splits["test"], splits["train"]) for name in group}
        self.assertEqual(set(names), listed)
        for skeleton in SKELETONS:
            self.assertIn(skeleton.name, splits[skeleton.split])
        self.assertEqual(len([s for s in SKELETONS if s.split == "dev"]), 4)
        self.assertEqual(len([s for s in SKELETONS if s.split == "test"]), 10)
        dev_total = sum(len(s.seeds) * (2 if s.recovery else 1) for s in SKELETONS if s.split == "dev")
        test_total = sum(len(s.seeds) * (2 if s.recovery else 1) for s in SKELETONS if s.split == "test")
        self.assertEqual((dev_total, test_total), (15, 60))
```

- [ ] **Step 4:** rerun `python -m unittest agent_from_scratch.tests.test_bench_skeletons -q`;
  expect OK, all 14 skeletons covered, dev=15/test=60 confirmed. Run the full suite; expect
  all green (should be well over 260 tests by now).
- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/skeletons.py agent_from_scratch/tests/test_bench_skeletons.py
git commit -m "feat: add the missing_file_report test skeleton, completing all 14 skeletons"
```

---

## Task 19: Wire `bench` into the evals dispatcher

**Files:** Modify `evals/__main__.py`.

**Interfaces — consumes:** `evals.bench.run.main`.

- [ ] **Step 1: write the failing test** (append to `tests/test_bench_run.py`)

```python
class DispatcherTests(unittest.TestCase):
    def test_bench_is_a_registered_command(self):
        from agent_from_scratch.evals.__main__ import COMMANDS
        self.assertEqual(COMMANDS["bench"], "bench.run")
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_run -q`; expect
  `KeyError: 'bench'`.

- [ ] **Step 3: implementation.** In `evals/__main__.py`:

```python
"""One entry point for evaluation: python -m agent_from_scratch.evals COMMAND [options]

Commands:
  dev       17 small tasks: reading, editing, recovery, stopping   (real model, ~2 min)
  pressure  9-file tasks that push the window past the elision and summary triggers (real model)
  bench     Stage 8 generated benchmark: --split dev (15 tasks) or --split test (60, needs --final)
  freeze    write/update the Stage 8 benchmark's frozen manifest (no model)
  compare   profile one run, or pair two runs by task ID          (no model)

Each command takes --help. Start with evals/README.md.
"""

import importlib
import sys

COMMANDS = {"dev": "run", "pressure": "pressure", "bench": "bench.run",
           "freeze": "bench.manifest", "compare": "trajectory"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command = sys.argv.pop(1)
    importlib.import_module(f"{__package__}.{COMMANDS[command]}").main()


if __name__ == "__main__":
    main()
```

(`freeze` maps to `bench.manifest`, created next task; the import is deferred to call time,
so referencing it here does not require the module to exist until `freeze` is actually run.)

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect all green (the `bench dev`/
  `bench test` paths still cannot run end-to-end as a CLI until Task 20 exists, because
  `run.py::main()` imports `manifest_drift` from `.manifest` — that import only executes when
  `main()` is called, which no test calls yet).
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/__main__.py agent_from_scratch/tests/test_bench_run.py && git commit -m "feat: wire bench and freeze into the evals dispatcher"`

---

## Task 20: The freeze manifest (`manifest.py`)

**Files:**
- Create: `evals/bench/manifest.py`
- Test: `tests/test_bench_manifest.py`

**Interfaces — consumes:** `bench.run.specs`, `bench.skeletons.SKELETONS`, `evals.verify.digest`,
`config.AgentLimits`, `config.CHAT_TEMPLATE_PATH`, `config.MAX_TOKENS`, `config.N_CTX`,
`config.TEMPERATURE`. **Produces:** `build_manifest() -> dict` (no model load, no disk writes
beyond reading source files), `save_manifest() -> None`, `manifest_drift() -> dict[str, tuple]`,
`main()`.

- [ ] **Step 1: write the failing test**

```python
"""The freeze manifest is reproducible and reports drift when a frozen file changes."""

from pathlib import Path
from unittest.mock import patch
import unittest

from agent_from_scratch.evals.bench.manifest import build_manifest, manifest_drift, MANIFEST_PATH


class ManifestTests(unittest.TestCase):
    def test_build_manifest_is_reproducible_and_has_no_model_dependent_fields(self):
        first, second = build_manifest(), build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["memory"], "off")
        self.assertEqual(len(first["tasks"]), 75)  # 15 dev + 60 test

    def test_manifest_drift_is_empty_before_any_freeze(self):
        with patch("agent_from_scratch.evals.bench.manifest.MANIFEST_PATH", Path("/nonexistent.json")):
            self.assertEqual(manifest_drift(), {})

    def test_manifest_drift_reports_a_changed_field(self):
        frozen = build_manifest()
        frozen["memory"] = "on"  # simulate a manifest that no longer matches the tree
        with patch("agent_from_scratch.evals.bench.manifest.MANIFEST_PATH") as path:
            path.exists.return_value = True
            import json
            path.read_text.return_value = json.dumps(frozen)
            drift = manifest_drift()
        self.assertIn("memory", drift)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_manifest -q`;
  expect `ModuleNotFoundError`.

- [ ] **Step 3: implementation.** Create `evals/bench/manifest.py`:

```python
"""Freeze the Stage 8 benchmark: everything that must not move once the test split runs.

Usage: python -m agent_from_scratch.evals freeze
Writes evals/bench/manifest.json. Run it once, after the last skeleton lands, and again only
when a deliberate change (a new skeleton, a limits default, a prompt file) needs a new freeze.
"""

import dataclasses
import json
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory

from .run import specs
from ..verify import digest
from ...config import AgentLimits, CHAT_TEMPLATE_PATH, MAX_TOKENS, N_CTX, TEMPERATURE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"


def _source_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in ROOT.rglob("*")
           if p.is_file() and p.suffix in {".py", ".jinja"} and "__pycache__" not in p.parts}


def build_manifest() -> dict:
    """Everything a Stage 8 test-split run must match. Loads no model; cheap enough to call
    before every `--final` run."""
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    tasks = []
    for skeleton, ctx in specs():
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
            task = skeleton.build(rng, workspace, ctx)
            tasks.append({"id": task.id, "skeleton": task.skeleton, "split": task.split,
                         "prompt_sha256": digest(task.prompt.encode()),
                         "expect_sha256": digest(json.dumps(dataclasses.asdict(task.expect),
                                                            sort_keys=True).encode())})
    return {"source_sha256": _source_hashes(),
           "system_prompt_sha256": digest((ROOT / "prompts/system.md").read_bytes()),
           "chat_template_sha256": digest(CHAT_TEMPLATE_PATH.read_bytes()),
           "n_ctx": N_CTX, "decoding": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
           "limits": dataclasses.asdict(AgentLimits()), "memory": "off",
           "splits_sha256": digest((HERE / "splits.json").read_bytes()),
           "tasks": tasks, "verifier_sha256": digest((HERE / "verify.py").read_bytes()),
           "git_revision": revision}


def save_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n")


def manifest_drift() -> dict[str, tuple]:
    """Fields that differ from the committed manifest; {} if unfrozen or unchanged."""
    if not MANIFEST_PATH.exists():
        return {}
    frozen = json.loads(MANIFEST_PATH.read_text())
    current = build_manifest()
    drift = {key: (frozen.get(key), current.get(key)) for key in frozen
            if key != "tasks" and frozen.get(key) != current.get(key)}
    if {t["id"]: t for t in frozen.get("tasks", [])} != {t["id"]: t for t in current.get("tasks", [])}:
        drift["tasks"] = "task set or content changed"
    return drift


def main():
    save_manifest()
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** rerun; expect OK, 3 tests. Run the full suite; expect all green.
- [ ] **Step 5: commit**

```bash
git add -f agent_from_scratch/evals/bench/manifest.py agent_from_scratch/tests/test_bench_manifest.py
git commit -m "feat: add the Stage 8 freeze manifest and drift check"
```

---

## Task 21: Prove the `--final`/`--allow-drift` guard end to end

Task 4's `run.py::main()` already checks `--final` and calls `manifest_drift()` (imported
**locally**, inside `main()`, specifically so `run.py` never imports `manifest.py` at module
level — `manifest.py` already imports `specs` from `run.py`, and a module-level import the
other way would be a circular import). This task adds the test that exercises the guard
through the CLI's argument parsing, now that `manifest.py` exists. No production code changes.

**Files:** Modify `tests/test_bench_run.py`.

**Interfaces — consumes:** `bench.manifest.manifest_drift` (patched at its definition site, not
at its use site, since `run.py` imports it fresh inside `main()` on every call).

- [ ] **Step 1: write the test**

```python
class FinalGuardTests(unittest.TestCase):
    def test_split_test_without_final_is_rejected(self):
        import sys
        from agent_from_scratch.evals.bench.run import main
        old_argv = sys.argv
        sys.argv = ["bench", "--output", "/tmp/should-not-be-created", "--split", "test"]
        try:
            with self.assertRaises(SystemExit):
                main()
        finally:
            sys.argv = old_argv

    def test_drift_without_allow_drift_is_rejected(self):
        import sys
        from unittest.mock import patch
        from agent_from_scratch.evals.bench.run import main
        old_argv = sys.argv
        sys.argv = ["bench", "--output", "/tmp/should-not-be-created-2", "--split", "test", "--final"]
        try:
            # Patch where manifest_drift is defined: `main()` does `from .manifest import
            # manifest_drift` fresh on every call, so the patched module attribute is what
            # it picks up, with no need for run.py to import manifest.py at module level.
            with patch("agent_from_scratch.evals.bench.manifest.manifest_drift",
                      return_value={"memory": ("off", "on")}):
                with self.assertRaises(SystemExit):
                    main()
        finally:
            sys.argv = old_argv
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_bench_run -q`; expect
  both to already pass — this task documents and locks in behavior Task 4 already implements,
  it does not add any.
- [ ] **Step 3:** run the full suite; expect all green.
- [ ] **Step 4: commit:** `git add -f agent_from_scratch/tests/test_bench_run.py && git commit -m "test: exercise the --final and --allow-drift guard through the CLI"`

---

## Task 22: Document `bench`/`freeze` in `evals/README.md`

**Files:** Modify `evals/README.md`. No tests (documentation only).

- [ ] **Step 1:** Read the current file to find the command table and file-map table (added in
  the empirical-study work).
- [ ] **Step 2:** Add a row to the "I want to know…" table:
  `| Does a change hold on the frozen Stage 8 benchmark? | \`bench --split dev\` (15 tasks, current tools) or \`bench --split test --final\` (60 tasks, needs a frozen manifest) | real | dev: ~1 min; test: ~5 min |`
- [ ] **Step 3:** Add a row to the file-map table for `bench/` pointing at
  `docs/STAGE8_DESIGN.md` for its shapes, and one line noting `freeze` writes
  `evals/bench/manifest.json` and should be run once, after the last skeleton lands.
- [ ] **Step 4:** No test to run; visually confirm the file renders as valid Markdown (no
  broken table pipes).
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/evals/README.md && git commit -m "docs: document the bench and freeze commands"`

---

## Task 23: Amend `STAGE.md` for the memory deferral and Stage 8 boxes

**Files:** Modify `docs/STAGE.md`. No tests (documentation only).

- [ ] **Step 1:** Find Stage 4A–4B's introduction (`## Stage 4 — durable memory · 4A–4B
  required`) and change `4A–4B required` to `4A–4B deferred past Stage 11 (decided September
  22, 2026)`. Add one sentence: "The Stage 8 freeze records `memory: off`; the primary weight
  comparisons in Stages 9–11 run with memory disabled, so 4A–4B is no longer a precondition of
  the freeze. Revisit after Stage 11. See `docs/STAGE8_DESIGN.md`."
- [ ] **Step 2:** In the "Next coding session" paragraph near the end of the file, remove
  "Then start Stage 4A–4B memory... Complete it before the Stage 8 freeze" and replace with a
  pointer to the now-complete Stage 8 benchmark and `docs/benchmark.md`.
- [ ] **Step 3:** In the Stage 8 section, tick the boxes this plan satisfies: the four bullets
  covering task families/variation, split isolation, independent verification/reporting, and
  the trainable-checkpoint bullet (leave that one unticked with a note "front half done here;
  the trainable-checkpoint run is Stage 9's first bullet" since Stage 9 has not happened yet).
- [ ] **Step 4:** No test to run; visually confirm no dangling references to the old wording.
- [ ] **Step 5: commit:** `git add -f agent_from_scratch/docs/STAGE.md && git commit -m "docs: defer Stage 4A-4B memory past Stage 11 and tick satisfied Stage 8 boxes"`

---

## Task 24: Real-model dev pilot (15 tasks)

Operational, not code. Fix **evaluator** bugs only; do not tune skeleton difficulty from
individual failures beyond what the design already fixed.

- [ ] **Step 1:** From the repository root:
  `python -m agent_from_scratch.evals bench --output outputs/bench-dev-pilot --split dev`
- [ ] **Step 2:** Read `outputs/bench-dev-pilot/summary.json` and each `outputs/bench-dev-pilot/<id>/result.json`.
  For any failure, inspect `checks` and the trace under `<id>/state/runs/*.jsonl` to decide:
  a genuine model failure (leave it) or an evaluator bug (fix the verifier or a skeleton's
  `expect`, add a regression test in `tests/test_bench_verify.py` or
  `tests/test_bench_skeletons.py`, rerun the full suite, commit the fix separately).
- [ ] **Step 3:** Create `docs/benchmark.md` recording: the pilot's pass rate by family/
  skeleton, `format_exact` vs content-only passes, any evaluator fixes made and why, and the
  raw command used.
- [ ] **Step 4:** `git add -f agent_from_scratch/docs/benchmark.md && git commit -m "docs: record the Stage 8 dev pilot on the real model"`

---

## Task 25: Freeze, then the real-model test run (60 tasks)

- [ ] **Step 1:** `python -m agent_from_scratch.evals freeze` — writes
  `evals/bench/manifest.json`.
- [ ] **Step 2:** `git add -f agent_from_scratch/evals/bench/manifest.json && git commit -m "feat: freeze the Stage 8 benchmark manifest"`
- [ ] **Step 3:** `python -m agent_from_scratch.evals bench --output outputs/bench-test-run --split test --final`
- [ ] **Step 4:** Confirm the run completed without `manifest_drift` firing (check
  `outputs/bench-test-run/metadata.json`'s `manifest_drift` field is `{}`). If a run fails to
  start because of unintended drift (for example a stray uncommitted edit), fix the tree,
  rerun `freeze`, and rerun the test split — do not use `--allow-drift` to paper over an
  accidental change.
- [ ] **Step 5:** Append the test-run results to `docs/benchmark.md`: total pass rate, by
  family and skeleton, `false_completion` count/denominator, the matched recovery table
  (`clean_passed`, `fault_passed`, `fault_encountered`, `recovered`), tokens, requests and
  elapsed time. State plainly that this is a **pipeline check only**: one greedy seed, and the
  base control for weight comparisons is the Stage 9 checkpoint, not this run.
- [ ] **Step 6:** `git add -f agent_from_scratch/docs/benchmark.md && git commit -m "docs: record the Stage 8 real-model test-split pipeline check"`

---

## Task 26: Independent items — real-model compaction demos

**Files:** Modify `docs/context-memory.md`. No new tests (these demos already have their own
scripts and are read-only against the codebase).

- [ ] **Step 1:** `python -m agent_from_scratch.examples.continuation_demo` (check the script's
  own `--help` for required flags; it pins `n_ctx=4096`).
- [ ] **Step 2:** `python -m agent_from_scratch.examples.compact_demo` (same window).
- [ ] **Step 3:** In `docs/context-memory.md`, record what each demo retained and lost across
  compaction at 4096 tokens, alongside the existing evidence. Also add the pressure-suite T3
  finding from the empirical study (the run that summarized away one of two codes in
  `find_two_reports`) as a second real-model example of information loss under summarization,
  with a link to `docs/EMPIRICAL_STUDY_PLAN.md`'s Task 3.3 section.
- [ ] **Step 4:** `git add -f agent_from_scratch/docs/context-memory.md && git commit -m "docs: log real-model compaction demo evidence and the pressure-suite retained/lost example"`

---

## Final check

- [ ] Run `python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3` one more
  time; confirm the count grew by roughly 35–40 tests (spec, faults, verify, run, skeletons,
  manifest, dispatcher, guard) over the pre-Stage-8 baseline and all are green.
- [ ] Run `python -m agent_from_scratch.evals.core_lines`; confirm the core count is unchanged
  (everything in this plan lives under `evals/`, `tests/` and `docs/`).
- [ ] Confirm `evals/bench/manifest.json` is committed and `git status` is clean.
