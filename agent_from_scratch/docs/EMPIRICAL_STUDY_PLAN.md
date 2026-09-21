# Harness empirical-study implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task by task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** apply the five transferable lessons of arXiv 2609.20804 to this 8k-window 7B agent:
staged elision before summarization, an out-of-history plan, domain-agnostic trajectory
metrics with paired comparison, warn-before-stop stuck detection plus post-edit parse checks,
and content/glob search tools.

**Architecture:** no new core concept. `ContextState` gains two view-only fields: `elided`,
a map from raw index to stub content, and `plan`, which is `None` when planning is off. Both
change `messages()` and never `raw`. `Agent._run_turn` gains a cheap elision step at a soft
window share, placed before the existing summary trigger, and routes the per-turn
`update_plan` tool itself. The stuck reminders reuse the existing `[Runtime feedback]` user-message
pattern. Eval metrics stay in `evals/`; a new `evals/trajectory.py` aggregates them. Every new
mechanism is a field in `AgentLimits`, so `TurnResult.settings` records it for ablation.

**Tech stack:** Python 3.14 standard library, `llama-cpp-python` (local Qwen2.5-7B Q4_K_M),
`loguru`, `unittest`. No new dependencies.

**Spec:** [harness-empirical-study.md](harness-empirical-study.md), sections "Order of work"
and "Evaluation". Design decisions this plan must not contradict:
[CONTEXT_STATE_DESIGN.md](CONTEXT_STATE_DESIGN.md), [context-memory.md](context-memory.md).

---

## Global Constraints

- Raw `TurnResult.messages` only ever grows; elision and the plan are **view-only**.
- "Never split a call/result pair." Elision replaces a tool message's content, never removes it.
- Never elide a message the actor has not yet seen (`index < last_sent`), and keep the two most
  recent tool batches verbatim (the paper's two-turn floor).
- Host-owned fields survive elision: `call_id`, `tool_name`, `ok`, `error_code`, `error_message`.
- No recall store, no `recall_event` tool (note, Finding 3).
- The agent may not be a coding agent: trajectory metrics use tool categories, never
  edit/localization concepts.
- Existing defaults stay reproducible: `planning=False` in `AgentLimits`, so library callers,
  the 228 existing tests and the frozen Stage 2B suite are unchanged unless they opt in.
  Elision defaults to on (`elide_ratio=0.6`) because it only acts under pressure.
- Trace `schema_version` stays 5: all record additions are optional fields with defaults.
- Coding rules (CLAUDE.md, STAGE.md): standard library, no helpers for one-time operations,
  no defensive checks for unreachable states, Google-style docstrings for large functions and
  one-liners for small ones, comments only where logic is not self-evident.
- Test command, from the repository root:
  `python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3`
- Size: HEAD is 2,804 physical / 2,257 code core lines. Record `python -m
  agent_from_scratch.evals.core_lines` after each stage; crossing 3,000 is the review trigger in
  context-memory.md, not a reason to compress code.
- Commits: conventional prefixes (`feat:`, `test:`, `docs:`); `docs/` is git-ignored but tracked,
  so new doc files need `git add -f`.

---

## File map

| File | Change |
|---|---|
| `config.py` | `AgentLimits`: `elide_ratio`, `elide_min_chars`, `planning`, `stuck_reminder_calls` |
| `context.py` | `window_share()`; `ContextState.elided`, `.plan`, `view()`, `elide()`, plan reminder in `messages()` |
| `compact.py` | summarizer reads `state.view(...)`; publish candidate carries `elided` and `plan` |
| `trace.py` | `ModelRequest.elided_messages`, `TurnResult.stuck_reminders` |
| `agent.py` | elision step; per-turn `PlanTool`; stuck reminders; REPL enables planning with larger budgets |
| `tools/plan.py` (new) | `PlanTool` (`update_plan`) |
| `tools/files.py` | post-write `.py`/`.json` diagnostics in `_FileTool._replace` |
| `tools/search.py` (new) | `GlobFilesTool`, `GrepTextTool` |
| `tools/register.py` | register the search tools |
| `evals/verify.py` | per-run trajectory fields in `metrics()` |
| `evals/trajectory.py` (new) | `profile()`, `compare()`, CLI |
| `evals/run.py` | `--limits` JSON overrides; writes `trajectory.json` |
| tests | `test_compact.py`, `test_plan.py` (new), `test_trajectory.py` (new), `test_turn.py`, `test_general_files.py`, `test_search.py` (new) |

---

## Stage 1 — T4: elide before summarizing

### Task 1.1: `ContextState.elide()` and the stubbed view

**Files:** Modify `context.py`, `config.py`. Test `tests/test_compact.py`.

**Interfaces — produces:**
- `ContextState.elided: dict[int, str]` (raw index → stub content), default empty.
- `ContextState.view(start: int, end: int | None = None) -> list[message]`.
- `ContextState.elide(min_chars: int) -> bool`: True when a new stub was added.
- `AgentLimits.elide_ratio: float | None = 0.6`, `AgentLimits.elide_min_chars: int = 400`.

- [ ] **Step 1: failing tests** (append to `CompactTests`)

```python
    def bulky(self, name, size=1000):
        return [call(call_id=name).to_message(),
                {"role": "tool", "tool_call_id": name, "content": json.dumps(
                    {"call_id": name, "tool_name": "calculator", "ok": True, "output": "x" * size,
                     "error_code": None, "error_message": None})}]

    def test_elide_stubs_bulky_outputs_outside_the_two_latest_batches(self):
        raw = [message("system", "rules"), message("user", "go"),
               *self.bulky("a"), *self.bulky("b"), *self.bulky("c")]
        original = deepcopy(raw)
        state = ContextState(raw, turn_start=1, last_sent=len(raw))
        self.assertTrue(state.elide(400))
        self.assertEqual(set(state.elided), {3})
        stub = json.loads(state.messages()[3]["content"])
        self.assertEqual((stub["call_id"], stub["ok"]), ("a", True))
        self.assertIn("elided: 1002 chars", stub["output"])
        self.assertEqual(state.messages()[5:], raw[5:])  # b and c stay verbatim
        self.assertEqual(raw, original)
        self.assertFalse(state.elide(400))  # nothing new

    def test_elide_skips_unseen_and_small_outputs(self):
        raw = [message("system", "rules"), message("user", "go"),
               *self.bulky("a"), *self.bulky("b"), *self.bulky("c")]
        self.assertFalse(ContextState(raw, turn_start=1, last_sent=3).elide(400))
        self.assertFalse(ContextState(raw, turn_start=1, last_sent=len(raw)).elide(5000))
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_compact -q`; expect
  `AttributeError: 'ContextState' object has no attribute 'elide'`.

- [ ] **Step 3: implementation.** In `config.py` add to `AgentLimits`:

```python
    elide_ratio: float | None = 0.6  # stub old tool outputs at this share of the usable window; None disables
    elide_min_chars: int = 400  # only outputs longer than this are elided
```

In `context.py` import `field` and `json`, add to `ContextState` after `instructions`:

```python
    # View-only stubs for bulky tool outputs, keyed by raw index; raw keeps the originals.
    elided: dict[int, str] = field(default_factory=dict)
```

and the methods, with `messages()` switched to `view()`:

```python
    def view(self, start: int, end: int | None = None) -> list[ChatCompletionRequestMessage]:
        """Return raw[start:end] as the actor sees it, with elided outputs replaced by stubs."""
        end = len(self.raw) if end is None else end
        return [{**self.raw[i], "content": self.elided[i]} if i in self.elided else self.raw[i]
                for i in range(start, end)]

    def elide(self, min_chars: int) -> bool:
        """Stub bulky tool outputs the actor has seen, outside the two most recent batches.

        Elision is irreversible in the view and costs no model call; the stub says to
        re-read or re-run. Host-owned fields (call ID, status, error) are kept.

        Returns:
            bool: True when at least one more output is now shown as a stub.
        """
        starts = [i for i, m in enumerate(self.raw) if m.get("tool_calls")]
        cutoff = min(self.last_sent, starts[-2]) if len(starts) >= 2 else 0
        added = False
        for index in range(self.covered, cutoff):
            message = self.raw[index]
            if (message["role"] != "tool" or index in self.elided
                    or len(message["content"]) <= min_chars):
                continue
            result = json.loads(message["content"])
            output = json.dumps(result.get("output"), ensure_ascii=False)
            if len(output) <= min_chars:
                continue
            result["output"] = (f"[tool output elided: {len(output)} chars. "
                                "Re-read or re-run to get it again.]")
            self.elided[index] = json.dumps(result)
            added = True
        return added
```

`messages()`: `history = self.view(1, self.turn_start)`, `current = self.view(self.turn_start)`,
and in the summary branch `current = [*pinned, *self.view(self.covered)]`.

- [ ] **Step 4:** rerun; expect OK. Run the full suite; expect 230 OK.
- [ ] **Step 5:** commit `feat: elide bulky tool outputs in the actor view without touching raw`.

### Task 1.2: trigger elision at the soft share; summarizer and publish see the same view

**Files:** Modify `context.py`, `agent.py`, `compact.py`, `trace.py`. Test `tests/test_compact.py`.

**Interfaces — produces:** `window_share(budget: dict | None) -> float | None`;
`ModelRequest.elided_messages: int = 0`.

- [ ] **Step 1: failing tests**

```python
    def test_soft_pressure_elides_without_a_summary_call(self):
        def measure(messages, schemas, **kwargs):
            prompt = 3000 if any("elided:" in m.get("content", "") for m in messages) else 5000
            return {"count_method": "exact", "prompt_tokens": prompt, "window_tokens": 8000,
                    "response_reserve": 512, "remaining_tokens": 8000 - 512 - prompt}
        self.model.measure_context.side_effect = measure
        self.model.generate.return_value = answer("Done")
        history = [message("user", "go"), *self.bulky("a"), *self.bulky("b"), *self.bulky("c"),
                   message("assistant", "ok")]
        result = Agent(self.model).run_turn("next", history)
        self.assertEqual([q.purpose for q in result.model_requests], ["agent"])
        self.assertEqual(result.model_requests[0].elided_messages, 1)
        self.assertEqual(result.model_requests[0].budget["prompt_tokens"], 3000)
        disabled = Agent(self.model, limits=replace(AgentLimits(), elide_ratio=None))
        self.assertEqual(disabled.run_turn("next", history).model_requests[0].elided_messages, 0)

    def test_summarizer_reads_the_elided_view(self):
        # Three batches: the cut policy keeps the latest two, so only batch a is summarized.
        raw = [message("system", "rules"), message("user", "old request"),
               *self.bulky("a"), *self.bulky("b"), *self.bulky("c"), message("user", "Use 7, not 6.")]
        self.state = ContextState(raw, turn_start=8, last_sent=8)
        self.assertTrue(self.state.elide(400))
        self.assertTrue(self.compact())
        sent = json.loads(self.requests[0].input_messages[1]["content"])["messages"]
        self.assertIn("elided:", sent[2]["content"])
```

- [ ] **Step 2:** run; expect `AttributeError` on `elided_messages` and a raw (unstubbed) summarizer input.
- [ ] **Step 3: implementation.**

`context.py`, next to `context_blocker`:

```python
def window_share(budget: dict | None) -> float | None:
    """Fraction of the usable window (window minus output reserve) an exact measurement fills."""
    if not budget or budget.get("count_method") != "exact" or budget.get("response_reserve") is None:
        return None
    return budget["prompt_tokens"] / (budget["window_tokens"] - budget["response_reserve"])
```

`trace.py` `ModelRequest`: `elided_messages: int = 0  # tool outputs shown as stubs in this view`.

`agent.py` `_run_turn`, right after the first `budget = self.llm.measure_context(...)`:

```python
                # Cheap first: stubs cost no model call, so they run before the summary trigger.
                share = window_share(budget)
                if (self.limits.elide_ratio is not None and share is not None
                        and share >= self.limits.elide_ratio
                        and state.elide(self.limits.elide_min_chars)):
                    prepared_messages = state.messages()
                    budget = self.llm.measure_context(prepared_messages, schemas)
```

and pass `elided_messages=len(state.elided)` to the actor's `ModelRequest(...)`.

`compact.py`: in `_summarize` use `"messages": state.view(state.covered, boundary)`; in
`_publish` build the candidate with `elided=state.elided`.

- [ ] **Step 4:** rerun the two tests and the full suite; expect 232 OK.
- [ ] **Step 5:** commit `feat: stage elision at 0.6 of the usable window before summarizing`.
- [ ] **Step 6:** record `core_lines` in the stage log (end of this file).

---

## Stage 2 — planning outside the transcript

### Task 2.1: `PlanTool` and the plan reminder in the view

**Files:** Create `tools/plan.py`; modify `context.py`. Test `tests/test_plan.py` (new).

**Interfaces — produces:**
- `PlanTool().invoke({"plan": [{"content": str, "status": "pending"|"in_progress"|"completed"}]})`;
  `PlanTool.render() -> str` lines `"[status] content"`.
- `ContextState.plan: str | None = None`: `None` off, `""` no plan yet, text = current plan;
  `messages()` appends one trailing `system` reminder when not `None`.

- [ ] **Step 1: failing tests** (`tests/test_plan.py`)

```python
"""The plan is re-injected every request and never enters raw history."""

import unittest

from agent_from_scratch.context import ContextState
from agent_from_scratch.tools.plan import PlanTool


class PlanTests(unittest.TestCase):
    def test_tool_replaces_the_whole_plan_and_renders_it(self):
        tool = PlanTool()
        result = tool.invoke({"plan": [{"content": "Read config", "status": "in_progress"},
                                       {"content": "Report", "status": "pending"}]})
        self.assertTrue(result.ok)
        self.assertEqual(tool.render(), "[in_progress] Read config\n[pending] Report")
        self.assertFalse(tool.invoke({"plan": [{"content": "x", "status": "done"}]}).ok)

    def test_view_carries_one_trailing_reminder_and_raw_is_unchanged(self):
        raw = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
        state = ContextState(list(raw), turn_start=1, last_sent=1)
        self.assertEqual(state.messages(), raw)
        state.plan = ""
        self.assertIn("not created a plan", state.messages()[-1]["content"])
        state.plan = "[pending] Report"
        self.assertEqual(state.messages()[-1]["role"], "system")
        self.assertIn("[pending] Report", state.messages()[-1]["content"])
        self.assertEqual(len(state.messages()), 3)
        self.assertEqual(state.raw, raw)
```

- [ ] **Step 2:** run `python -m unittest agent_from_scratch.tests.test_plan -q`; expect `ModuleNotFoundError`.
- [ ] **Step 3: implementation.** `tools/plan.py`:

```python
"""A todo list the harness shows back to the model before every request."""

from .base import Tool


class PlanTool(Tool):
    name = "update_plan"
    description = (
        "Create or replace your task plan (a todo list). The harness shows the current plan "
        "back to you before every request. For any task of about 3+ steps, call this FIRST. "
        "Pass the COMPLETE list every time; it replaces the previous one. Keep exactly one "
        "task in_progress and mark a task completed as soon as it is done. Skip planning for a "
        "single trivial step or a purely informational request. Touches no files."
    )
    parameters = {
        "type": "object", "properties": {"plan": {"type": "array", "items": {
            "type": "object", "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            }, "required": ["content", "status"], "additionalProperties": False}}},
        "required": ["plan"], "additionalProperties": False,
    }

    def __init__(self):
        self.plan: list[dict] = []

    def execute(self, plan: list[dict]) -> dict:
        self.plan = plan
        return {"tasks": len(plan), "in_progress": sum(t["status"] == "in_progress" for t in plan)}

    def render(self) -> str:
        """Return the plan as one '[status] content' line per task."""
        return "\n".join(f"[{task['status']}] {task['content']}" for task in self.plan)
```

`context.py` `ContextState`: field `plan: str | None = None  # None: planning off; "": no plan yet`,
and in `messages()` the final return becomes:

```python
        # The plan lives outside raw: one fresh copy per request, so it never accumulates
        # and never needs summarizing.
        if self.plan is not None:
            current = [*current, {"role": "system", "content": (
                "[Current plan; update it with update_plan as you progress]\n" + self.plan
                if self.plan else
                "[Planning] You have not created a plan yet. Unless this is a single trivial "
                "step or a purely informational request, call update_plan first.")}]
        return build_messages(instructions=rules, history=history, current_turn=current)
```

`compact.py` `_publish`: the candidate also takes `plan=state.plan`.

- [ ] **Step 4:** rerun; expect OK; full suite 234 OK.
- [ ] **Step 5:** commit `feat: add update_plan and re-inject the plan outside raw history`.

### Task 2.2: wire planning into the turn, and budget for it in the REPL

**Files:** Modify `config.py`, `agent.py`. Test `tests/test_plan.py`.

**Interfaces — consumes:** `PlanTool`, `ContextState.plan`. **Produces:** `AgentLimits.planning: bool = False`.

- [ ] **Step 1: failing tests** (append; reuse `TurnTests` setup via subclassing)

```python
from dataclasses import replace
from unittest.mock import patch
from agent_from_scratch.agent import Agent
from agent_from_scratch.config import AgentLimits
from agent_from_scratch.llm import LLM
from agent_from_scratch.tests.test_turn import answer


def plan_call(status="in_progress"):
    return LLM.parse_response({"choices": [{"message": {"role": "assistant", "content":
        '<tool_call>{"name": "update_plan", "arguments": {"plan": '
        f'[{{"content": "Add numbers", "status": "{status}"}}]}}}}</tool_call>'}}]})


class PlanTurnTests(unittest.TestCase):  # setUp/script as in TurnTests
    def test_planning_turn_shows_the_plan_and_never_dispatches_it_to_the_registry(self):
        self.agent.limits = replace(AgentLimits(), planning=True)
        self.script(plan_call(), answer("4"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("2+2")
        execute.assert_not_called()
        self.assertIn("update_plan", result.model_requests[0].tools)
        self.assertIn("not created a plan", self.seen[0][-1]["content"])
        self.assertIn("[in_progress] Add numbers", self.seen[1][-1]["content"])
        self.assertEqual(sum(m["role"] == "system" for m in self.seen[1]), 2)
        self.assertFalse(any(m["role"] == "system" for m in result.messages[1:]))

    def test_planning_off_adds_no_tool_or_reminder(self):
        self.script(answer("4"))
        result = self.agent.run_turn("2+2")
        self.assertNotIn("update_plan", result.model_requests[0].tools)
        self.assertEqual(self.seen[0][-1]["role"], "user")
```

Do not subclass `test_turn.TurnTests`: unittest discovery would rerun every inherited test
under this module. `PlanTurnTests(unittest.TestCase)` carries its own copy of the
`TurnTests.setUp` and `script` fixtures (scripted `Mock(spec=LLM)`, patched `PROMPTS_DIR`).

- [ ] **Step 2:** run; expect failure: `planning` is not an `AgentLimits` field.
- [ ] **Step 3: implementation.** `config.py`: `planning: bool = False  # update_plan tool + per-request plan reminder`.

`agent.py` `run_turn`, after creating `state`: `state.plan = "" if self.limits.planning else None`.
`_run_turn`, before the loop: `planner = PlanTool() if state.plan is not None else None`;
in the loop, `schemas = self.registry.schemas()` becomes

```python
                schemas = self.registry.schemas()
                if planner is not None:
                    schemas = {**schemas, planner.name: planner.to_schema()}
```

and tool dispatch becomes

```python
                    if planner is not None and call.name == planner.name:
                        # A harness component, not a workspace action: never the registry's.
                        tool_result = planner.invoke(call.arguments, call.call_id)
                        if tool_result.ok:
                            state.plan = planner.render()
                    else:
                        tool_result = self.execute_tool(call.name, call.arguments, call.call_id)
```

REPL entry point (`__main__`): `limits=AgentLimits(planning=True, max_iterations=30, max_tool_calls=60)`
— planning's measured cost for weak models (note, Finding 4) would otherwise hit the 20/40 ceilings.

- [ ] **Step 4:** rerun; full suite 236 OK.
- [ ] **Step 5:** commit `feat: run update_plan as a per-turn harness tool when planning is on`.
- [ ] **Step 6:** record `core_lines`.

---

## Stage 3 — trajectory metrics and paired comparison (eval only, no core lines)

### Task 3.1: per-run trajectory fields

**Files:** Modify `evals/verify.py`; `trace.py` gains `TurnResult.stuck_reminders: int = 0`
here (Stage 4 increments it). Test `tests/test_trajectory.py` (new). Tasks 3.1 and 3.2 share
one test module and land in one commit.

**Interfaces — produces:** `metrics(result)` adds `peak_context_ratio: float | None`,
`elided_messages: int`, `plan_updates: int`, `stuck_reminders: int`, `actions: list[str]`
(one label per non-blocked actor request, from `ACTION_KINDS`/priority below).

Action label per actor request: a non-completed request is `"error"`; no calls is `"answer"`;
otherwise the highest-priority category among its calls, priority `modify > execute > explore > plan > other`,
with categories: explore = `list_files, read_file, glob_files, grep_text, web_fetch, web_search`;
modify = `write_file, edit_file`; execute = `shell, calculator`; plan = `update_plan`.

- [ ] **Step 1: failing test**

```python
"""Trajectory metrics describe run shape with tool categories, not coding stages."""

import unittest
from unittest.mock import Mock

from agent_from_scratch.agent import Agent
from agent_from_scratch.evals.trajectory import compare, profile
from agent_from_scratch.evals.verify import metrics
from agent_from_scratch.llm import LLM, ResponseError, ResponseErrorCode
from agent_from_scratch.tests.test_turn import answer, call


class TrajectoryTests(unittest.TestCase):
    def run_script(self, *responses):
        model = Mock(spec=LLM)
        model.settings.return_value = {}
        model.read_usage.side_effect = LLM.read_usage
        model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 3744,
            "window_tokens": 8000, "response_reserve": 512, "remaining_tokens": 3744}
        model.generate.side_effect = list(responses)
        return Agent(model).run_turn("2+2")

    def test_metrics_label_each_actor_request(self):
        record = metrics(self.run_script(ResponseError(ResponseErrorCode.INVALID_RESPONSE),
                                         call(), answer("4")))
        self.assertEqual(record["actions"], ["error", "execute", "answer"])
        self.assertEqual(record["peak_context_ratio"], 0.5)
        self.assertEqual((record["elided_messages"], record["plan_updates"],
                          record["stuck_reminders"]), (0, 0, 0))
```

- [ ] **Step 2:** run; expect `ModuleNotFoundError: evals.trajectory` until Task 3.2 exists,
  then `KeyError: 'actions'`.
- [ ] **Step 3: implementation** in `evals/verify.py` (`from ..context import window_share`):

```python
ACTION_KINDS = {"list_files": "explore", "read_file": "explore", "glob_files": "explore",
                "grep_text": "explore", "web_fetch": "explore", "web_search": "explore",
                "write_file": "modify", "edit_file": "modify", "shell": "execute",
                "calculator": "execute", "update_plan": "plan"}
ACTION_PRIORITY = ("modify", "execute", "explore", "plan", "other")
```

inside `metrics`:

```python
    names = {call["id"]: call["function"]["name"]
             for m in result.messages for call in m.get("tool_calls", [])}
    actions = []
    for request in actors:
        kinds = {ACTION_KINDS.get(names.get(call_id), "other") for call_id in request.call_ids}
        actions.append("error" if request.status != "completed" else
                       next((k for k in ACTION_PRIORITY if k in kinds), "answer"))
    shares = [s for q in actors if (s := window_share(q.budget)) is not None]
```

and the returned dict gains:

```python
        "peak_context_ratio": round(max(shares), 4) if shares else None,
        "elided_messages": max((q.elided_messages for q in actors), default=0),
        "plan_updates": sum(r["tool_name"] == "update_plan" and r["ok"] for r in rows),
        "stuck_reminders": result.stuck_reminders,
        "actions": actions,
```

### Task 3.2: `profile`, `compare`, and eval-run integration

**Files:** Create `evals/trajectory.py`; modify `evals/run.py`. Test `tests/test_trajectory.py`.

**Interfaces — produces:** `profile(records: list[dict]) -> dict`;
`compare(a: list[dict], b: list[dict]) -> dict` with `both`, `only_a`, `only_b`, `neither`,
`mcnemar_p`; CLI `python -m agent_from_scratch.evals.trajectory RUN [OTHER_RUN]`;
`evals.run --limits '{"planning": true}'`.

- [ ] **Step 1: failing tests** (append)

```python
    def test_profile_reports_survival_mix_and_overflow(self):
        records = [{"id": "a", "passed": True, "stop_reason": "final_response",
                    "actions": ["explore", "answer"], "peak_context_ratio": 0.4,
                    "elided_messages": 1, "compact_requests": 0, "plan_updates": 0,
                    "stuck_reminders": 0, "model_requests": 2, "usage": {"total_tokens": 10}},
                   {"id": "b", "passed": False, "stop_reason": "context_limit",
                    "actions": ["explore"], "peak_context_ratio": 0.9,
                    "elided_messages": 3, "compact_requests": 2, "plan_updates": 0,
                    "stuck_reminders": 1, "model_requests": 3, "usage": {"total_tokens": None}}]
        summary = profile(records)
        self.assertEqual(summary["context_limit_rate"], 0.5)
        self.assertEqual(summary["survival"][0], {"request": 1, "active": 1.0, "mix": {"explore": 2}})
        self.assertEqual(summary["survival"][1], {"request": 2, "active": 0.5, "mix": {"answer": 1}})
        self.assertEqual(summary["mean"]["elided_messages"], 2)
        self.assertIsNone(summary["mean"]["total_tokens"])

    def test_compare_pairs_tasks_and_uses_the_exact_mcnemar_test(self):
        a = [{"id": str(i), "passed": False} for i in range(6)]
        b = [{"id": str(i), "passed": i < 5} for i in range(6)]
        result = compare(a, b)
        self.assertEqual((result["both"], result["only_a"], result["only_b"], result["neither"]),
                         (0, 0, 5, 1))
        self.assertAlmostEqual(result["mcnemar_p"], 0.0625)
```

- [ ] **Step 2:** run; expect failures on the missing functions.
- [ ] **Step 3: implementation** `evals/trajectory.py`:

```python
"""Domain-agnostic trajectory profile and paired comparison of eval runs.

Usage: python -m agent_from_scratch.evals.trajectory RUN_DIR [OTHER_RUN_DIR]
Reads each directory's results.json. With two runs, pairs tasks by ID (second minus first).
"""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
from statistics import mean, median

MEANS = ("elided_messages", "compact_requests", "plan_updates", "stuck_reminders", "model_requests")


def profile(records: list[dict]) -> dict:
    """Summarize run shape: stop reasons, survival by actor request, mechanism use and cost.

    Args:
        records: per-task records from `evals.run`, each carrying `metrics()` fields; an eval
            run always has at least one.

    Returns:
        dict: pass count, stop-reason counts, overflow (`context_limit`) rate, median actor
            requests and peak window share, per-run means, and one survival row per request
            index with the share of runs still active and their action mix.
    """
    lengths = [len(r["actions"]) for r in records]
    peaks = [r["peak_context_ratio"] for r in records if r["peak_context_ratio"] is not None]
    tokens = [r["usage"]["total_tokens"] for r in records]
    survival = []
    for index in range(max(lengths, default=0)):
        active = [r["actions"][index] for r in records if len(r["actions"]) > index]
        survival.append({"request": index + 1, "active": round(len(active) / len(records), 4),
                         "mix": dict(Counter(active))})
    stops = Counter(r["stop_reason"] for r in records)
    return {
        "runs": len(records), "passed": sum(r["passed"] for r in records),
        "stop_reasons": dict(stops),
        "context_limit_rate": round(stops["context_limit"] / len(records), 4),
        "median_actor_requests": median(lengths),
        "median_peak_context_ratio": median(peaks) if peaks else None,
        # Unknown usage in any run leaves the mean unknown rather than silently smaller.
        "mean": {**{key: round(mean(r[key] for r in records), 4) for key in MEANS},
                 "total_tokens": round(mean(tokens), 1) if None not in tokens else None},
        "survival": survival,
    }


def compare(first: list[dict], second: list[dict]) -> dict:
    """Pair two runs by task ID; exact two-sided McNemar p over the discordant tasks."""
    a = {r["id"]: r["passed"] for r in first}
    b = {r["id"]: r["passed"] for r in second}
    shared = sorted(a.keys() & b.keys())
    only_a = sum(a[t] and not b[t] for t in shared)
    only_b = sum(b[t] and not a[t] for t in shared)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"tasks": len(shared), "both": sum(a[t] and b[t] for t in shared),
            "only_a": only_a, "only_b": only_b,
            "neither": sum(not a[t] and not b[t] for t in shared), "mcnemar_p": round(p, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="One or two eval output directories")
    args = parser.parse_args()
    if len(args.runs) > 2:
        parser.error("Give one run to profile or two runs to compare.")
    records = [json.loads((run / "results.json").read_text()) for run in args.runs]
    report = {"profiles": {str(run): profile(r) for run, r in zip(args.runs, records)}}
    if len(records) == 2:
        report["paired"] = compare(*records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

`evals/run.py`: argument
`parser.add_argument("--limits", default="{}", help='JSON AgentLimits overrides, e.g. \'{"planning": true}\'')`;
after `args = parser.parse_args()` validate with
`overrides = json.loads(args.limits)` and `replace(AgentLimits(), **overrides)` inside
`try/except (ValueError, TypeError) → parser.error(...)`; record `"limit_overrides": overrides`
in `metadata.json`; `run_case(..., overrides=overrides)` applies
`replace(agent.limits, **{**overrides, "max_iterations": task["max_iterations"]})` (the task's
frozen budget wins); after the loop, `save(out / "trajectory.json", profile(records))`.

- [ ] **Step 4:** rerun; full suite green.
- [ ] **Step 5:** commit `feat: profile eval trajectories and pair runs with an exact McNemar test`.

### Task 3.3: real-model ablation on the dev suite

Three fresh runs of the 17 dev tasks, then profiles and pairwise comparisons:

```sh
python -m agent_from_scratch.evals.run --output outputs/study-t3 --limits '{"elide_ratio": null}'
python -m agent_from_scratch.evals.run --output outputs/study-t4
python -m agent_from_scratch.evals.run --output outputs/study-t4-plan --limits '{"planning": true}'
python -m agent_from_scratch.evals.trajectory outputs/study-t3 outputs/study-t4
python -m agent_from_scratch.evals.trajectory outputs/study-t4 outputs/study-t4-plan
```

Expected and acceptable: T3 vs T4 may be **identical** — this suite reads 1 KiB windows and
may never cross 0.6 of 5,952 tokens; that would be recorded as "no pressure reached", not as
evidence either way. Planning runs under the suite's frozen per-task `max_iterations` (≤ 20),
so it measures planning's cost at fixed budget. Record results in the stage log; no
significance claims at n = 17.

---

## Stage 4 — stuck detection and post-edit diagnostics

### Task 4.1: warn before stopping; detect identical successful calls

**Files:** Modify `agent.py`, `config.py`, `trace.py` (field added in 3.1). Test `tests/test_turn.py`.

Behaviour (thresholds scaled to a 20-call budget; paper: remind 5, stop 8 at 300 steps):
- identical **failing** calls: after the `max_same_failures - 1`-th (default 2nd), append one
  `[Runtime feedback]` reminder; the existing stop at `max_same_failures` (3) is unchanged;
- identical **successful** calls (same name and canonical arguments, consecutive across
  batches): at exactly `stuck_reminder_calls` (default 3), append one reminder; no stop;
- a reminder is appended after the batch's results, so no call/result pair is split; each
  increments `TurnResult.stuck_reminders`.

- [ ] **Step 1: failing tests** (append to `TurnTests`)

```python
    def test_identical_successful_calls_get_one_reminder(self):
        self.script(call(), call(), call(), call(), answer("4"))
        result = self.agent.run_turn("2+2")
        reminders = [m for m in result.messages if "already have this result" in m.get("content", "")]
        self.assertEqual(len(reminders), 1)
        self.assertEqual(result.stuck_reminders, 1)
        self.assertEqual(result.messages.index(reminders[0]), 8)  # after the 3rd call's result

    def test_identical_failures_warn_once_before_the_stop(self):
        self.script(call(left="bad"), call(left="bad"), call(left="bad"))
        result = self.agent.run_turn("test")
        self.assertIn("keeps failing", self.seen[2][-1]["content"])
        self.assertEqual((result.stop_reason, result.stuck_reminders), ("no_progress", 1))
```

- [ ] **Step 2:** run; expect `stuck_reminders == 0` failures.
- [ ] **Step 3: implementation.** `config.py`: `stuck_reminder_calls: int = 3  # remind after this many identical successful calls`.
`agent.py` module constants:

```python
REPEATED_CALL = ("[Runtime feedback] You called {tool} with the same arguments {count} times "
                 "and already have this result. Move on to the next concrete step.")
REPEATED_FAILURE = ("[Runtime feedback] You called {tool} with the same arguments {count} times "
                    "and it keeps failing the same way. Repeating it will not work: read the "
                    "error, then try a different call or reconsider the approach.")
```

In `_run_turn`: locals `streak_key, streak = None, 0` next to `last_failure`; per executed call
compute `key = (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False))`,
`streak = streak + 1 if key == streak_key else 1; streak_key = key`; reuse `key` in the existing
`repeated_failure(key + (tool_result.error_code,))`. Set `reminder = None` before the batch; on
success `if streak == self.limits.stuck_reminder_calls: reminder = REPEATED_CALL.format(...)`; on
a failure that did not stop, `if failure_count == self.limits.max_same_failures - 1: reminder =
REPEATED_FAILURE.format(...)`. After the batch loop:

```python
                if reminder is not None:
                    raw_messages.append({"role": "user", "content": reminder})
                    result.stuck_reminders += 1
```

- [ ] **Step 4:** rerun and full suite; existing stop-at-3 tests stay green.
- [ ] **Step 5:** commit `feat: remind before a stuck stop and flag repeated successful calls`.

### Task 4.2: post-write parse diagnostics for `.py` and `.json`

**Files:** Modify `tools/files.py`. Test `tests/test_general_files.py`.

- [ ] **Step 1: failing test**

```python
    def test_writes_report_parse_diagnostics_without_failing(self):
        with TemporaryDirectory() as directory:
            write, edit = WriteFileTool(directory), EditFileTool(directory)
            bad = write.invoke({"path": "c.json", "content": '{"a": 1,}'})
            self.assertTrue(bad.ok)
            self.assertIn("JSONDecodeError", bad.output["diagnostics"])
            good = write.invoke({"path": "d.json", "content": '{"a": 1}'})
            self.assertNotIn("diagnostics", good.output)
            write.invoke({"path": "m.py", "content": "x = 1\n"})
            broken = edit.invoke({"path": "m.py", "old_text": "x = 1", "new_text": "x = ("})
            self.assertIn("SyntaxError", broken.output["diagnostics"])
```

- [ ] **Step 2:** run; expect `KeyError: 'diagnostics'`.
- [ ] **Step 3: implementation** in `_FileTool._replace`, replacing the final `return`:

```python
        result = {"path": self._display(target), "bytes_written": len(data), "changed": True,
                  "created": before is None, "version": file_version(target.stat())}
        # Parse-only checks surface a broken file now instead of on the next run.
        try:
            if target.suffix.lower() == ".json":
                json.loads(data)
            elif target.suffix.lower() == ".py":
                compile(data, str(target), "exec")
        except (ValueError, SyntaxError) as error:
            result["diagnostics"] = f"{type(error).__name__}: {error}"
        return result
```

(`import json`; append " Results include diagnostics when a written .py or .json file does not
parse." to the `write_file` and `edit_file` descriptions.)

- [ ] **Step 4:** rerun and full suite.
- [ ] **Step 5:** commit `feat: report parse diagnostics after writing Python or JSON files`.
- [ ] **Step 6:** record `core_lines`.

---

## Stage 5 — content and glob search

### Task 5.1: `glob_files` and `grep_text`

**Files:** Create `tools/search.py`; modify `tools/register.py`. Test `tests/test_search.py` (new).

**Interfaces — produces:** `GlobFilesTool(workspace, **_FileTool kwargs)` →
`{"path", "matches": [relative paths], "truncated"}`; `GrepTextTool(workspace, ...)` →
`{"path", "matches": ["path:line: text"], "truncated"}`. Both skip `ListFilesTool._IGNORED`
directories and honour `restrict_to_workspace` for every match.

- [ ] **Step 1: failing tests**

```python
"""Search tools find files by name pattern and by content, skipping build/cache trees."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.tools.register import default_registry
from agent_from_scratch.tools.search import GlobFilesTool, GrepTextTool


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        (self.root / "sub").mkdir()
        (self.root / "node_modules").mkdir()
        (self.root / "a.json").write_text('{"port": 80}\n')
        (self.root / "sub" / "b.json").write_text('{\n  "Port": 8080\n}\n')
        (self.root / "node_modules" / "c.json").write_text('{"port": 1}\n')
        (self.root / "blob.bin").write_bytes(b"port\0\0")

    def test_glob_lists_matching_files_outside_ignored_trees(self):
        result = GlobFilesTool(self.root).invoke({"pattern": "**/*.json"})
        self.assertEqual(result.output["matches"], ["a.json", "sub/b.json"])
        limited = GlobFilesTool(self.root).invoke({"pattern": "**/*.json", "max_matches": 1})
        self.assertEqual((limited.output["matches"], limited.output["truncated"]), (["a.json"], True))

    def test_grep_reports_path_and_line_and_skips_binary(self):
        tool = GrepTextTool(self.root)
        self.assertEqual(tool.invoke({"query": "port"}).output["matches"], ['a.json:1: {"port": 80}'])
        found = tool.invoke({"query": "port", "case_sensitive": False, "include": "*.json"})
        self.assertEqual(found.output["matches"], ['a.json:1: {"port": 80}', 'sub/b.json:2:   "Port": 8080'])
        self.assertEqual(tool.invoke({"query": "("}).error_code, "invalid_arguments")

    def test_workspace_confinement_and_registration(self):
        tool = GrepTextTool(self.root / "sub", restrict_to_workspace=True)
        self.assertEqual(tool.invoke({"query": "port", "path": ".."}).error_code, "denied")
        self.assertTrue({"glob_files", "grep_text"} <= set(default_registry.schemas()))
```

- [ ] **Step 2:** run; expect `ModuleNotFoundError`.
- [ ] **Step 3: implementation** `tools/search.py`:

```python
"""Find files by glob pattern or by regular expression over their contents."""

from pathlib import Path
import re

from .base import ToolErrorCode, ToolExecutionError
from .files import ListFilesTool, _FileTool


class _SearchTool(_FileTool):
    def _files(self, root: Path, pattern: str):
        """Yield regular files under root matching pattern, outside ignored and escaped paths."""
        for item in sorted(root.glob(pattern)):
            if (item.is_file() and not ListFilesTool._IGNORED & set(item.relative_to(root).parts)
                    and (not self.restrict_to_workspace
                         or item.resolve().is_relative_to(self.workspace))):
                yield item

    def _directory(self, path: str) -> Path:
        root = self._resolve(path)
        if not root.is_dir():
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, f"Not an existing directory: {root}.")
        return root


class GlobFilesTool(_SearchTool):
    name = "glob_files"
    description = ("Find files whose path matches a glob pattern such as '**/*.json' or 'src/*.py'. "
                   "Returns sorted paths relative to path. Use grep_text to search file contents.")
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string"}, "path": {"type": "string", "description": "Directory; default '.'."},
        "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000},
    }, "required": ["pattern"], "additionalProperties": False}

    def execute(self, pattern: str, path: str = ".", max_matches: int = 200) -> dict:
        root = self._directory(path)
        matches = [item.relative_to(root).as_posix() for item in self._files(root, pattern)]
        return {"path": self._display(root), "matches": matches[:max_matches],
                "truncated": len(matches) > max_matches}


class GrepTextTool(_SearchTool):
    name = "grep_text"
    description = ("Search file contents with a Python regular expression. Returns 'path:line: text' "
                   "matches under a directory. include filters file names by glob, e.g. '*.json'. "
                   "Binary files and common build/cache directories are skipped.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"}, "path": {"type": "string", "description": "Directory; default '.'."},
        "include": {"type": ["string", "null"]}, "case_sensitive": {"type": "boolean"},
        "max_matches": {"type": "integer", "minimum": 1, "maximum": 200},
    }, "required": ["query"], "additionalProperties": False}

    def execute(self, query: str, path: str = ".", include: str | None = None,
                case_sensitive: bool = True, max_matches: int = 30) -> dict:
        try:
            regex = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error as error:
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, f"Invalid regular expression: {error}")
        root = self._directory(path)
        matches = []
        for item in self._files(root, f"**/{include or '*'}"):
            data = item.read_bytes()[:self.max_file_bytes]
            if b"\0" in data[:8192]:
                continue
            for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{item.relative_to(root).as_posix()}:{number}: {line[:200]}")
                    if len(matches) > max_matches:
                        return {"path": self._display(root), "matches": matches[:max_matches],
                                "truncated": True}
        return {"path": self._display(root), "matches": matches, "truncated": False}
```

`tools/register.py`: import and add `GlobFilesTool(workspace=workspace)` and
`GrepTextTool(workspace=workspace)` after `ListFilesTool`.

- [ ] **Step 4:** rerun and full suite.
- [ ] **Step 5:** commit `feat: add glob_files and grep_text search tools`.
- [ ] **Step 6:** record `core_lines`; update the note's order-of-work section and
  `evals/README.md` tool list; commit `docs: record the empirical-study stages and evidence`.

---

## Stage log

Filled in during execution: tests, core line counts, commits and real-model evidence per stage.
