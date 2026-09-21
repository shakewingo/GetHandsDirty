# Stage 3B remainder and Stage 3C implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task by task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** finish Stage 3B items 2-4 (rule reload at the compact boundary, continuity
guarantees, bounded per-attempt recovery with its own summarizer budget) and all of
Stage 3C (versioned summary checkpoint, restart replay, trace/eval export), so a long
conversation compacts twice, survives a restart, and reports truthfully what it did.

**Architecture:** no new module. `ContextState` gains one optional field (an instruction
override) so a compact boundary can publish reloaded rules without editing raw evidence.
`Compactor.attempt()` becomes a bounded loop of at most two summary calls over one cut,
each with its own output reserve. `SessionStore` gains a second record kind — a versioned
checkpoint written *after* the raw messages it references — and `Agent.run_turn` accepts
one on entry so the first prepared view of a restarted session is already summary +
uncovered suffix. Trace records stay at `schema_version` 5 (all additions are optional
fields, matching the precedent set when `budget` was first added at schema 3).

**Tech stack:** Python 3.14 standard library, `llama-cpp-python` (local Qwen2.5-7B
Q4_K_M), `loguru`, `unittest`, Pyright. No new dependencies.

**Spec:** [STAGE.md](STAGE.md) 3B items 2-4 and 3C items 1-3, plus the gate paragraph
that closes 3C. Design decisions this plan must not contradict:
[CONTEXT_STATE_DESIGN.md](CONTEXT_STATE_DESIGN.md) and [context-memory.md](context-memory.md).

---

## Global Constraints

Copied verbatim from the spec; every task's requirements implicitly include these.

- "Never split a call/result pair."
- "Keep run ID, iteration/failure counters, registry, and actual workspace state intact."
- "Do not replay tools or treat summarized file state as current without rereading when needed."
- "At most two summary calls per attempt, four per run, and one attempt at an unchanged boundary."
- "Count summary requests in total cost/request limits."
- "Repeated compaction cannot reset the task budget."
- "Persist raw new-turn messages before publishing a checkpoint referencing them."
- "Save an explicit raw turn delta, not a slice of compacted messages."
- "Invalid/stale checkpoints fall back to raw history and its budget check; `/reset` also
  clears the session checkpoint."
- "Incomplete turns stay evidence only; exact mid-tool resume remains outside scope."
- "Preserve old readers and update eval export."
- Raw `TurnResult.messages` only ever grows; compaction publishes a **disposable view**.
- Coding rules (CLAUDE.md): no helpers for one-time operations, no defensive validation for
  states internal code cannot reach, Google-style docstrings for large functions and compact
  one-liners for small ones, comments only where logic is not self-evident.
- Core size alarm: review scope around **2,500 physical lines**. HEAD is **2,584 physical /
  2,138 code** across 15 core files. This plan adds roughly **120 physical lines**, so
  Task 1 (the overdue design-boundary review) gates everything else.

---

## Assessment: what Stage 3B item 1 already delivered

I read every core module against the four 3B bullets and the three 3C bullets. Most of
3B item 2 and nearly all of 3B item 3 were already satisfied by item 1's implementation.
The genuinely missing work is much smaller than the checkbox count suggests.

| Spec bullet | Status at HEAD | Evidence | Work left |
|---|---|---|---|
| **3B-2** compact old turns first, then older exchanges in the ongoing turn | **done** | [context.py:191-192](../context.py#L191-L192) clamps the cutoff to `turn_start` while `covered < turn_start` | — |
| **3B-2** never split a call/result pair | **done** | [context.py:169-184](../context.py#L169-L184) builds the `safe` set from whole batches only | — |
| **3B-2** rebuild prompt with summary + retained suffix + fresh observations | **done** | [context.py:141-150](../context.py#L141-L150) | — |
| **3B-2** **reload stable rules** at the boundary | **missing** | `raw[0]` is frozen at [agent.py:75-78](../agent.py#L75-L78); STAGE.md 3A says "compact-boundary reloading waits for Stage 3B" | **Task 5** |
| **3B-2** reload bounded memory | **not applicable yet** | `memory.py` does not exist; Stage 4A owns it | documented seam only (Task 5 step 7) |
| **3B-2** recheck fit before publishing | **done** | [compact.py:113-120](../compact.py#L113-L120) | — |
| **3B-3** run ID / counters / registry intact | **done by construction** | `Compactor` touches only `covered`, `summary`, `attempted_boundary`, `summary_calls`; `run_id`, `tool_attempts`, `failure_count`, `used_ids` are `_run_turn` locals | **regression tests, Task 6** |
| **3B-3** no tool replay, reread summarized file state | **done** | no execution path in `compact.py`; [prompts/compact.md](../prompts/compact.md) instructs rereads | **regression tests, Task 6** |
| **3B-3** truncated output reuses bounded correction feedback, never a partial call | **done** | [llm.py:189-190](../llm.py#L189-L190) raises before any `ToolCall` is built; [agent.py:36-39](../agent.py#L36-L39) supplies the hint | **regression test under pressure, Task 6** |
| **3B-4** four summary calls per run, one attempt per unchanged boundary, charged to `max_iterations` | **done** | [compact.py:57-65](../compact.py#L57-L65) | — |
| **3B-4** **at most two summary calls per attempt** | **missing** | `attempt()` makes exactly one call and returns `FAILED` | **Task 3** |
| **3B-4** **separate summarizer input/output budgets** | **missing** | `_summarize` uses `self.llm.max_tokens`, the actor's reserve | **Task 2** |
| **3B-4** failure preserves raw evidence and stops with `context_limit` | **done** | [agent.py:200-208](../agent.py#L200-L208) | — |
| **3C-1** versioned checkpoint, raw boundary, digest, config | **missing** | `SessionStore` has one record kind | **Task 7** |
| **3C-2** replay summary + uncovered suffix on restart; stale fallback; `/reset` clears it | **missing** | `run_repl` passes raw history only | **Task 8** |
| **3C-3** trace records actual inputs, purpose, boundaries, usage, before/after | **done** | `ModelRequest.input_messages` / `purpose` / `covered_boundary` / `compact_before` / `compact_after` | — |
| **3C-3** **preserve old readers and update eval export** | **missing** | `evals/verify.py` and `evals/foundation.py` never read `purpose`, `budget` or `compact_after`; nothing reads the pre-rename `context` key | **Task 9** |

One documented debt falls inside 3B item 4's scope. CONTEXT_STATE_DESIGN.md says:

> `AgentLimits.max_tool_calls_per_response` is recorded in settings, but the parser enforces
> `config.MAX_TOOL_CALLS_PER_RESPONSE` (8) directly. [...] Reconcile this when wiring Stage 3's
> budgets so recorded configuration describes the limits actually enforced.

Confirmed at [llm.py:202](../llm.py#L202) and [llm.py:243](../llm.py#L243). **Task 4** fixes it.

---

## File Structure

| File | Change | Responsibility after the change |
|---|---|---|
| [config.py](../config.py) | modify | adds `summary_max_tokens` and `max_summary_calls_per_attempt` to `AgentLimits` |
| [llm.py](../llm.py) | modify | `measure_context` / `generate` / `parse_response` accept per-request output and batch limits; the instance default stays the actor's |
| [context.py](../context.py) | modify | `ContextState` gains an optional `instructions` override used by `messages()`; raw is still never edited |
| [compact.py](../compact.py) | modify | `attempt()` becomes a bounded retry loop; `_publish` reloads rules; module exposes `compact_prompt_digest()` |
| [session.py](../session.py) | modify | second record kind (`checkpoint`) with its own validator, `append_checkpoint()` and `load_checkpoint()` |
| [agent.py](../agent.py) | modify | owns the turn's `ContextState`, accepts a checkpoint, saves one after a completed turn, reloads rules for the compactor |
| [trace.py](../trace.py) | modify | adds `ModelRequest.instructions` and the `request_budget()` legacy-key reader |
| [evals/verify.py](../evals/verify.py) | modify | actor/summary split and compaction counters in `metrics()` / `summarize()` |
| [evals/foundation.py](../evals/foundation.py) | modify | same counters in `measure()` |
| [examples/continuation_demo.py](../examples/) | **create** | real-model restart evidence: checkpoint replay versus raw replay versus a stale checkpoint |
| [examples/compact_demo.py](../examples/compact_demo.py) | modify | reports the separate summarizer reserve and per-request error messages |
| [tests/test_compact.py](../tests/test_compact.py) | modify | per-attempt retry, summarizer reserve, rule reload, continuity regressions |
| [tests/test_session.py](../tests/test_session.py) | modify | checkpoint records, digests, staleness, write ordering |
| [tests/test_turn.py](../tests/test_turn.py) | modify | checkpoint replay through `run_turn`, `/reset`, enforced batch limit |
| [tests/test_eval.py](../tests/test_eval.py) | modify | the new export fields and the legacy `context` key |
| docs | modify | STAGE.md checkboxes, CONTEXT_STATE_DESIGN.md, context-memory.md evidence |

---

### Task 1: The overdue design-boundary review

STAGE.md makes this a hard gate: *"The design-boundary review that the audit section
requires is therefore due before 3B grows further."* This plan grows 3B, so it runs first.
No production code changes.

**Files:**
- Modify: `docs/context-memory.md` (new "Design-boundary review" section)

**Interfaces:**
- Consumes: `evals/core_lines.py` (unchanged)
- Produces: a written decision that later tasks cite when they add lines

- [ ] **Step 1: Measure HEAD**

```bash
cd /home/easyvps/GetHandsDirty
python -m agent_from_scratch.evals.core_lines HEAD | tail -5
```

Expected (verified 2026-09-20): `"nonempty_files": 15, "physical": 2584, "code": 2138`.

- [ ] **Step 2: Write the review into `docs/context-memory.md`**

Append this section, then fill the three bracketed judgements yourself — the point of the
gate is that a human states the boundary, not that a number is recorded.

```markdown
## Design-boundary review (2,584 physical lines)

The 2,500-line alarm fired at Stage 3B item 1 and is answered here, before the rest of
3B and 3C add roughly 120 more physical lines.

Distribution at HEAD: tools account for 1,019 of 2,138 code lines (47.7%); the
loop/model/context/session core accounts for the rest. The growth since Stage 2 is in
`context.py`, `compact.py` and `agent.py` — the mechanisms this sprint exists to learn —
not in accidental structure.

Decisions:

- **Keep one `Compactor` class in `compact.py`.** [state why the plan/summarize/publish
  split stays inside one class rather than becoming a module boundary]
- **Keep checkpoints inside `SessionStore`.** [state why a separate `checkpoint.py` would
  duplicate the single-writer, atomic-replacement guarantee `session.py` already owns]
- **Do not compress readable code to return under 2,500.** The audit section forbids it,
  and the measure is a scope alarm, not a quality target. [state the revised alarm, if any]

No module is split or merged as a result of this review.
```

- [ ] **Step 3: Commit**

```bash
git add agent_from_scratch/docs/context-memory.md
git commit -m "docs: answer the 2,500-line design-boundary review before Stage 3B grows

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: A separate summarizer output reserve (3B item 4)

Today `_summarize` measures and generates with `self.llm.max_tokens`, the actor's reserve
(2,048 by default, 512 in the demo). A handoff should be far shorter than a tool-using
actor turn, and reserving the actor's budget for it inflates the summary's own fit check —
which is exactly what rejects an oversized summary input.

**Files:**
- Modify: `agent_from_scratch/config.py:30-41`
- Modify: `agent_from_scratch/llm.py:131-153` (`measure_context`), `llm.py:262-273` (`generate`)
- Modify: `agent_from_scratch/compact.py:67-97` (`_summarize`)
- Test: `agent_from_scratch/tests/test_compact.py`

**Interfaces:**
- Consumes: `LLM.measure_context(messages, tools)`, `LLM.generate(messages, tools)`
- Produces:
  - `AgentLimits.summary_max_tokens: int = 512`
  - `LLM.measure_context(messages, tools, *, max_tokens: int | None = None) -> dict[str, Any]`
  - `LLM.generate(messages, tools, *, max_tokens: int | None = None) -> LLMResponse`
  - both treat `None` as "use the instance default", so every existing call site is unchanged

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compact.py` inside `CompactTests`:

```python
    def test_summary_uses_its_own_output_reserve(self):
        self.limits = replace(AgentLimits(), summary_max_tokens=128)
        self.assertTrue(self.compact())
        summary_reserve = [call.kwargs.get("max_tokens")
                           for call in self.model.generate.call_args_list]
        self.assertEqual(summary_reserve, [128])
        measured = [call.kwargs.get("max_tokens")
                    for call in self.model.measure_context.call_args_list
                    if call.args[0][0]["content"].startswith("Summarize the supplied")]
        self.assertEqual(measured, [128])
```

`CompactTests.measure` is the scripted budget helper; widen its signature to
`def measure(self, messages, schemas, **kwargs)` in the same step so the mock accepts the
new keyword.

- [ ] **Step 2: Run it and watch it fail**

```bash
cd /home/easyvps/GetHandsDirty
python -m unittest agent_from_scratch.tests.test_compact.CompactTests.test_summary_uses_its_own_output_reserve -v
```

Expected: FAIL — `AttributeError` on `summary_max_tokens`, or `[None] != [128]`.

- [ ] **Step 3: Add the limit**

In `config.py`, inside `AgentLimits`, after `max_compact_calls`:

```python
    summary_max_tokens: int = 512  # summarizer output reserve, independent of the actor's
```

- [ ] **Step 4: Accept a per-request reserve in `llm.py`**

Replace the head of `measure_context`:

```python
    def measure_context(
        self, messages: List[ChatCompletionRequestMessage], tools: Dict[str, ChatCompletionTool],
        *, max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Measure the next input without generation or changing the model's KV cache.

        Args:
            messages: the exact model-facing view to be sent.
            tools: the schemas that will accompany it.
            max_tokens: output reserve for this one request; None uses the instance default.

        Returns:
            dict: count method, prompt tokens, window, reserve and remaining room, before
                any safety margin.
        """
        configured = self.max_tokens if max_tokens is None else max_tokens
        reserve = configured if configured is not None and configured > 0 else None
```

and delete the old `reserve = self.max_tokens if ... else None` line. The rest of the
method is unchanged.

In `generate`, replace the signature and the `max_tokens` argument:

```python
    def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        tools: Dict[str, ChatCompletionTool],
        *,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=list(tools.values()),
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            stream=False,
        )
```

- [ ] **Step 5: Spend it in `compact.py`**

In `_summarize`, replace the two calls:

```python
            request.budget = self.llm.measure_context(
                request.input_messages, {}, max_tokens=self.limits.summary_max_tokens)
```

```python
            response = self.llm.generate(deepcopy(request.input_messages), {},
                                         max_tokens=self.limits.summary_max_tokens)
```

- [ ] **Step 6: Run the test and the suite**

```bash
python -m unittest agent_from_scratch.tests.test_compact.CompactTests.test_summary_uses_its_own_output_reserve -v
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: the new test PASSes; **207 tests** pass overall.

- [ ] **Step 7: Commit**

```bash
git add agent_from_scratch/config.py agent_from_scratch/llm.py agent_from_scratch/compact.py agent_from_scratch/tests/test_compact.py
git commit -m "feat: give the summarizer its own output reserve

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Two bounded summary calls per attempt (3B item 4)

The spec allows two calls per attempt. Today one unusable summary — an empty response, a
tool call, or a candidate that does not shrink — ends the attempt, and
`state.attempted_boundary` then forbids ever retrying that same cut. One corrective retry
at the same cut is the cheapest recovery available, and it must stay inside the four-per-run
and `max_iterations` ceilings.

**Files:**
- Modify: `agent_from_scratch/config.py` (one field)
- Modify: `agent_from_scratch/compact.py:35-65` (`attempt`, `_plan`, new `_exhausted`) and `67-100` (`_summarize`)
- Test: `agent_from_scratch/tests/test_compact.py`

**Interfaces:**
- Consumes: `Compactor.attempt(state, schemas, requests, iteration, *, before=None) -> CompactOutcome` (signature unchanged), `AgentLimits.summary_max_tokens` from Task 2
- Produces:
  - `AgentLimits.max_summary_calls_per_attempt: int = 2`
  - `Compactor._exhausted(state: ContextState, requests: list[ModelRequest]) -> bool`
  - `Compactor._summarize(state, boundary, request, retry: str | None = None) -> str | None`
  - one `ModelRequest` per summary call, each `purpose="compact"` with the same
    `covered_boundary`, so `used_model_calls` and `metrics()` charge both

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compact.py`:

```python
    def test_unusable_summary_retries_once_at_the_same_cut_then_gives_up(self):
        self.model.generate.side_effect = [answer(""), answer("Goal: report 7. Next: answer.")]
        self.assertTrue(self.compact())
        self.assertEqual(len(self.requests), 2)
        self.assertEqual({q.covered_boundary for q in self.requests}, {self.state.covered})
        self.assertEqual(self.state.summary_calls, 2)
        retry = self.requests[1].input_messages
        assert retry is not None
        self.assertIn("previous attempt was rejected", retry[0]["content"])

    def test_attempt_never_exceeds_two_calls_or_the_run_budget(self):
        self.model.generate.side_effect = [answer(""), answer(""), answer("Goal: unused.")]
        self.assertFalse(self.compact())
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.model.generate.call_count, 2)
        self.limits = replace(AgentLimits(), max_compact_calls=1)
        self.state = ContextState(self.raw, turn_start=3, last_sent=3)
        self.requests = []
        self.model.generate.side_effect = [answer(""), answer("Goal: unused.")]
        self.assertFalse(self.compact())
        self.assertEqual(len(self.requests), 1)  # the run ceiling stops the retry

    def test_oversized_summary_input_does_not_retry(self):
        self.model.measure_context.side_effect = lambda messages, schemas, **kwargs: {
            "count_method": "exact", "prompt_tokens": 9000, "response_reserve": 512,
            "remaining_tokens": -100, "window_tokens": 8192}
        self.assertFalse(self.compact())
        self.assertEqual(len(self.requests), 1)
        self.model.generate.assert_not_called()
```

Note on the first test: `answer("")` must produce a response whose `content` is empty so
`_summarize` rejects it at the "nonempty text summary" check. Confirm the `answer` helper
in `tests/test_turn.py` builds a `direct` response from raw content before relying on it;
if it strips or defaults, script the mock response inline instead.

- [ ] **Step 2: Run them and watch them fail**

```bash
python -m unittest agent_from_scratch.tests.test_compact -v 2>&1 | grep -E "retries|exceeds|oversized"
```

Expected: FAIL — `len(self.requests)` is 1 where 2 is expected, and `compact()` returns
False for the first test because one empty response ends the attempt.

- [ ] **Step 3: Add the limit**

In `config.py`, after `summary_max_tokens`:

```python
    max_summary_calls_per_attempt: int = 2  # one corrective retry at the same cut
```

- [ ] **Step 4: Restructure `attempt` into a bounded loop**

Replace `compact.py:35-65` (`attempt` and `_plan`) with:

```python
    def attempt(self, state: ContextState, schemas: dict, requests: list[ModelRequest],
                iteration: int, *, before: dict | None = None) -> CompactOutcome:
        """Summarize one legal cut, with at most one corrective retry at that same cut.

        The outcome, not a mutated field, tells the caller what was spent. `before` is the
        caller's own measurement of this same view and these same schemas; supplying it
        avoids tokenizing the turn's largest input a second time.

        Args:
            state: the turn's context state; only `covered`, `summary`, `attempted_boundary`
                and `summary_calls` may change, and only on success or on a spent call.
            schemas: the tool schemas the actor's request will carry.
            requests: the turn's request list; one entry is appended per summary call.
            iteration: the agent loop iteration these calls belong to.
            before: a measurement of `state.messages()` with `schemas`, or None to take one.

        Returns:
            CompactOutcome: APPLIED when a smaller fitting view was published, SKIPPED when
                a gate refused before anything was spent, FAILED when a call reached the
                model and produced no usable view.
        """
        boundary = self._plan(state, requests)
        if boundary is None:
            return CompactOutcome.SKIPPED
        # Record the attempt before spending anything: this exact cut is never retried
        # by a *later* attempt, whatever the retry below does inside this one.
        state.attempted_boundary = boundary
        outcome = CompactOutcome.SKIPPED
        retry: str | None = None
        for _ in range(self.limits.max_summary_calls_per_attempt):
            if outcome is not CompactOutcome.SKIPPED and self._exhausted(state, requests):
                break
            request = ModelRequest(iteration, len(state.raw), purpose="compact",
                                   covered_boundary=boundary, last_sent_boundary=state.last_sent)
            requests.append(request)
            outcome = CompactOutcome.FAILED
            summary = self._summarize(state, boundary, request, retry)
            if summary is not None and self._publish(state, boundary, summary, schemas,
                                                     request, before):
                return CompactOutcome.APPLIED
            if request.status is ModelRequestStatus.BLOCKED:
                break  # The summarizer's own input does not fit; a retry cannot change that.
            retry = request.error_message
        return outcome

    def _plan(self, state: ContextState, requests: list[ModelRequest]) -> int | None:
        """Return the cut worth a model call, or None to refuse without spending one."""
        boundary = state.compact_boundary()
        if (boundary <= state.covered or boundary == state.attempted_boundary
                or self._exhausted(state, requests)):
            return None
        return boundary

    def _exhausted(self, state: ContextState, requests: list[ModelRequest]) -> bool:
        """True when another summary call would exceed its own or the turn's ceiling."""
        return (state.summary_calls >= self.limits.max_compact_calls
                # Leave the actor the last slot: a summary nobody can act on is wasted.
                or used_model_calls(requests) >= self.limits.max_iterations - 1)
```

- [ ] **Step 5: Let `_summarize` correct itself**

Change its signature and prompt assembly in `compact.py`:

```python
    def _summarize(self, state: ContextState, boundary: int, request: ModelRequest,
                   retry: str | None = None) -> str | None:
        """Generate the handoff for raw[covered:boundary]; None when nothing usable came back.

        Args:
            state: the turn's context state, read for the previous summary and raw messages.
            boundary: the exclusive end of the range to summarize.
            request: the record this call writes its input, budget, usage and errors into.
            retry: why the previous call in this attempt was unusable, so the final call can
                correct it instead of repeating it; None on the first call.

        Returns:
            str | None: the handoff text, or None when the call was blocked, failed or
                returned something the actor cannot use.
        """
        generated = False
        try:
            if self._prompt is None:
                self._prompt = (PROMPTS_DIR / "compact.md").read_text(encoding="utf-8")
            instructions = self._prompt
            if retry:
                instructions += (f"\n\nThe previous attempt was rejected: {retry}\n"
                                 "Return only the plain-text handoff, and make it shorter.")
            request.input_messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps({
                    "previous_summary": state.summary,
                    "messages": state.raw[state.covered:boundary],
                    "current_request": state.raw[state.turn_start].get("content"),
                }, ensure_ascii=False)},
            ]
```

The rest of the method body — `request.tools = {}` onward, with Task 2's `max_tokens`
arguments — is unchanged.

- [ ] **Step 6: Run the tests and the suite**

```bash
python -m unittest agent_from_scratch.tests.test_compact -v 2>&1 | tail -5
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **210 tests** pass. Verify the existing
`test_manual_and_automatic_paths_share_summary_and_preserve_raw_session` still asserts
`["compact", "agent"]` — one successful summary must still cost exactly one call.

- [ ] **Step 7: Commit**

```bash
git add agent_from_scratch/config.py agent_from_scratch/compact.py agent_from_scratch/tests/test_compact.py
git commit -m "feat: allow one corrective summary retry inside a compact attempt

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Enforce the batch limit that settings already record (3B item 4)

`TurnResult.settings` records `max_tool_calls_per_response`, but `LLM.parse_response`
enforces the module constant. Overriding the field alone changes nothing, so a frozen
evaluation configuration can describe a limit that never applied. CONTEXT_STATE_DESIGN.md
assigns this reconciliation to Stage 3's budget wiring.

**Files:**
- Modify: `agent_from_scratch/llm.py:62` (message), `176` (`parse_response`), `202`, `243`, `262-284` (`generate`)
- Modify: `agent_from_scratch/agent.py:214` (the actor's `generate` call)
- Test: `agent_from_scratch/tests/test_response.py`

**Interfaces:**
- Consumes: `AgentLimits.max_tool_calls_per_response`
- Produces:
  - `LLM.parse_response(response, max_tool_calls: int = MAX_TOOL_CALLS_PER_RESPONSE) -> LLMResponse`
  - `LLM.generate(messages, tools, *, max_tokens=None, max_tool_calls: int | None = None)`
  - `Compactor` passes neither, so summary calls keep the default (they carry no tools)

- [ ] **Step 1: Write the failing test**

The mocked `LLM` in `tests/test_turn.py` returns already-parsed `LLMResponse` objects and
so never reaches the parser. Test the parser directly, in `tests/test_response.py`:

```python
    def test_parse_response_honours_a_configured_batch_limit(self):
        response = {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "a", "type": "function",
                            "function": {"name": "calculator", "arguments": "{}"}},
                           {"id": "b", "type": "function",
                            "function": {"name": "calculator", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]}
        with self.assertRaises(ResponseError) as caught:
            LLM.parse_response(response, 1)
        self.assertEqual(caught.exception.code, ResponseErrorCode.TOO_MANY_TOOL_CALLS)
        self.assertEqual(len(LLM.parse_response(response, 2).tool_calls), 2)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m unittest agent_from_scratch.tests.test_response -v 2>&1 | grep -i configured_batch
```

Expected: FAIL — `parse_response() takes 1 positional argument but 2 were given`.

- [ ] **Step 3: Thread the limit through the parser**

In `llm.py`, make the default message number-free and carry the actual limit as detail:

```python
    ResponseErrorCode.TOO_MANY_TOOL_CALLS: "Too many tool calls in one response.",
```

Change the signature at line 176 and both checks:

```python
    @staticmethod
    def parse_response(response, max_tool_calls: int = MAX_TOOL_CALLS_PER_RESPONSE) -> LLMResponse:
        """Validate the entire native/Qwen batch before allowing any execution."""
```

```python
            if len(calls) > max_tool_calls:
                raise ResponseError(ResponseErrorCode.TOO_MANY_TOOL_CALLS,
                                    f"At most {max_tool_calls} are supported.")
```

```python
        if len(extracted) > max_tool_calls:
            raise ResponseError(ResponseErrorCode.TOO_MANY_TOOL_CALLS,
                                f"At most {max_tool_calls} are supported.")
```

In `generate`, add the parameter and pass it:

```python
        *,
        max_tokens: int | None = None,
        max_tool_calls: int | None = None,
    ) -> LLMResponse:
```

```python
            parsed = LLM.parse_response(
                response, MAX_TOOL_CALLS_PER_RESPONSE if max_tool_calls is None else max_tool_calls)
```

- [ ] **Step 4: Pass the configured limit from the agent**

In `agent.py:214`:

```python
                response = self.llm.generate(prepared_messages, schemas,
                                             max_tool_calls=self.limits.max_tool_calls_per_response)
```

- [ ] **Step 5: Run the tests and the suite**

```bash
python -m unittest agent_from_scratch.tests.test_response agent_from_scratch.tests.test_turn -q 2>&1 | tail -3
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **211 tests** pass. `tests/test_response.py:176,179` assert only the error code,
so the reworded message does not break them — confirm that in the run, do not assume it.

- [ ] **Step 6: Remove the debt note**

In `docs/CONTEXT_STATE_DESIGN.md`, replace the paragraph beginning *"One existing budget
caveat"* with:

```markdown
`AgentLimits.max_tool_calls_per_response` is now the limit the parser enforces:
`Agent` passes it to `LLM.generate`, which forwards it to `parse_response`. Summary calls
carry no tools and keep the module default. Recorded configuration and enforced limits agree.
```

- [ ] **Step 7: Commit**

```bash
git add agent_from_scratch/llm.py agent_from_scratch/agent.py agent_from_scratch/tests/test_response.py agent_from_scratch/docs/CONTEXT_STATE_DESIGN.md
git commit -m "fix: enforce the per-response tool-call limit that settings record

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Reload stable rules at the compact boundary (3B item 2)

Instructions are loaded once at turn start and pinned as `raw[0]`. A long turn can outlive
an edit to `AGENTS.md` or the configured user defaults, and the compact boundary is the one
point in a turn where republishing the prompt is already happening. Raw evidence must not
change: the override lives on `ContextState`, and `ModelRequest.input_messages` continues
to record what each request actually carried.

**Files:**
- Modify: `agent_from_scratch/context.py:121-150` (`ContextState`, `messages`)
- Modify: `agent_from_scratch/compact.py:23-33` (`__init__`), `102-125` (`_publish`)
- Modify: `agent_from_scratch/agent.py:142-149` (compactor construction), new `_reload_instructions`
- Modify: `agent_from_scratch/trace.py:36-57` (`ModelRequest`)
- Test: `agent_from_scratch/tests/test_compact.py`

**Interfaces:**
- Consumes: `load_instructions(config) -> tuple[str, dict[str, Any]]`, `InstructionLoadError`
- Produces:
  - `ContextState.instructions: ChatCompletionRequestMessage | None = None`
  - `Compactor(llm, limits, *, reload_instructions: Callable[[], tuple[str | None, dict]] | None = None)`
  - `Agent._reload_instructions() -> tuple[str | None, dict[str, Any]]`
  - `ModelRequest.instructions: dict | None = None` (optional and additive; trace stays at `schema_version` 5)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compact.py`:

```python
    def test_compact_publishes_reloaded_rules_without_editing_raw(self):
        with TemporaryDirectory() as directory:
            rules = Path(directory, "AGENTS.md")
            rules.write_text("Original workspace rule.")
            agent = Agent(self.model, None, registry=ToolRegistry([]),
                          instruction_config=InstructionConfig(workspace=Path(directory)))
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            seen_first = False

            def rewrite(messages, schemas, **kwargs):
                nonlocal seen_first
                if not seen_first:
                    seen_first = True
                    rules.write_text("Revised workspace rule.")
                return self.measure(messages, schemas)

            self.pressure = True
            self.model.measure_context.side_effect = rewrite
            result = agent.run_turn("Use 7, not 6.",
                                    [{"role": "user", "content": "old"},
                                     {"role": "assistant", "content": "x" * 400}])
            summary, actor = result.model_requests
            assert actor.input_messages is not None
            self.assertIn("Revised workspace rule.", actor.input_messages[0]["content"])
            self.assertIn("Original workspace rule.", result.messages[0]["content"])
            self.assertEqual(summary.instructions["sources"][-1]["status"], "loaded")

    def test_failed_reload_keeps_the_turn_snapshot_and_records_the_error(self):
        compactor = Compactor(self.model, self.limits,
                              reload_instructions=lambda: (None, {"status": "error",
                                                                  "detail": "unreadable"}))
        self.assertIs(compactor.attempt(self.state, {}, self.requests, 1),
                      CompactOutcome.APPLIED)
        self.assertIsNone(self.state.instructions)
        self.assertEqual(self.requests[0].instructions["status"], "error")
```

- [ ] **Step 2: Run them and watch them fail**

```bash
python -m unittest agent_from_scratch.tests.test_compact -v 2>&1 | grep -i reload
```

Expected: FAIL — `TypeError: Compactor.__init__() got an unexpected keyword argument
'reload_instructions'`.

- [ ] **Step 3: Give `ContextState` an instruction override**

In `context.py`, add the field and extend the class docstring's invariant paragraph:

```python
    attempted_boundary: int = 0
    summary_calls: int = 0
    # Rules republished at a compact boundary. None keeps raw[0], the turn-start snapshot;
    # raw itself is never edited, so trace evidence of earlier requests stays exact.
    instructions: ChatCompletionRequestMessage | None = None
```

and in `messages()`, replace the final line:

```python
        rules = self.raw[:1] if self.instructions is None else [self.instructions]
        return build_messages(instructions=rules, history=history, current_turn=current)
```

- [ ] **Step 4: Reload inside `_publish`**

In `compact.py`, extend `__init__`:

```python
    def __init__(self, llm: LLM, limits: AgentLimits, *,
                 reload_instructions: Callable[[], tuple[str | None, dict]] | None = None):
        self.llm = llm
        self.limits = limits
        self.reload_instructions = reload_instructions
        self._prompt: str | None = None
```

with `from collections.abc import Callable` at the top. Then in `_publish`, insert before
the candidate is built:

```python
        try:
            instructions = state.instructions
            if self.reload_instructions is not None:
                text, request.instructions = self.reload_instructions()
                if text is not None:
                    instructions = {"role": "system", "content": text}
            candidate = ContextState(raw=state.raw, turn_start=state.turn_start,
                                     last_sent=state.last_sent, covered=boundary,
                                     summary=summary, instructions=instructions)
```

and on success also publish the rules:

```python
            state.covered, state.summary = boundary, summary
            state.instructions = instructions
            return True
```

The existing smaller-and-fitting check needs no change: `before` measures the view the
turn keeps if this fails (old rules, raw history) and `after` measures the view it would
adopt (new rules, summary). Comparing those two is exactly the decision being made, so
reloaded rules that grow more than the summary shrinks correctly refuse the swap.

- [ ] **Step 5: Supply the reloader from `Agent`**

In `agent.py`, add the method after `_block_on_budget`:

```python
    def _reload_instructions(self) -> tuple[str | None, dict[str, Any]]:
        """Reread rule sources at a compact boundary; keep the turn snapshot on failure.

        Returns:
            tuple: the reloaded system text, or None to keep the current snapshot, and the
                provenance recorded on the compact request either way.
        """
        try:
            return load_instructions(self.instruction_config)
        except InstructionLoadError as error:
            # Mid-turn this must not end a turn already recovering from context pressure;
            # turn start still refuses to run at all on the same error.
            logger.error("Keeping the turn's instruction snapshot: {}", error)
            return None, {"status": "error", "detail": str(error)}
```

and change the compactor construction at `agent.py:149`:

```python
        compactor = Compactor(self.llm, self.limits,
                              reload_instructions=self._reload_instructions)
```

- [ ] **Step 6: Record it in the trace**

In `trace.py`, add to `ModelRequest` after `compact_after`:

```python
    instructions: dict | None = None  # Rule provenance when a compact boundary reloaded them.
```

Update the `schema_version` comment on `TurnResult` to note the addition without bumping:

```python
    schema_version: int = 5  # 5 renamed ModelRequest.context to budget; later fields are additive.
```

- [ ] **Step 7: Record the Stage 4 seam, without building it**

In `docs/CONTEXT_STATE_DESIGN.md`, add one row to the state-ownership table after the
`ContextState` row:

```markdown
| Reloaded rules | `ContextState.instructions` overrides `raw[0]` from a compact boundary onward; raw is never edited. Stage 4A's bounded memory index will attach at the same point and is not implemented here. |
```

- [ ] **Step 8: Run the tests and the suite**

```bash
python -m unittest agent_from_scratch.tests.test_compact -v 2>&1 | tail -5
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **213 tests** pass.

- [ ] **Step 9: Commit**

```bash
git add agent_from_scratch/context.py agent_from_scratch/compact.py agent_from_scratch/agent.py agent_from_scratch/trace.py agent_from_scratch/tests/test_compact.py agent_from_scratch/docs/CONTEXT_STATE_DESIGN.md
git commit -m "feat: republish reloaded rules at the compact boundary

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Continuity regressions (3B item 3)

This item is a guarantee, not a feature. Reading the code, every clause already holds:
`Compactor` touches four `ContextState` fields and nothing else, and `run_id`,
`tool_attempts`, `failure_count`, `last_failure`, `used_ids` and `self.registry` are
outside its reach. What is missing is the evidence, and the spec's gate names the exact
pressure points: *"Test pressure immediately after a tool result/parser error, a failing
summarizer [...]"*. Expect no production changes; if a test fails, that is the finding.

**Files:**
- Test: `agent_from_scratch/tests/test_compact.py`

**Interfaces:**
- Consumes: `Agent.run_turn`, `Compactor.attempt`, `CompactOutcome`, the `answer` / `call`
  helpers imported from `tests/test_turn.py`, the module-level `exchange(name)` helper,
  `AgentLimits.max_same_failures`
- Produces: no new runtime interface

- [ ] **Step 1: Pressure immediately after a tool result**

Model three generations — a write, then pressure at the next iteration, then the answer —
following the shape of the existing
`test_manual_and_automatic_paths_share_summary_and_preserve_raw_session`. Flip
`self.pressure` inside the measurement mock after the first tool result lands, so
compaction happens with the tool observation already in raw:

```python
    def test_pressure_right_after_a_tool_result_keeps_counters_and_effects(self):
        with TemporaryDirectory() as directory:
            target = Path(directory, "note.txt")
            self.model.generate.side_effect = [
                call(name="write_file", arguments={"path": "note.txt", "content": "once"},
                     call_id="w1").to_message_response(),
                answer("Goal: the write already happened."),
                answer("Wrote note.txt once."),
            ]

            def pressure_after_tool(messages, schemas, **kwargs):
                self.pressure = any(m["role"] == "tool" for m in messages)
                return self.measure(messages, schemas)

            self.model.measure_context.side_effect = pressure_after_tool
            agent = Agent(self.model, None, registry=ToolRegistry([WriteFileTool(directory)]))
            result = agent.run_turn("Write note.txt once.", deepcopy(self.raw[1:3]))
            self.assertEqual(target.read_text(), "once")          # never replayed
            self.assertEqual([q.purpose for q in result.model_requests],
                             ["agent", "compact", "agent"])
            self.assertEqual([q.iteration for q in result.model_requests], [1, 2, 2])
            self.assertTrue(all(q.call_ids[0].startswith(result.run_id)
                                for q in result.model_requests if q.call_ids))
```

`call(...).to_message_response()` stands in for whatever `tests/test_turn.py` already uses
to script a tool-calling `LLMResponse`; read the helper names at the top of that module and
substitute the real one rather than adding a new helper. The three assertions that matter
are the single write, the unchanged run ID prefix on generated call IDs, and the iteration
numbers — a compaction must not restart the loop counter.

- [ ] **Step 2: Parser feedback survives a compaction unseen**

```python
    def test_parser_feedback_survives_compaction_and_stays_unsent(self):
        self.raw.extend(exchange("old") + exchange("recent"))
        self.raw.append({"role": "user", "content": "[Runtime feedback] The output was cut off."})
        self.state.last_sent = len(self.raw) - 1
        self.assertTrue(self.compact())
        published = self.state.messages()
        self.assertIn(self.raw[-1], published)
        self.assertGreaterEqual(self.state.last_sent, self.state.covered)
```

- [ ] **Step 3: A failing summarizer preserves raw evidence and completed effects**

```python
    def test_failing_summarizer_preserves_raw_evidence_and_completed_writes(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            raw_before = deepcopy(self.raw[1:3])
            result = agent.run_turn("Use 7, not 6.", raw_before, session_id="failing")
            self.assertEqual(result.stop_reason, "context_limit")
            self.assertEqual(result.messages[1:3], raw_before)
            self.assertEqual(SessionStore(Path(directory, "sessions")).load_history("failing"), [])
```

- [ ] **Step 4: A truncated response never executes a partial call**

```python
    def test_truncated_response_under_pressure_executes_nothing(self):
        self.pressure = True
        self.model.generate.side_effect = [
            answer("Goal: continue."),
            ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE),
            answer("Done."),
        ]
        result = Agent(self.model, None, registry=ToolRegistry([])).run_turn(
            "Use 7, not 6.", deepcopy(self.raw[1:3]))
        feedback = [m for m in result.messages if "[Runtime feedback]" in str(m.get("content"))]
        self.assertEqual(len(feedback), 1)
        self.assertIn("cut off", feedback[0]["content"])
        self.assertFalse(any(m.get("role") == "tool" for m in result.messages))
```

- [ ] **Step 5: Run them**

```bash
python -m unittest agent_from_scratch.tests.test_compact -v 2>&1 | tail -6
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **217 tests** pass with no production change. If one fails, stop and fix the
runtime — this task's value is that it can fail.

- [ ] **Step 6: Commit**

```bash
git add agent_from_scratch/tests/test_compact.py
git commit -m "test: pin continuity guarantees across compaction boundaries

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Persist a versioned summary checkpoint (3C item 1)

**Files:**
- Modify: `agent_from_scratch/session.py` (imports, `_validate_record`, `load_history`, new `_validate_checkpoint`, `append_checkpoint`, `load_checkpoint`, module-level `checkpoint_digest`)
- Modify: `agent_from_scratch/compact.py` (module-level `compact_prompt_digest`)
- Modify: `agent_from_scratch/agent.py:66-99` (`run_turn`), `129-140` (`_save_session`), `142-148` (`_run_turn`)
- Test: `agent_from_scratch/tests/test_session.py`, `agent_from_scratch/tests/test_compact.py`

**Interfaces:**
- Consumes: `write_jsonl`, `SessionStore.load_records`, `ContextState.covered` / `.summary`
- Produces:
  - `session.checkpoint_digest(messages: list[ChatCompletionRequestMessage]) -> str`
  - `SessionStore.append_checkpoint(session_id, run_id, *, covered: int, summary: str, history: list, config: dict) -> None`
  - `SessionStore.load_checkpoint(session_id: str, history: list) -> dict | None`
  - `compact.compact_prompt_digest() -> str`
  - `Agent._run_turn(result, state, *, compact=False, on_progress=None)` — takes the state instead of `history_length`
  - Checkpoint record shape (`schema_version` 2, **session** schema only):
    ```json
    {"schema_version": 2, "kind": "checkpoint", "run_id": "...", "created_at": "...",
     "covered": 12, "summary": "...", "source_sha256": "...",
     "config": {"compact_prompt_sha256": "...", "summary_max_tokens": 512, "model": {}}}
    ```
    `covered` counts **session** messages, one less than `ContextState.covered`, which
    indexes `raw` with its system message at 0.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_session.py`, importing `checkpoint_digest` alongside `SessionStore`:

```python
    def test_checkpoint_requires_saved_messages_and_binds_to_their_digest(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(directory)
            history: list[ChatCompletionRequestMessage] = [
                {"role": "user", "content": "Remember CEDAR-42"},
                {"role": "assistant", "content": "Noted."},
            ]
            config = {"compact_prompt_sha256": "a" * 64, "summary_max_tokens": 512, "model": {}}
            with self.assertRaises(ValueError):
                store.append_checkpoint("s", "run1", covered=2, summary="Goal: x",
                                        history=history, config=config)
            store.append("s", "run1", history)
            store.append_checkpoint("s", "run1", covered=2, summary="Goal: x",
                                    history=history, config=config)
            record = store.load_checkpoint("s", history)
            assert record is not None
            self.assertEqual(record["covered"], 2)
            self.assertEqual(record["source_sha256"], checkpoint_digest(history))
            self.assertEqual(store.load_history("s"), history)

    def test_stale_or_out_of_range_checkpoints_are_ignored(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(directory)
            history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "a"},
                                                           {"role": "assistant", "content": "b"}]
            store.append("s", "run1", history)
            store.append_checkpoint("s", "run1", covered=2, summary="Goal: x", history=history,
                                    config={"compact_prompt_sha256": "a" * 64,
                                            "summary_max_tokens": 512, "model": {}})
            edited = [{"role": "user", "content": "EDITED"}, history[1]]
            self.assertIsNone(store.load_checkpoint("s", edited))
            self.assertIsNone(store.load_checkpoint("s", history[:1]))
            store.reset("s")
            self.assertIsNone(store.load_checkpoint("s", history))

    def test_invalid_checkpoint_records_are_rejected_with_a_line_number(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            store = SessionStore(directory)
            base = {"schema_version": 2, "kind": "checkpoint", "run_id": "r",
                    "created_at": "2026-09-20T00:00:00+00:00", "covered": 1,
                    "summary": "Goal: x", "source_sha256": "a" * 64, "config": {}}
            for key, value in (("schema_version", 1), ("covered", 0), ("summary", ""),
                               ("config", None), ("source_sha256", 42)):
                with self.subTest(key=key):
                    path.write_text(json.dumps({**base, key: value}) + "\n")
                    with self.assertRaisesRegex(ValueError, "line 1"):
                        store.load_records("broken")
```

- [ ] **Step 2: Run them and watch them fail**

```bash
python -m unittest agent_from_scratch.tests.test_session -v 2>&1 | tail -5
```

Expected: FAIL — `ImportError: cannot import name 'checkpoint_digest'`.

- [ ] **Step 3: Add the digest and the checkpoint validator**

At the top of `session.py`, extend the imports:

```python
from datetime import datetime, timezone
from hashlib import sha256
```

and add after them:

```python
def checkpoint_digest(messages: list[ChatCompletionRequestMessage]) -> str:
    """Bind a checkpoint to the exact raw prefix it claims to summarize."""
    return sha256(json.dumps(messages, sort_keys=True,
                             ensure_ascii=False).encode("utf-8")).hexdigest()
```

Inside `SessionStore`, split the record validator. Replace the first two statements of
`_validate_record` with:

```python
    @staticmethod
    def _validate_record(record: object) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
            raise ValueError("Expected a record with a string run_id.")
        if record.get("kind") == "checkpoint":
            SessionStore._validate_checkpoint(record)
            return
        if not isinstance(record.get("messages"), list):
            raise ValueError("Expected a record with a messages list.")
```

The remainder of the method — the `schema_version` check onward — stays exactly as it is.
Add the new validator directly below it:

```python
    @staticmethod
    def _validate_checkpoint(record: dict) -> None:
        """A checkpoint is only usable if its boundary, digest and configuration are intact."""
        if record.get("schema_version") != 2:
            raise ValueError("Unsupported checkpoint schema_version.")
        if type(record.get("covered")) is not int or record["covered"] < 1:
            raise ValueError("Checkpoint covered must be a positive integer.")
        for name in ("summary", "source_sha256", "created_at"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError(f"Checkpoint {name} must be nonempty text.")
        if not isinstance(record.get("config"), dict):
            raise ValueError("Checkpoint config must be an object.")
```

- [ ] **Step 4: Skip checkpoints when replaying history**

`load_history` currently indexes `record["messages"]`, which a checkpoint does not have:

```python
    def load_history(self, session_id: str) -> list[ChatCompletionRequestMessage]:
        # Legacy records were written only for completed turns. Missing metadata stays absent.
        return [message for record in self.load_records(session_id)
                if record.get("kind") != "checkpoint"
                and ("schema_version" not in record
                     or record["stop_reason"] == RunStopReason.FINAL_RESPONSE)
                for message in record["messages"]]
```

- [ ] **Step 5: Write and read checkpoints**

Add to `SessionStore`, after `append`:

```python
    def append_checkpoint(self, session_id: str, run_id: str, *, covered: int, summary: str,
                          history: list[ChatCompletionRequestMessage], config: dict) -> None:
        """Publish a summary checkpoint over an already-saved prefix of this session.

        Args:
            session_id: the session whose index gains the record.
            run_id: the run whose compaction produced the summary.
            covered: how many session messages the summary represents, counted from the
                start of replayable history.
            summary: the handoff text to replay in place of that prefix.
            history: the full replayable history this checkpoint was computed against.
            config: what produced the summary, for a later reader to judge staleness.

        Raises:
            ValueError: if the boundary is not already backed by saved messages, which is
                what keeps raw evidence on disk before anything references it.
        """
        saved = self.load_history(session_id)
        if covered > len(saved) or history[:covered] != saved[:covered]:
            raise ValueError("Checkpoint boundary is not backed by saved session messages.")
        record = {"schema_version": 2, "kind": "checkpoint", "run_id": run_id,
                  "created_at": datetime.now(timezone.utc).isoformat(), "covered": covered,
                  "summary": summary, "source_sha256": checkpoint_digest(history[:covered]),
                  "config": config}
        self._validate_record(record)
        records = self.load_records(session_id)
        records.append(record)
        write_jsonl(self._path(session_id), records)

    def load_checkpoint(self, session_id: str,
                        history: list[ChatCompletionRequestMessage]) -> dict | None:
        """Return the newest checkpoint still backed by this history, else None.

        A stale or out-of-range boundary is not an error: the caller replays raw history and
        lets the ordinary budget check decide. Only the newest checkpoint is considered, so
        an edited session cannot silently fall back to an older summary of the same prefix.
        """
        for record in reversed(self.load_records(session_id)):
            if record.get("kind") != "checkpoint":
                continue
            if (record["covered"] <= len(history)
                    and checkpoint_digest(history[:record["covered"]]) == record["source_sha256"]):
                return record
            return None
        return None
```

- [ ] **Step 6: Run the session tests**

```bash
python -m unittest agent_from_scratch.tests.test_session -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 7: Hand the agent its own state object**

In `agent.py`, `run_turn` now builds the state so that `_save_session` can read it. Replace
the body between `trace = TraceStore(...)` and `result.elapsed_seconds = ...`:

```python
        turn_start = 1 + len(history or [])
        state = ContextState(raw=result.messages, turn_start=turn_start, last_sent=turn_start)
        try:
            self._run_turn(result, state, compact=compact, on_progress=on_progress)
        except KeyboardInterrupt as error:
            # A completed model request can still be followed by an interrupted tool.
            if result.model_requests and result.model_requests[-1].status == ModelRequestStatus.STARTED:
                result.model_requests[-1].status = ModelRequestStatus.INTERRUPTED
            self._finish_pending_tools(result,
                "Interrupted. Inspect effects before retrying; remaining batch calls were not executed.",
                interrupted=True, output=getattr(error, "output", None))
            result.stop_reason = RunStopReason.INTERRUPTED
        result.elapsed_seconds = round(monotonic() - started, 2)
        trace.save_run(result)
        self._save_session(result, turn_start - 1, state)
        return result
```

and change `_run_turn`'s signature and its first lines:

```python
    def _run_turn(self, result: TurnResult, state: ContextState, *, compact: bool = False,
                  on_progress: Callable[[str], None] | None = None) -> None:
        # Keep the original layout for tracing, session saving and evaluators.
        # Only raw_messages receives new events; prepared views are disposable.
        raw_messages = result.messages
        compactor = Compactor(self.llm, self.limits,
                              reload_instructions=self._reload_instructions)
```

deleting the now-duplicated `turn_start = 1 + history_length` and `state = ContextState(...)`
lines.

- [ ] **Step 8: Save the checkpoint after the raw delta**

Replace `_save_session`:

```python
    def _save_session(self, result: TurnResult, history_length: int, state: ContextState) -> None:
        """Append this turn's raw delta, then a checkpoint that references it.

        The two writes are ordered and separately atomic: a checkpoint never names messages
        that are not already on disk. An incomplete turn contributes neither.
        """
        if self.state_dir is None or result.session_id is None:
            return
        completed = result.stop_reason == RunStopReason.FINAL_RESPONSE
        store = SessionStore(Path(self.state_dir, "sessions"))
        try:
            store.append(result.session_id, result.run_id,
                         result.messages[1 + history_length:] if completed else [],
                         started_at=result.started_at, stop_reason=result.stop_reason)
        except (OSError, ValueError) as error:
            logger.error("Could not save session for run {}; this turn will not be remembered: {}",
                         result.run_id, error)
            return
        if not completed or not state.summary or state.covered <= 1:
            return
        try:
            store.append_checkpoint(
                result.session_id, result.run_id, covered=state.covered - 1,
                summary=state.summary, history=result.messages[1:],
                config={"compact_prompt_sha256": compact_prompt_digest(),
                        "summary_max_tokens": self.limits.summary_max_tokens,
                        "model": self.llm.settings()})
        except (OSError, ValueError) as error:
            logger.error("Could not save the summary checkpoint for run {}; "
                         "the next turn replays raw history: {}", result.run_id, error)
```

`result.messages[1:]` is exactly the session history after this turn's append: the list is
`[system, *history, *turn messages]`, and the delta just written is `[1 + history_length:]`.

Add `compact_prompt_digest` to the `from .compact import ...` line in `agent.py`.

- [ ] **Step 9: Add the prompt digest to `compact.py`**

At module level, after the imports:

```python
def compact_prompt_digest() -> str:
    """Identify the summarizer instructions a checkpoint was produced under."""
    return sha256((PROMPTS_DIR / "compact.md").read_bytes()).hexdigest()
```

with `from hashlib import sha256` at the top.

- [ ] **Step 10: Test the agent-level write**

Add to `tests/test_compact.py`:

```python
    def test_completed_compacted_turn_saves_a_checkpoint_after_its_raw_delta(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            result = agent.run_turn("Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            lines = Path(directory, "sessions", "cp.jsonl").read_text().splitlines()
            turn, checkpoint = (json.loads(line) for line in lines)
            self.assertNotIn("kind", turn)
            self.assertEqual(checkpoint["kind"], "checkpoint")
            self.assertEqual(checkpoint["run_id"], result.run_id)
            self.assertEqual(checkpoint["covered"], 2)
            self.assertEqual(len(checkpoint["config"]["compact_prompt_sha256"]), 64)

    def test_incomplete_turn_saves_no_checkpoint(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
            Agent(self.model, directory, registry=ToolRegistry([])).run_turn(
                "Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            self.assertIsNone(SessionStore(Path(directory, "sessions")).load_checkpoint("cp", []))

    def test_checkpoint_write_failure_leaves_the_saved_turn_intact(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            with patch("agent_from_scratch.session.SessionStore.append_checkpoint",
                       side_effect=OSError("disk full")):
                result = agent.run_turn("Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            self.assertEqual(result.stop_reason, "final_response")
            store = SessionStore(Path(directory, "sessions"))
            self.assertEqual(len(store.load_history("cp")), 2)
            self.assertIsNone(store.load_checkpoint("cp", store.load_history("cp")))
```

The `covered == 2` assertion is the one that catches an off-by-one between the raw index
and the session index: the two-message history is fully covered, and `ContextState.covered`
is 3 at that point.

- [ ] **Step 11: Run everything**

```bash
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **223 tests** pass.

- [ ] **Step 12: Commit**

```bash
git add agent_from_scratch/session.py agent_from_scratch/compact.py agent_from_scratch/agent.py agent_from_scratch/tests/test_session.py agent_from_scratch/tests/test_compact.py
git commit -m "feat: persist a versioned summary checkpoint after its raw delta

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Replay a checkpoint on restart (3C item 2)

**Files:**
- Modify: `agent_from_scratch/agent.py:66-99` (`run_turn` signature and state seeding), `324-336` (REPL history loading)
- Test: `agent_from_scratch/tests/test_compact.py`, `agent_from_scratch/tests/test_turn.py`

**Interfaces:**
- Consumes: `SessionStore.load_checkpoint`, `ContextState.covered` / `.summary`
- Produces: `Agent.run_turn(user_input, history=None, *, session_id=None, compact=False, checkpoint: dict | None = None, on_progress=None) -> TurnResult`
  - `history` stays the **full raw** history, so session slicing and raw evidence are unchanged
  - the checkpoint only seeds `covered` and `summary`, keeping `1 <= covered <= turn_start`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compact.py`:

```python
    def test_restart_replays_summary_and_uncovered_suffix(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            Agent(self.model, directory, registry=ToolRegistry([])).run_turn(
                "Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            # A fresh store and agent stand in for a restarted process.
            store = SessionStore(Path(directory, "sessions"))
            history = store.load_history("cp")
            checkpoint = store.load_checkpoint("cp", history)
            assert checkpoint is not None
            self.pressure = False
            self.model.generate.side_effect = [answer("Still 7.")]
            restarted = Agent(self.model, directory, registry=ToolRegistry([]))
            result = restarted.run_turn("What is the limit?", history, session_id="cp",
                                        checkpoint=checkpoint)
            actor = result.model_requests[0]
            assert actor.input_messages is not None
            self.assertTrue(actor.input_messages[1]["content"].startswith("[Conversation summary:"))
            self.assertEqual(actor.input_messages[-1]["content"], "What is the limit?")
            self.assertEqual(result.messages[1:1 + len(history)], history)
            self.assertEqual(result.settings["checkpoint"]["covered"], checkpoint["covered"])

    def test_stale_checkpoint_falls_back_to_raw_history(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            Agent(self.model, directory, registry=ToolRegistry([])).run_turn(
                "Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            store = SessionStore(Path(directory, "sessions"))
            edited = store.load_history("cp")
            edited[0] = {"role": "user", "content": "EDITED"}
            self.assertIsNone(store.load_checkpoint("cp", edited))
```

Add to `tests/test_turn.py`:

```python
    def test_reset_clears_the_session_checkpoint(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(Path(directory, "sessions"))
            history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "a"},
                                                           {"role": "assistant", "content": "b"}]
            store.append("s", "run1", history)
            store.append_checkpoint("s", "run1", covered=2, summary="Goal: x", history=history,
                                    config={"compact_prompt_sha256": "a" * 64,
                                            "summary_max_tokens": 512, "model": {}})
            agent = Agent(Mock(spec=LLM), directory)
            agent._session_command("/reset", "s", store)
            self.assertEqual(store.load_history("s"), [])
            self.assertIsNone(store.load_checkpoint("s", history))
```

- [ ] **Step 2: Run them and watch them fail**

```bash
python -m unittest agent_from_scratch.tests.test_compact -v 2>&1 | grep -i restart
```

Expected: FAIL — `run_turn() got an unexpected keyword argument 'checkpoint'`.

- [ ] **Step 3: Accept a checkpoint in `run_turn`**

Change the signature and seed the state built in Task 7:

```python
    def run_turn(self, user_input: str, history: list[ChatCompletionRequestMessage] | None = None,
                 *, session_id: str | None = None,
                 compact: bool = False,
                 checkpoint: dict | None = None,
                 on_progress: Callable[[str], None] | None = None) -> TurnResult:
        """Execute supplied history; compact=True requests the same path used under pressure.

        Args:
            user_input: this turn's request.
            history: the full raw replayable history; a checkpoint never replaces it.
            session_id: associates and saves this run; it does not load history.
            compact: summarize before the first generation, as pressure would.
            checkpoint: a validated `SessionStore` checkpoint whose summary replaces the
                covered prefix in the model-facing view only.
            on_progress: receives assistant narration that accompanies a tool batch.

        Returns:
            TurnResult: raw messages, stop reason, and one record per model request.
        """
```

and after the state is constructed:

```python
        if checkpoint is not None:
            state.covered, state.summary = checkpoint["covered"] + 1, checkpoint["summary"]
            result.settings["checkpoint"] = {key: checkpoint[key] for key in
                                             ("run_id", "covered", "source_sha256")}
```

`load_checkpoint` guarantees `covered <= len(history)`, so `state.covered <= turn_start`
and the documented `1 <= covered <= boundary <= last_sent` invariant holds on entry.

- [ ] **Step 4: Load one in the REPL**

Replace the history block at `agent.py:324-328`:

```python
            try:
                history = store.load_history(session_id) if store else []
                checkpoint = store.load_checkpoint(session_id, history) if store else None
            except (OSError, ValueError) as error:
                print(f"Session unavailable: {error}. Use /new, /reset, or /session <id> to recover.")
                continue
```

and pass it:

```python
                result = self.run_turn(user_input, history, session_id=session_id,
                                       compact=compact_next, checkpoint=checkpoint,
                                       on_progress=lambda text: print(f"{agent_label} {text}"))
```

`/reset` already unlinks the session file, and checkpoints live in that same file, so the
checkpoint is cleared with it — Step 1's test pins that rather than adding code.

- [ ] **Step 5: Run everything**

```bash
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **226 tests** pass.

- [ ] **Step 6: Commit**

```bash
git add agent_from_scratch/agent.py agent_from_scratch/tests/test_compact.py agent_from_scratch/tests/test_turn.py
git commit -m "feat: replay a summary checkpoint and its uncovered suffix on restart

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Trace readers and eval export (3C item 3)

The trace already records actual inputs, purpose, boundaries, usage and before/after sizes.
What is missing is that nothing *reads* them: `evals/verify.py` and `evals/foundation.py`
count every request the same way, so a run that compacted twice is indistinguishable from
one that did not, and no reader handles the pre-schema-5 `context` key.

**Files:**
- Modify: `agent_from_scratch/trace.py` (module-level `request_budget`)
- Modify: `agent_from_scratch/evals/verify.py:125-155` (`metrics`), `158-186` (`summarize`)
- Modify: `agent_from_scratch/evals/foundation.py:31-83` (`measure`)
- Test: `agent_from_scratch/tests/test_eval.py`

**Interfaces:**
- Consumes: `ModelRequest.purpose` / `.budget` / `.compact_after` / `.error_message`
- Produces:
  - `trace.request_budget(record: dict) -> dict | None`
  - additive metric keys in both exporters: `actor_requests`, `compact_requests`,
    `compactions_applied`, `max_actor_prompt_tokens`
  - `summarize()` totals for `compact_requests` and `compactions_applied`
  - every existing key keeps its meaning, so the frozen Stage 2B suite still reads

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_eval.py`:

```python
    def test_metrics_separate_actor_and_summary_requests(self):
        result = TurnResult(messages=[], stop_reason=RunStopReason.FINAL_RESPONSE)
        result.model_requests = [
            ModelRequest(1, 4, purpose="compact", status=ModelRequestStatus.COMPLETED,
                         compact_after={"prompt_tokens": 300}),
            ModelRequest(1, 4, purpose="compact", status=ModelRequestStatus.COMPLETED,
                         error_message="Summary did not produce a smaller fitting actor input."),
            ModelRequest(1, 4, purpose="agent", status=ModelRequestStatus.COMPLETED,
                         budget={"prompt_tokens": 900}),
            ModelRequest(2, 6, purpose="agent", status=ModelRequestStatus.BLOCKED,
                         budget={"prompt_tokens": 9000}),
        ]
        report = metrics(result)
        self.assertEqual(report["model_requests"], 3)
        self.assertEqual(report["actor_requests"], 1)
        self.assertEqual(report["compact_requests"], 2)
        self.assertEqual(report["compactions_applied"], 1)
        self.assertEqual(report["max_actor_prompt_tokens"], 900)

    def test_request_budget_reads_both_schema_names(self):
        self.assertEqual(request_budget({"budget": {"prompt_tokens": 5}}), {"prompt_tokens": 5})
        self.assertEqual(request_budget({"context": {"prompt_tokens": 7}}), {"prompt_tokens": 7})
        self.assertIsNone(request_budget({}))
```

- [ ] **Step 2: Run them and watch them fail**

```bash
python -m unittest agent_from_scratch.tests.test_eval -v 2>&1 | tail -5
```

Expected: FAIL — `ImportError` on `request_budget`, then `KeyError: 'actor_requests'`.

- [ ] **Step 3: Add the legacy-key reader**

In `trace.py`, after `used_model_calls`:

```python
def request_budget(record: dict) -> dict | None:
    """Read one request's pre-generation measurement across the schema-5 rename.

    Records at schema_version 5 and later use `budget`; 4 and earlier use `context`.
    """
    budget = record.get("budget")
    return record.get("context") if budget is None else budget
```

- [ ] **Step 4: Split the counts in `evals/verify.py`**

In `metrics`, after `requests = [...]`:

```python
    actors = [q for q in requests if q.purpose == "agent"]
    summaries = [q for q in requests if q.purpose == "compact"]
```

and add to the returned dict, beside `"model_requests"`:

```python
        "actor_requests": len(actors),
        "compact_requests": len(summaries),
        # A published summary records its measured result and no error; a rejected one does not.
        "compactions_applied": sum(q.compact_after is not None and q.error_message is None
                                   for q in summaries),
        "max_actor_prompt_tokens": max(((q.budget or {}).get("prompt_tokens") or 0
                                        for q in actors), default=0),
```

In `summarize`, add beside `"model_requests"`:

```python
        "compact_requests": sum(r["compact_requests"] for r in records),
        "compactions_applied": sum(r["compactions_applied"] for r in records),
```

- [ ] **Step 5: Mirror it in `evals/foundation.py`**

In `measure`, add the same four keys to the returned dict, reusing the identical
expressions so the two exporters cannot drift. Place them after `"model_requests"`.

- [ ] **Step 6: Run the tests and the suite**

```bash
python -m unittest agent_from_scratch.tests.test_eval agent_from_scratch.tests.test_behavior_eval -q 2>&1 | tail -3
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
```

Expected: **228 tests** pass. The Stage 2B behavioral tests must be untouched: every
existing key keeps its value.

- [ ] **Step 7: Commit**

```bash
git add agent_from_scratch/trace.py agent_from_scratch/evals/verify.py agent_from_scratch/evals/foundation.py agent_from_scratch/tests/test_eval.py
git commit -m "feat: report compaction in the eval export and read the pre-rename budget key

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Real-model evidence and stage documentation

Deterministic tests pin policy; they make no claim about what a real summary retains. The
gate asks for a multi-turn conversation and a long single turn that **compact twice and
continue**, plus restart evidence.

**Files:**
- Create: `agent_from_scratch/examples/continuation_demo.py`
- Modify: `agent_from_scratch/examples/compact_demo.py`
- Modify: `agent_from_scratch/docs/context-memory.md`, `docs/CONTEXT_STATE_DESIGN.md`, `docs/STAGE.md`

**Interfaces:**
- Consumes: `Agent.run_turn(..., checkpoint=...)`, `SessionStore.load_checkpoint`, `LLM`
- Produces: `outputs/<run>/report.json` with a per-case record of model calls, actor prompt
  tokens, checkpoint boundary and observed answer

- [ ] **Step 1: Write the continuation demo**

```python
"""Compact a pressured turn, checkpoint it, then continue after a simulated restart.

Run: python -m agent_from_scratch.examples.continuation_demo --output outputs/continuation-demo
One session directory carries three cases: a first turn that compacts and checkpoints, a
restart that replays summary + uncovered suffix, and a restart whose checkpoint no longer
matches an edited history. The fixture is synthetic; no workspace tools are exposed.
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
        history = pressured_history(model, instructions, first)
        state_dir = str(args.output / "state")
        report = {"cases": {}}

        opening = Agent(model, state_dir, registry=ToolRegistry([])).run_turn(
            first, history, session_id="continuation")
        report["cases"]["first_turn"] = asdict(opening)

        # A fresh store and agent stand in for a restarted process.
        store = SessionStore(Path(state_dir, "sessions"))
        saved = store.load_history("continuation")
        checkpoint = store.load_checkpoint("continuation", saved)
        follow_up = "Without rereading anything, state the project code and the limit again."

        replayed = Agent(model, state_dir, registry=ToolRegistry([])).run_turn(
            follow_up, saved, session_id="continuation", checkpoint=checkpoint)
        report["cases"]["restart_with_checkpoint"] = asdict(replayed)

        control = Agent(model, None, registry=ToolRegistry([])).run_turn(follow_up, saved)
        report["cases"]["restart_raw_control"] = asdict(control)

        edited: list[ChatCompletionRequestMessage] = [
            {"role": "user", "content": "EDITED"}, *saved[1:]]
        report["cases"]["stale_checkpoint_ignored"] = {
            "checkpoint_found": store.load_checkpoint("continuation", edited) is not None}

        report["comparison"] = {
            "checkpoint_covered": None if checkpoint is None else checkpoint["covered"],
            "saved_messages": len(saved),
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
```

- [ ] **Step 2: Report the new budgets in the existing demo**

In `examples/compact_demo.py`, include the summarizer reserve and each request's error in
the printed line, so a rejected first call in an attempt is visible:

```python
            print(name, result.stop_reason, repr(result.final_answer),
                  [(q.purpose, q.status, (q.budget or {}).get("prompt_tokens"), q.error_message)
                   for q in result.model_requests], flush=True)
```

and record the limits alongside the budget:

```python
        report = {"initial_budget": budget, "limits": asdict(AgentLimits()), "cases": {}}
```

- [ ] **Step 3: Run both demos against the local model**

```bash
cd /home/easyvps/GetHandsDirty
python -m agent_from_scratch.examples.compact_demo --output outputs/stage3b-rest-20260920
python -m agent_from_scratch.examples.continuation_demo --output outputs/stage3c-20260920
```

Record what actually happens, including failures. The expected shape — not a guarantee —
is that `restart_with_checkpoint_prompt_tokens` is well below
`restart_raw_prompt_tokens`, and that `stale_checkpoint_ignored.checkpoint_found` is
`false`. If the replayed answer loses `CEDAR-42` or `7 kg` where the raw control keeps
them, **that is the result**; write it down as a retained/lost fact, as the September 17
diagnostic did.

- [ ] **Step 4: Write the evidence into `docs/context-memory.md`**

Append a section in the style of the existing ones: what ran, a table of model calls and
actor prompt tokens per case, retained versus lost facts, the output directory, and an
explicit statement that this is one diagnostic and not a benchmark. Add a handoff-log entry
naming what was built, what broke, and the next smallest gap.

- [ ] **Step 5: Update `docs/CONTEXT_STATE_DESIGN.md`**

- Change the "Updated" line and the sentence "Stage 3A and Stage 3B item 1 are implemented;
  checkpoints remain planned."
- In the state table, change the "Versioned summary checkpoint" row from planned to
  implemented, naming `SessionStore.append_checkpoint` / `load_checkpoint`, the session
  `schema_version` 2 record and the `covered = ContextState.covered - 1` mapping.
- Note that trace `schema_version` stays 5 and that `request_budget()` is how a reader
  handles records at 4 and earlier.

- [ ] **Step 6: Update `docs/STAGE.md`**

- Tick 3B items 2, 3 and 4 and all three 3C items, each with the test count, what was
  implemented, and its honest limit — follow the established phrasing, which always names
  what the item does *not* establish.
- Rewrite the Stage 3 preamble: it currently says "summary checkpoints remain planned".
- Replace the "Next coding session" line with Stage 4A.
- Append the Stage 3 completion row to the **Core code size audit** table using the
  committed snapshot, and fill in the source fingerprint:

```bash
cd /home/easyvps/GetHandsDirty
python -m agent_from_scratch.evals.core_lines INDEX
```

Preserve the older rows and state the change from 2,584 physical explicitly. If the count
is still above 2,500, cite the Task 1 review rather than compressing code.

- [ ] **Step 7: Final verification**

```bash
python -m unittest discover -s agent_from_scratch/tests -q 2>&1 | tail -3
pyright agent_from_scratch 2>&1 | tail -3
```

Expected: all tests pass; zero Pyright errors and warnings across the package.

- [ ] **Step 8: Commit**

```bash
git add agent_from_scratch/examples agent_from_scratch/docs
git commit -m "docs: record Stage 3B/3C continuation evidence and close the stage

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Risks and things a reviewer should push back on

- **`state.attempted_boundary` is set once per attempt, before the retry loop.** That is
  deliberate: the retry belongs to this attempt, and a *later* attempt still may not reuse
  the cut. A reviewer could argue the second call should pick a smaller cut instead; the
  spec says "one attempt at an unchanged boundary", which this satisfies either way. I chose
  the same cut because it isolates the summarizer's own failure from the boundary choice.
- **Reloading rules can make the candidate larger.** `_publish` then refuses, and the turn
  keeps the old snapshot and the old view. That is correct but slightly surprising: an
  enlarged `AGENTS.md` can turn a compaction that would have succeeded into a `context_limit`
  stop. Task 5 Step 4 documents it; if the real-model demo hits it, say so in the evidence.
- **Only the newest checkpoint is considered.** An older checkpoint over a still-intact
  shorter prefix is discarded when the newest one is stale. The simpler rule is easier to
  reason about and always falls back to complete raw history, which is never wrong, only
  larger.
- **`covered - 1` is the only place the raw index and the session index meet.** Get it wrong
  and a checkpoint silently drops or duplicates one message. Task 7 Step 10 pins
  `checkpoint["covered"] == 2` against a two-message history for exactly this reason.
- **Test-count expectations are estimates.** Each task states the count it should reach; if
  the actual number differs because a test was parameterized, record the real number and
  move on rather than adding tests to hit a target.
- **Line budget.** These tasks add roughly 120 physical lines to a core already 84 over the
  review alarm. Task 1 is the gate; do not reopen it by compressing code in later tasks.

## Self-review against the spec

- 3B item 2 → Task 5 (rules). Ordering, pair safety and the fit recheck were already
  implemented; the assessment table cites file and line evidence for each.
- 3B item 3 → Task 6, covering all four named pressure points.
- 3B item 4 → Tasks 2, 3 and 4.
- 3C item 1 → Task 7. 3C item 2 → Task 8. 3C item 3 → Task 9.
- Gate ("compact twice and continue", restart, failing summarizer, checkpoint write
  failure) → Task 6 Steps 3-4, Task 7 Step 10, Task 8 Step 1, Task 10 Step 3.
- Gate ("keep prompt-size measurements and real-model retained/lost facts in
  `docs/context-memory.md`") → Task 10 Step 4.
- No task contains "TBD", "add error handling" or "similar to Task N"; every code step shows
  the code, and every interface named in a later task is defined in an earlier one.
