# Stage 3B item 1: compact without changing raw evidence

Implemented September 17, 2026; reviewed September 18. This completes the first item,
not all of Stage 3B/3C.

`ContextState` owns a summary and three raw offsets: the current-turn start, covered
prefix, and last prefix sent to the actor. Summary generation does **not** advance
the last-sent offset. Parser feedback and tool observations beyond it stay verbatim.
`ContextState.messages()` selects the model-facing view; `build_messages()` deep-copies it.

`Compactor.attempt()` in `compact.py` is the shared entry point, returning `CompactOutcome`
(`APPLIED` / `SKIPPED` / `FAILED`). Automatic attempts start when exact
remaining room falls below margin + headroom (256 + 512 tokens by default). Library
callers can use `agent.run_turn(request, history, compact=True)`, and the REPL reaches the
same path with `/compact`, which marks the next request. It first selects old history,
then older ongoing-turn exchanges on later
attempts. It keeps the current request, instructions, two recent tool batches and
all unseen observations. A batch includes every result, including skipped calls.

The summarizer receives the compact prompt, previous summary, selected raw messages
and current request, with no tools. One call per attempt, at most four per run;
summary calls also consume `max_iterations`, leaving room for an actor call. Both
use the model's configured output reserve. Summary input is measured independently.
Oversized source is rejected, not silently sliced. A candidate replaces the live
summary only when the complete rebuilt actor input is smaller and fits. Failed
attempts stop with `context_limit`; tools already executed remain in effect.

Raw `TurnResult.messages` never shrinks. Session saving and outcome verifiers still
read raw exchanges. Trace schema 4 adds actual input messages/schemas, purpose,
raw boundaries and before/after sizes; summaries count in existing usage metrics.
Legacy trace loading remains unchanged; old records can use raw-prefix inputs.

## Deterministic evidence

**206 tests pass** with the existing `transformer-practice` environment:

```sh
python -m unittest discover -s agent_from_scratch/tests -q
```

The 12 new tests cover manual/automatic paths, two compactions, recent batches,
unseen parser/tool feedback, a failed/skipped batch boundary, malformed/empty/tool
summaries, oversized input and output candidates, shared request limits, repeated
failure counters, a write executed exactly once, raw session replay, and trace/cost
accounting. All 42 Python files pass Pyright with zero errors/warnings after the
TypedDict and fixture-typing review fixes. Scripted token counts test policy;
they do not establish model accuracy.

## Local-model diagnostic and a real failure

Reproduce with a fresh output directory:

```sh
python -m agent_from_scratch.examples.compact_demo --output outputs/compact-demo
```

Local Qwen2.5-7B Q4_K_M, temperature 0, 4,096-token window, 512-token output reserve.
The synthetic history establishes project `CEDAR-42` and a 6 kg limit; the current
request corrects it to 7 kg and asks for the code and limit. Repetitive archive notes
place the full input in the early-compaction band. All measured generated prompt
counts matched backend-reported usage, including summary calls.

The first prompt dropped the code from the summary. The actor then invented
`PROJ-12345`, although full history returned `CEDAR-42` and `7 kg`. Evidence is
preserved in `outputs/stage3b-item1-20260917/report.json` and its run traces.
The revised compact prompt explicitly preserves facts needed by the current request,
and the serialized current request follows the history so it remains salient.

Revised evidence: `outputs/stage3b-item1-revised-20260917/report.json`.

| Mode | Model calls | Actor prompt tokens | Observed answer |
|---|---:|---:|---|
| Full history | 1 | 2,826 | `CEDAR-42`, `7 kg` |
| Automatic | 2 | 393 | Correct code and number; final JSON omitted `kg` |
| Manual | 2 | 393 | Same as automatic |
| Oversized current request | 0 | 6,302, blocked | `context_limit` |

Both revised summaries preserved `CEDAR-42` and `7 kg`; the summary input itself was
2,951 tokens. A smaller actor prompt does not imply lower total cost: compaction
adds a model call. This is one diagnostic, not a benchmark or a guarantee against
loss/hallucination. The remaining unit omission is an observed answer-quality failure.

## Scope review and next gap

Core size is 2,582 physical / 2,139 code lines across 15 core files, excluding tests,
demos and the existing untracked memory stub. This is **82 physical lines above** the
2,500-line review alarm, so the design-boundary review that alarm asks for is now due.
An earlier class refactor was reverted on September 18; the September 20 one landed as
`9802c90`..`ed29c05` with behaviour unchanged. No `LoopState` was added.

Rules still use the turn-start snapshot. Separate summarizer decoding budgets,
rule/memory reload at compact boundaries, broader real-model long-turn evidence,
and durable summary checkpoints/restart remain later items. The 512-token early
headroom cannot guarantee arbitrary tool outputs fit. Recent protected batches or
an oversized unsent observation can still stop continuation. A failed attempt now ends
the turn only when the request it was meant to relieve cannot be sent anyway, so a manual
`/compact` that fails no longer discards a turn the budget can still serve. Raw session
replay remains complete; a restarted run must compact again if necessary.

## Design-boundary review (2,584 physical lines)

The 2,500-line alarm fired at Stage 3B item 1 and is answered here, before the rest of
3B and 3C add roughly 120 more physical lines. STAGE.md requires this review before
Stage 3B grows further.

Distribution at HEAD (`evals/core_lines.py`, 15 core files): **2,584 physical / 2,138 code**.
Tools account for **1,019 of 2,138 code lines (47.7%)**; the loop, model, context, session
and support modules account for the other 1,119. The largest single file is `agent.py`
(362 physical / 318 code), followed by `tools/files.py` (329/294) and `llm.py` (290/248).
Growth since Stage 2 is concentrated in `context.py`, `compact.py` and `agent.py` — the
mechanisms this sprint exists to learn — not in accidental structure.

Decisions:

- **Keep one `Compactor` class in `compact.py` (142 physical lines).** Its three phases
  share one mutable subject, the turn's `ContextState`, and one budget, `self.limits`.
  Neither is meaningful alone: planning cannot decide whether a cut was worth taking
  without knowing whether publishing accepted it, and publishing cannot re-derive the cut.
  A module boundary here would exist only to move lines across a file edge, and would
  force that shared state through a parameter list to do it.
- **Keep checkpoints inside `SessionStore`.** A checkpoint is only valid if the raw
  messages it names are already on disk, so the component that decides whether to publish
  one must be the component that knows what is saved. `SessionStore` already owns the
  single-writer atomic replacement (`write_jsonl`) and the session-ID path resolution that
  rejects escapes and symlinks. A separate `checkpoint.py` would either duplicate both
  guarantees or reach back through `SessionStore` for them.
- **Do not compress readable code to return under 2,500.** The audit section forbids it,
  and the measure is a scope alarm, not a quality target. **Revised alarm: 3,000 physical
  lines**, to be re-reviewed when reached. The rationale is that Stage 4A-4B's `memory.py`
  is the last planned core addition on the required path and is budgeted at roughly 200-250
  physical lines; 3C adds about 120. That puts the projected end of the required stages near
  2,950. Passing 3,000 would mean something unplanned was added, which is exactly when a
  review earns its cost. The original 1,500-2,000 target is recorded as missed, not moved:
  roughly half the core is the general tool surface added in Stage 2, and shrinking it would
  remove capability rather than structure.

No module is split or merged as a result of this review.

## Stage 3B items 2-4 and Stage 3C

Implemented September 20, 2026 as ten reviewed commits, `2615c21`..`6acbbb1` plus the
documentation commit that carries this section. This completes Stage 3.

**3B item 2 — rules at the boundary.** Ordering (old turns first), pair safety and the
pre-publish fit recheck were already in place from item 1. What was added is rule reload:
`ContextState.instructions` overrides `raw[0]` in the model-facing view from a compact
boundary onward, and `Compactor._publish` rereads the sources and measures the candidate
with the rules it would actually carry. Raw evidence never changes, so a trace still shows
the turn-start snapshot at `messages[0]` while `ModelRequest.input_messages` shows what each
request really sent. A failed reload keeps the turn snapshot and records the error on the
compact request instead of ending a turn already under context pressure. Reloading rules
that grow more than the summary shrinks correctly refuses the swap; bounded memory is a
Stage 4A attachment at the same point and is not implemented.

**3B item 3 — continuity.** No production change was required: `Compactor` touches four
`ContextState` fields and nothing else, and `run_id`, `tool_attempts`, `failure_count`,
`used_ids` and the registry are `_run_turn` locals beyond its reach. Four regressions now
make that checkable, at the pressure points the stage gate names.

**3B item 4 — bounded recovery with its own budget.** `AgentLimits.summary_max_tokens`
(512) is the summarizer's output reserve, separate from the actor's, and
`LLM.measure_context` / `LLM.generate` take an optional per-request `max_tokens`.
`Compactor.attempt` is now a bounded loop of at most `max_summary_calls_per_attempt` (2)
calls over one cut; the retry carries why the previous call was rejected. A blocked
summarizer input breaks out at once, because a retry cannot make its own input fit. The
per-run ceiling of four and the `max_iterations` reserve for the actor are rechecked
between calls. Separately, `AgentLimits.max_tool_calls_per_response` is now the limit the
parser enforces rather than a recorded value the parser ignored.

**3C — checkpoints and restart.** `SessionStore` gained a `schema_version` 2 `checkpoint`
record in the same session file: summary, session boundary, source digest and summary
configuration (compact-prompt hash, summarizer reserve, model settings). `covered` counts
session messages, one less than `ContextState.covered`, whose raw index 0 is the system
message. `append_checkpoint` refuses any boundary the session does not already hold, which
is what enforces "persist raw messages before publishing a checkpoint referencing them" and
also makes library-supplied history safe: a checkpoint that could never be replayed is not
written, the raw turn is still saved, and the refusal is logged. `load_checkpoint` considers
only the newest record and returns None on any digest mismatch, so a stale or edited session
falls back to raw history and its ordinary budget check. `/reset` clears checkpoints with
the session file. `run_turn(..., checkpoint=...)` seeds `covered`/`summary` while `history`
stays the full raw list, so session slicing and raw evidence are unchanged.

Trace `schema_version` stays 5: `ModelRequest.instructions` is additive, matching the
precedent set when `budget` was first added at schema 3. `trace.request_budget()` reads the
measurement from records on either side of the schema-5 rename. `metrics()` and `measure()`
now split requests by purpose and report `actor_requests`, `compact_requests`,
`compactions_applied` and `max_actor_prompt_tokens`; every pre-existing key keeps its
meaning, so the frozen Stage 2B suite still reads.

### Deterministic evidence

**228 tests pass** and 44 Python files report **0 Pyright errors and 0 warnings**:

```sh
python -m unittest discover -s agent_from_scratch/tests -q
pyright agent_from_scratch
```

The 22 new tests cover the summarizer's own reserve, the corrective retry and its two
ceilings, a blocked summary input that does not retry, reloaded rules published without
editing raw, a failed reload keeping the snapshot, run ID / iteration / registry continuity
across a compaction, unsent parser feedback surviving a cut, a failing summarizer preserving
raw evidence and writing no session, a truncated response executing nothing, checkpoint
digest binding and staleness, invalid checkpoint records, a checkpoint written after its raw
delta, a checkpoint refused over unsaved history, a checkpoint write failure leaving the
saved turn intact, restart replay of summary + uncovered suffix, `/reset` clearing a
checkpoint, and the new export fields including the legacy `context` key.

Three pre-existing tests changed meaning rather than breaking: one attempt now spends up to
two calls, so a failed compaction charges two requests and two usage records. Scripted model
fixtures across seven test modules gained `**kwargs`; they were positional-only two-argument
functions, so the new per-request keyword reached them as a `TypeError` that the compactor
recorded as a model error. Scripted token counts test policy; they make no tokenizer claim.

### Real-model diagnostic: not run, and why

**No new local-model evidence was produced for Stage 3B items 2-4 or Stage 3C.** The
development host for this session has **3 GB of RAM, about 2 GB available, and no GPU**,
while the Qwen2.5-7B Q4_K_M weights are 4.0 GB plus a 0.69 GB second shard. The first
`run_turn` was OOM-killed after the session seed was written and before any run trace, so
the partial output directory was deleted rather than kept as misleading evidence. The
earlier September 17-18 diagnostics in this document were produced on a machine that can
hold the model; this one cannot.

The demo code is written and syntax-checked, and is the remaining gap for this stage:

```sh
python -m agent_from_scratch.examples.continuation_demo --output outputs/stage3c-<date>
python -m agent_from_scratch.examples.compact_demo --output outputs/stage3b-rest-<date>
```

`continuation_demo.py` seeds a session the way the REPL would, runs a pressured turn that
compacts and checkpoints, then uses a fresh `SessionStore` and `Agent` as a stand-in for a
restarted process: one restart replays summary + uncovered suffix, one replays full raw
history as a control, and one checks that an edited history yields no checkpoint. Its
`report.json` records actor prompt tokens per case and both answers, so the retained and
lost facts can be compared directly. `compact_demo.py` now prints each request's
`error_message`, which is what makes a rejected first call inside an attempt visible.

Until that runs, Stage 3's **gate is met deterministically but not behaviourally**: nothing
here establishes what a real summary retains across a restart, and the 512-token early
headroom still cannot guarantee that arbitrary tool outputs fit.

### Scope after Stage 3

Core size is **2,800 physical / 2,254 code lines** across 15 core files, up **216 physical**
from 2,584 at `d7f3887`. That is nearly double the roughly 120 lines this work was estimated
at; the excess is Google-style docstrings on the new public surface (`append_checkpoint`,
`load_checkpoint`, `attempt`, `_summarize`, `run_turn`, `_save_session`, `_reload_instructions`)
rather than new branching. It remains below the 3,000-line alarm set in the design-boundary
review above. Source fingerprint `ece7ac5b6790923d05f59bb3c553cfe00e3fa99eca695b1a0ef7a312de8e8e4f`.

## September 18 handoff log

- Completed Stage 3B item 1: shared automatic/manual compaction, protected raw history
  and complete tool batches, bounded summary calls, and actual-input traces.
- Reviewed the boundary logic and corrected optional TypedDict access and fixture types.
- Reverted the subsequent class refactor; retained the reviewed first-item implementation.
- Validation: 203 tests; 41 Python files with zero Pyright errors/warnings. The local-model
  diagnostic retained the corrected fact in the summary but still omitted its unit in the answer.

## September 20 handoff log, second session

- Completed Stage 3B items 2-4 and all of Stage 3C in ten reviewed commits: summarizer output
  reserve, one corrective retry per compact attempt, an enforced per-response batch limit,
  rule reload at the compact boundary, continuity regressions, versioned checkpoints, restart
  replay, and compaction reporting in the eval export.
- Stage 3B item 3 needed no production code: it already held by construction. The four new
  regressions are the deliverable, and they passed first time.
- What broke, and what it showed: widening `LLM.generate`/`measure_context` with a keyword
  surfaced eleven scripted test fixtures with positional-only signatures; the resulting
  `TypeError` was swallowed by the compactor's broad `except Exception` and reported as a
  model error, which is how a fixture bug reached a `context_limit` stop. The first checkpoint
  test also failed correctly: it passed history the session had never saved, and
  `append_checkpoint` refused a checkpoint that could never be replayed. Both were fixed in
  the tests, not by weakening the runtime.
- Validation: 228 tests; 44 Python files with 0 Pyright errors and 0 warnings.
- **Next smallest gap:** no real-model evidence. This host has 3 GB RAM and no GPU, so the
  7B weights were OOM-killed mid-turn. `examples/continuation_demo.py` is written and waiting.

## September 20 handoff log

- Restructured compaction into `compact.py`: `Compactor` with `_plan` / `_summarize` /
  `_publish`, and `CompactOutcome` replacing the bool-plus-mutated-field signalling the
  agent loop previously had to decode. Thirteen single-concern commits, behaviour preserved;
  `8672ec4` is a verbatim move readable with `--color-moved`.
- One definition each for the context-fit rule (`context_blocker`) and for model calls
  charged to the turn (`used_model_calls`); the actor request is built after compaction,
  retiring a `pop()`/`append()` pair that also, undocumented, kept that request out of the
  compaction budget.
- Behaviour changes, both with regression tests: a failed attempt no longer ends a turn
  the budget can still serve, and an unavailable measurement keeps `context_unavailable`
  instead of being relabelled `context_limit`.
- Naming: the turn's `ContextState` is `state`; `ModelRequest.context` is `budget` at
  `schema_version` 5, with records at 4 and earlier carrying the old key.
- The REPL gained `/compact`.
- Validation: 206 tests; 42 Python files with zero Pyright errors/warnings. No new
  real-model diagnostic was run for this structural change.
