# Stage 3B item 1: compact without changing raw evidence

Implemented September 17, 2026; reviewed September 18. This completes the first item,
not all of Stage 3B/3C.

`ContextState` owns a summary and three raw offsets: the current-turn start, covered
prefix, and last prefix sent to the actor. Summary generation does **not** advance
the last-sent offset. Parser feedback and tool observations beyond it stay verbatim.
`ContextState.messages()` selects the model-facing view; `ContextBuilder` deep-copies it.

`compact_context()` in `context.py` is the shared function. Automatic attempts start when exact
remaining room falls below margin + headroom (256 + 512 tokens by default). Library
callers can use `agent.run_turn(request, history, compact=True)`; no CLI command was
added. It first selects old history, then older ongoing-turn exchanges on later
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

**203 tests pass** with the existing `transformer-practice` environment:

```sh
python -m unittest discover -s agent_from_scratch/tests -q
```

The 12 new tests cover manual/automatic paths, two compactions, recent batches,
unseen parser/tool feedback, a failed/skipped batch boundary, malformed/empty/tool
summaries, oversized input and output candidates, shared request limits, repeated
failure counters, a write executed exactly once, raw session replay, and trace/cost
accounting. All 41 Python files pass Pyright with zero errors/warnings after the
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

Core size is 2,473 physical / 2,094 code lines, excluding tests, demos and the
existing untracked memory stub. This remains 27 physical lines below the 2,500-line
review alarm. The September 18 class refactor was reverted; the first-item
implementation and earlier type fixes are retained. No `LoopState` was added.

Rules still use the turn-start snapshot. Separate summarizer decoding budgets,
rule/memory reload at compact boundaries, broader real-model long-turn evidence,
and durable summary checkpoints/restart remain later items. The 512-token early
headroom cannot guarantee arbitrary tool outputs fit. Recent protected batches or
an oversized unsent observation can still stop continuation. Raw session replay
remains complete; a restarted run must compact again if necessary.

## September 18 handoff log

- Completed Stage 3B item 1: shared automatic/manual compaction, protected raw history
  and complete tool batches, bounded summary calls, and actual-input traces.
- Reviewed the boundary logic and corrected optional TypedDict access and fixture types.
- Reverted the subsequent class refactor; retained the reviewed first-item implementation.
- Validation: 203 tests; 41 Python files with zero Pyright errors/warnings. The local-model
  diagnostic retained the corrected fact in the summary but still omitted its unit in the answer.
