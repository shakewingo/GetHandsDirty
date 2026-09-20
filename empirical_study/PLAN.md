# Tiny Agent Harness Study Implementation Plan

> **For agentic workers:** Execute task-by-task with TDD and a persistent ledger.

**Goal:** Implement independently switchable harness interventions and record a
reproducible, staged accuracy/efficiency pilot for review.

**Architecture:** Keep the current actor and compactor. Add model-view elision,
separate plan state, bounded action support, and an independent study runner.

**Tech Stack:** Python unittest, existing llama-cpp-python/Qwen2.5-7B, local JSON evidence.

**Spec:** [DESIGN.md](DESIGN.md)

## Global constraints

- Existing defaults remain unchanged; interventions are opt-in.
- No source mutation during model panels; no concurrent model panels.
- Preserve untracked memory.py; no external spending, publishing or merges.
- Treat all study tasks as development data and report actual run limitations.

## Review focus

1. Elision must preserve unseen data, error envelopes and paired call IDs.
2. Plan tool state must not leak between agents or successive user turns.
3. Diagnostics must not turn a completed write into a failed/replayed operation.
4. Search must bound scanned files/bytes/results and reject workspace escapes.
5. Metrics must include failed tasks and never equate normal termination with success.

## Stage 00: baseline

- [x] Create branch from latest `feat/agent-foundation`.
- [x] Run existing suite; expected zero failures.
- [x] Run unchanged 17-task regression panel; record all outcomes, not an expected score.
- [x] Commit protocol and baseline report.

## Stage 01: study runner

Files: `agent_from_scratch/evals/harness_study.py`,
`agent_from_scratch/tests/test_harness_study.py`.

- [x] Write tests for known-good and wrong/incomplete artifact outcomes, no-op
  mutations, unknown usage and cost aggregation. Run to establish missing API failure.
- [x] Implement `make_case(case_id, workspace)`, `score_case(case, result, workspace,
  before)`, `summarize(records)` and a fresh-output-only CLI.
- [x] Freeze task selection, source/fixture/schema/settings hashes; preserve run traces.
- [x] Run focused tests then full suite; save baseline general-tool model panel.
- [x] Commit runner, reference tests, stage report and compact measurements.

## Stage 02: elision

Files: `config.py`, `context.py`, `compact.py`, `agent.py`, `elision.py`,
`tests/test_elision.py` (under `agent_from_scratch`).

- [x] Write failing tests for raw preservation, old bulky body elision, metadata,
  unchanged recent/unsent/error observations, and measured no-op rejection.
- [x] Implement bounded view replacements plus a soft-threshold prepass. Preserve
  replacements when testing summary candidates, but summarize original evidence.
- [x] Check existing summary behavior and all context/batch tests.
- [x] Record a matched model pilot and activation counts; commit report and code.

## Stage 03: planning

Files: `planning.py`, `agent.py`, `config.py`, `tests/test_planning.py`.

- [x] Test plan validation, latest-only injection, per-turn isolation, and traced schema.
- [x] Implement bounded `PlanState`/`UpdatePlanTool`, registry copy, and ephemeral
  state injection. No forced completion or hidden verifier feedback.
- [x] Run full suite and matched planning-only and elision+planning pilots.
- [x] Record overhead, successes, regressions and plan usage; commit.

## Stage 04: action support

Files: `tools/search.py`, `support.py`, `agent.py`, `config.py`, corresponding tests.

- [x] Test bounded literal search with paths/line numbers and escaping symlinks.
- [x] Test repeat reminder once per identical successful streak, reset on changed
  observations, and no effect on existing failure budgets.
- [x] Test JSON/Python diagnostics on writes, no diagnostic-as-task-success, and
  unchanged operation success on a diagnostic error.
- [x] Implement each behind its own flag. Run focused and full tests.
- [x] Run separate pilots (never attribute a bundled result to one component); commit.

## Stage 05: synthesis and review

- [x] Produce task-paired comparison, tokens/time per successful task, stop reasons,
  intervention activation, and explicit caveats.
- [x] Review whole diff; fix material findings with regression tests.
- [x] Verify branch status, untouched user file, tracked reports, and source hashes.
- [x] Record adoption decision and remaining experiments. Keep branch for user review.

Exact test cases and implementation decisions are recorded in each stage report;
the ledger carries refinements as evidence arrives.

## Completion record

All planned implementation and pilot stages are complete. Empirical success was not
an acceptance prerequisite: negative results and inactive mechanisms are reported
in STAGE-05.md, with defaults retained off. Component commits precede a common
frozen model panel; final measured results are committed together after the panel.
No further efficacy, held-out benchmark, or production rollout is implied.
