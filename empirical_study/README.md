# Tiny agent harness empirical study

Branch: `empirical-study/harness-design`  
Starting point: `feat/agent-foundation` at `d7f3887` (2026-09-20)

## Review map

Each stage has its own commit, a report here, and immutable raw evidence under
`outputs/empirical-study/`. Reports and compact machine-readable measurements are
tracked; raw model transcripts remain local because they can be large.

| Stage | Deliverable | State |
|---|---|---|
| 00 | Freeze current baseline and study protocol | [Complete](STAGE-00.md) |
| 01 | General-tool development suite and independent verifiers | [Complete](STAGE-01.md); [model evidence](STAGE-05.md) |
| 02 | Opt-in conservative elision before summarization | [Complete](STAGE-02.md); [model evidence](STAGE-05.md) |
| 03 | Opt-in bounded, per-turn persistent plan | [Complete](STAGE-03.md); [model evidence](STAGE-05.md) |
| 04 | Search, repeat reminders, and post-edit diagnostics as separate interventions | [Complete](STAGE-04.md); [model evidence](STAGE-05.md) |
| 05 | Matched real-model comparisons, review, and adoption decision | [Complete](STAGE-05.md) |
| 06 | Merge remote compact enhancements; revalidate harness interactions | [Complete](STAGE-06.md) |

Start with the [latest integration findings](STAGE-06.md), [review guide](REVIEW.md),
and [original experiment findings](STAGE-05.md).

Stage 05 used runtime `42c7228`. Stage 06 integrates remote foundation `54e863d`
into runtime `90e1f79`; historical scores are not scores for this new runtime.

Read [design](DESIGN.md), [execution plan](PLAN.md), and [ledger](LEDGER.md).
The study does not change production defaults based on a small development sample.

## Evidence rules

- Task success comes from independent artifact/evidence checks, never `final_response`.
- Failed runs contribute to total cost and elapsed time.
- Compare matched task IDs, fixtures, model settings, seeds, and intervention flags.
- Report wins/losses, success, tokens, elapsed time, compaction and tool failures.
- Scripted tests establish implementation correctness, not model capability.
- A greedy sample is a pilot, not an independent repeated-trial estimate.
- The old restricted 17-task benchmark is a regression panel, not a general-tool score.
- No benchmark-answer feedback enters the runtime. Legitimate syntax checks may.

Source: [An Empirical Study of Harness Design for Coding Agents](https://arxiv.org/pdf/2609.20804).
This is a local transfer study, not a reproduction of its benchmark scores.
