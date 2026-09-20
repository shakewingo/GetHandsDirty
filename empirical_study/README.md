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
| 01 | General-tool development suite and independent verifiers | [Implemented/tested](STAGE-01.md); model panel pending |
| 02 | Opt-in conservative elision before summarization | [Implemented/tested](STAGE-02.md); model panel pending |
| 03 | Opt-in bounded, per-turn persistent plan | [Implemented/tested](STAGE-03.md); model panel pending |
| 04 | Search, repeat reminders, and post-edit diagnostics as separate interventions | [Implemented/tested](STAGE-04.md); model panel pending |
| 05 | Matched real-model comparisons, review, and adoption decision | Planned |

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
