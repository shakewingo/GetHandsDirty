# Execution ledger — plan: empirical_study/PLAN.md

## Setup

- 2026-09-20: branch `empirical-study/harness-design` created from `d7f3887`.
- Existing untracked `agent_from_scratch/memory.py` contains an unused stub; preserved.
- Baseline: 206 unit/integration tests pass in 4.083 seconds.
- Started unchanged 17-task real-model regression panel; source is frozen during run.
- Ruling: use current checkout/new branch as requested, avoiding another worktree
  and duplicated model-cache configuration.
- Ruling: user explicitly requested phased planning and execution; proceed under
  that authorization, retaining reviewable stage documents and commits.
- Ruling: use `empirical_study/` for tracked review artifacts because the existing
  project documentation paths are ignored by git.

## Interface preflight

- Context/compactor: actor views are disposable; raw offsets cannot move.
- Runner/interventions: configuration must be serializable in trace settings.
- Registry/planning: never mutate the shared default registry.
- Diagnostics/verifier: runtime syntax checks are observations, not outcome scoring.

## Stage 00 complete

- Unchanged real-model regression: 10/17, 44 requests, 93.66 s; see STAGE-00.md.
