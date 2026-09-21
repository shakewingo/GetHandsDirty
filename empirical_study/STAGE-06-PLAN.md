# Foundation compact integration plan

> Execute inline; maintain this checklist and Stage 06 evidence for review.

**Goal:** Merge remote foundation enhancements into the current harness branch and
validate their interaction with experimental context management and planning.

**Architecture:** Keep upstream independent summary budget, bounded retry, rule
reload and session checkpoint behavior. Preserve opt-in elision/planning/support;
fix integration defects, without treating historical scores as current results.

**Spec:** User request (2026-09-21) and DESIGN.md; local merge explicitly authorized.
**Base:** harness f12bec2; remote foundation 54e863d; common ancestor d7f3887.

## Constraints and review focus

- Preserve user memory.py and all historical experiments; no push or default flag change.
- Merge into the current checkout/branch as requested; do not switch working branch.
- Summary candidate measurement must retain plan, elision and reloaded rules together.
- Compact-boundary rule reload must retain enabled harness protocol instructions.
- Checkpoint replay must retain raw evidence and uncovered suffix; plan stays per-turn.
- New summary retry/reserve must count every call and leave an actor slot.
- Freeze code before any new model run; use fresh output paths and separate evidence.

## Tasks

- [x] Fetch origin; inspect all 11 new commits and compact-related interfaces.
- [x] Merge origin/feat/agent-foundation with --no-commit; resolve overlapping fields
      by preserving both implementations, then run the complete unittest suite.
- [x] Add regression tests for rule reload with planning and combined candidate
      state; demonstrate failure before fixing material integration issues.
- [x] Run full suite; independent review of merge and interaction fixes.
- [ ] Commit merged runtime; run focused real-model history_retention/history_edit
      probes with baseline, elision, planning and elision_planning at the same merged
      revision (the planning profiles also exercise the repaired reload path). Preserve original
      suite/prompt/scoring and compare only task-matched conditions. This tests whether
      the new 512-token summary reserve admits histories blocked in Stage 05.
- [ ] Record upstream changes, applicability/optimization decisions, test and model
      outcomes in STAGE-06.md; link README/LEDGER and commit evidence.

The user's request already authorizes merge, investigation and compatibility fixes;
no separate plan approval is needed. Further behavioral redesign (new recall, new
elision format, new cycle detection) requires a separate controlled experiment and
is recorded as follow-up rather than bundled into this integration.

## Execution evidence

- Remote fetched successfully; exactly 11 new foundation commits.
- Five conflicted files resolved by composing both branches; initial merged suite
  251 tests passed.
- New actor-input regression failed with planning=True (protocol count 0 instead
  of 1); shared instruction assembly fixes the root cause. Combined candidate test
  also covers atomic rejection, plan cost, retained elision and reloaded rules.
- Full suite after fix: 253 tests pass in 3.958 seconds. Independent review completed.

- Reviewer found an upstream legacy-results regression: new compact fields crashed
  old --review exports. Two regression tests reproduced failures before the fix;
  missing/mixed counts now remain unknown, modern counts still sum correctly.
- Final merged suite: 255 tests pass in 3.979 s. Independent reviewer ran 70
  targeted tests; no other material integration findings. Original memory.py hash
  unchanged. Runtime is ready to freeze for the 8-case focused panel.
