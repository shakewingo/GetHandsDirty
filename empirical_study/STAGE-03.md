# Stage 03 — bounded persistent plan

Opt-in `planning_enabled` adds a current-turn update_plan tool and concise protocol.
The plan has 1-5 steps (160 characters each) and a 240-character completion condition.
Exactly one step is active unless all steps are complete. Invalid updates preserve
prior state. The injected latest-plan copy is ephemeral; raw update calls remain
ordinary evidence. Existing shared registries are not mutated.

Three focused tests verify validation/atomic update, latest-only injection, isolation
between turns, actual traced schema, and plan preservation in compaction candidate
measurement. Full suite: 221 tests pass. Initial missing-module RED recorded.

Profiles `planning` and `elision_planning` will be compared against baseline and
elision at the same frozen revision. Default remains off. Planning status is a model
assertion, not an outcome verifier or permission to end a task.
