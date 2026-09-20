# Stage 03 — bounded persistent plan

Opt-in `planning_enabled` adds a current-turn update_plan tool and concise protocol.
The plan has 1-5 steps (160 characters each) and a 240-character completion condition.
Exactly one step is active unless all steps are complete. Invalid updates preserve
prior state. The injected latest-plan copy is ephemeral; raw update calls remain
ordinary evidence. Existing shared registries are not mutated.

Three focused tests verify validation/atomic update, latest-only injection, isolation
between turns, actual traced schema, and plan preservation in compaction candidate
measurement. Full suite: 221 tests pass. Initial missing-module RED recorded.

Profiles `planning` and `elision_planning` were compared against baseline and
elision at the same frozen revision. Default remains off. Planning status is a model
assertion, not an outcome verifier or permission to end a task.

## Matched model evidence (frozen runtime 42c7228)

| Profile | Strict success | Seconds | Total tokens | Model calls | Successful plan calls |
|---|---:|---:|---:|---:|---:|
| Baseline | 1/8 | 156.77 | 72,441 | 28 | 0 |
| Planning | 1/8 | 186.74 | 87,948 | 29 | 1 |
| Elision + planning | 2/8 | 228.20 | 118,516 | 39 | 0 |

The combined profile's extra pass is json_repair, where neither a plan update nor
elision activated. This is not evidence of component synergy. Random run IDs, file
versions and task execution traces vary despite greedy decoding; repeated and held-out
measurements are necessary. No default change is justified.

This implementation tests a short system protocol, tool and latest-plan injection.
It does not reproduce every paper prompt block (in particular, a separate initial
no-plan reminder), and cannot establish that planning in general is ineffective.
