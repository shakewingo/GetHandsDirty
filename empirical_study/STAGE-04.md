# Stage 04 — separately controlled action support

Three independent interventions, no change to default tools or behavior:

- `search` profile registers SearchFilesTool: literal queries, paths and line numbers;
  200 files / 2,000 directory entries / 1 MiB total / 64 KiB per file / 40 matches.
  Skips symlinks and ignored build/cache directories; incomplete scans are flagged.
- `repeat_reminder_enabled`: one reminder at the third identical successful read,
  list or search observation. Changed arguments/evidence reset it. No new forced stop.
  Reminder appears after the whole tool batch, preserving call/result adjacency.
- `diagnostics_enabled`: syntax-only Python/JSON observations after completed file
  edits/writes, capped at 128 KiB. Errors do not change write success or replay it.

Six support tests cover search boundaries/symlinks, repeated observations,
non-replay diagnostics and actual runtime placement. Existing batches also pass.
Full suite: 229 tests pass. Initial missing-module RED recorded.

The search tool is available for explicit registry construction, and the pilot adds
it only in its search profile. No automatic arbitrary workspace selection is added.
Parallel tool execution remains deferred: existing sequential batching is preserved.
Stage 05 reports each intervention separately, including whether it activated.

Independent review before freeze found and fixed: prior-turn-only compaction boundary
blocking current-turn elision, absolute-path recovery scoring, and whitespace bypass
of plan/search bounds. Four regression failures were observed before their fixes.
Full log: `outputs/empirical-study/stage-04-final-tests.log`.

## Matched model evidence

Search: 1/4 vs matched baseline 1/4; local search task fell from 53.54 to 9.50 s,
but nested task exhausted its budget and total subset cost increased.
Repeat: 0/2 vs 0/2, with zero reminders activated; efficacy remains unmeasured.
Diagnostics: 1/3 vs 0/3, one valid syntax result after an already-correct write;
this does not demonstrate error-driven repair. The nested task exhausted its budget.
See [Stage 05](STAGE-05.md) for paired costs, activation and causal limitations.
