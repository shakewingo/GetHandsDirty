# Stage 8 benchmark: real-model evidence

Design: [STAGE8_DESIGN.md](STAGE8_DESIGN.md). Plan: [STAGE8_PLAN.md](STAGE8_PLAN.md). This file
records the dev pilot (Task 24) and, once frozen, the test-split pipeline check (Task 25).

## Dev pilot (15 tasks)

```sh
python -m agent_from_scratch.evals bench --output outputs/bench-dev-pilot --split dev
```

Run September 22, 2026, one greedy pass (temperature 0), Qwen2.5-7B-Instruct Q4_K_M, `N_CTX =
32768`. Two runs were made: the first surfaced an evaluator bug (below); the second, after the
fix, is the one reported here.

### Evaluator fix made during the pilot

The first run scored 7/15. `single_field_edit-0` did the file edit correctly — `files`,
`unchanged`, `evidence` and `process` all passed, and the reply contained `DONE` as a whole
token — but still failed, because the reply was wrapped in a sentence
("The file `config.json` has been successfully updated...") and the skeleton declared
`Answer(value="DONE", format="required")`, which demands an exact match.

Checked against the approved design (`STAGE8_DESIGN.md`'s scoring rules and its per-skeleton
table): only tasks whose deliverable **is** the reply token (`UNCHANGED`, `NEED_INPUT`,
`MISSING`, `CHECKED`) are `format="required"`; "value-extraction and edit tasks are advisory."
Five skeletons — `single_field_edit`, `pointer_nested_edit`, `rename_key_all_files`,
`append_list_item`, `flaky_write_retry` — had all been implemented with `format="required"` on
their `DONE` claim token, a transcription slip present since Tasks 6/12/13/14/16. This is an
evaluator bug, not a model failure: it would have scored a correct edit as a failure solely for
prose around the completion word.

Fixed in commit `b509e6b`: all five now use the default `advisory` format. A gate-test
regression, `test_advisory_answers_tolerate_a_reply_wrapped_in_prose`, was added — it scripts a
prose-wrapped reply for every non-required skeleton and asserts the run still passes. The
existing gate tests could not have caught this: their scripted solutions always reply with the
bare exact token, so `format_exact` was always trivially `True` there regardless of whether the
field was actually load-bearing.

### Result, second run

| Skeleton | Family | Passed | `format_exact` | Notes |
|---|---|---:|---:|---|
| `pointer_lookup` | inspection | 3/3 | 0/3 | Always advisory; the model never replies with the bare filename alone, and doesn't need to |
| `single_field_edit` | updates | 2/3 | 2/3 | One seed's write was garbled (below) |
| `check_fix_recheck` | recovery | 4/6 | 4/6 | 3 clean pass; 1/3 fault pairs recovers |
| `no_op_correct_config` | stopping | 0/3 | 3/3 | Exact reply every time, but an unwanted extra write sinks all three (below) |
| **Total** | | **9/15** | 9/15 | |

`false_completion`: 1/9 claims reviewed (a task claimed `DONE` while its state checks did not
all pass). `invalid_calls`: 2. `model_requests`: 66. `elapsed_seconds`: 348. `total_tokens`:
122,292. No run reached `context_limit`; the two non-`final_response` stops were `max_iterations`
(one fault-recovery run) — the dev tasks stay far below the window's pressure thresholds, as
already established in the pressure suite (`EMPIRICAL_STUDY_PLAN.md`).

### Genuine model failures (left as-is; not evaluator bugs)

- **`single_field_edit`, the failing seed.** The model wrote the *line-numbered* text it had
  just read back verbatim (`"1| {\n2|   \"output\": ...`) instead of stripping the `N| ` prefix
  `read_file` adds for display, producing invalid JSON. `write_file`'s own post-write diagnostic
  caught this (`"JSONDecodeError: Extra data..."`) in the tool result, and the model proceeded
  to claim `DONE` anyway without reading the diagnostic. Another seed (in the first run)
  hallucinated a literal `{json_content}` placeholder into the write. Both are real 7B failures
  at a task the harness gives it every tool needed to do correctly.
- **`no_op_correct_config`, all three seeds.** The prompt is explicit: "If both are already
  correct, reply only UNCHANGED and do not call write_file." The model reads the file, confirms
  it is correct, and **writes it back anyway** (again via the same line-numbered-content bug
  above) before replying `UNCHANGED`. `no_write_attempts` correctly fails all three. This is the
  task the design table calls "appropriate stopping," and the model does not stop.
- **`check_fix_recheck` fault condition.** Two of three fault pairs failed to recover: one hit
  `max_iterations` before finishing; the other finished but reported a mismatched value. The
  third recovered correctly (ran the check, saw the failure, fixed the field, reran the check,
  replied `CHECKED`). This is exactly the mixed outcome the recovery-family skeletons are
  designed to surface, not evidence of a broken skeleton — the clean condition passed all three
  times, isolating the difficulty to the recovery step itself.

### Reading these numbers

This is a **pipeline check on one greedy sample**, not a capability claim: Metal decoding is not
bit-reproducible (documented already in `EMPIRICAL_STUDY_PLAN.md`), and a second run of the
unfixed evaluator produced a different specific failing seed for `single_field_edit`/
`check_fix_recheck` than the first. The value of this run is that every failure is
**interpretable** (a real write-content bug, a real appropriate-stopping miss, a real recovery
gap) rather than a scoring artifact — which was the evaluator-bug check this task exists to run.
The 7B model is not the training control; see `docs/STAGE8_DESIGN.md`'s real-model protocol.

## Test-split pipeline check (60 tasks)

```sh
python -m agent_from_scratch.evals freeze
python -m agent_from_scratch.evals bench --output outputs/bench-test-run --split test --final
```

Run September 22, 2026, once, after freezing `evals/bench/manifest.json` (git revision `7a01104`
onward). Same settings as the dev pilot: greedy, Qwen2.5-7B-Instruct Q4_K_M, `N_CTX = 32768`.
`metadata.json`'s `manifest_drift` field is `{}` — the run matched its own frozen manifest.

### A second evaluator bug, found freezing for the first time

`--final` refused immediately with `Manifest drift in ['git_revision']`. Committing
`evals/bench/manifest.json` always changes `HEAD`, and the manifest recorded that revision as a
field `manifest_drift()` compared — so every freeze would self-drift against its own commit the
instant it landed, permanently. Fixed in `3da39b9`: `git_revision` stays in the manifest for
provenance but is excluded from the drift comparison, with a regression test
(`test_manifest_drift_never_gates_on_git_revision`). This is a design gap the dev pilot could
not have surfaced, since Task 20/21's tests build and check a manifest in memory without ever
freezing and immediately re-checking against the committed file in one real sequence.

### Result

| Skeleton | Family | Passed | `format_exact` |
|---|---|---:|---:|
| `deep_chain_lookup` | inspection | 4/6 | 0/6 |
| `grep_locate` | inspection | 6/6 | 6/6 |
| `sum_across_files` | inspection | 4/6 | 0/6 |
| `pointer_nested_edit` | updates | 0/6 | 1/6 |
| `rename_key_all_files` | updates | 0/6 | 0/6 |
| `append_list_item` | updates | 0/6 | 0/6 |
| `transient_read_failure` | recovery | 6/6 | 0/6 |
| `flaky_write_retry` | recovery | 0/6 | 0/6 |
| `ambiguous_choice_stop` | stopping | 1/6 | 2/6 |
| `missing_file_report` | stopping | 0/6 | 3/6 |
| **Total** | | **21/60** | 12/60 |

`false_completion`: 9/24 claims reviewed. `invalid_calls`: 20. `model_requests`: 289.
`elapsed_seconds`: 2,769 (~46 min). `total_tokens`: 643,164. No `context_limit` stops.

### Reading the family split

- **`inspection` (14/18) and the `transient_read_failure` recovery pair (6/6) are the model's
  strength here**: multi-file reads with a clear single value to report. `grep_locate` is a
  clean 6/6 — a `grep_text` query plus one confirming read is well within reach.
- **`updates` is 0/18**, even with the `format="required"` bug already fixed (confirmed: several
  of these replies now pass `answer_content` and are correctly graded `advisory`, but still fail
  on `files` or `normal_finish`). Sampled representative failures directly rather than inferring
  from the aggregate:
  - `pointer_nested_edit-0`: `max_iterations` reached with no answer — manifest read, settings
    read, and a correct write did not fit the skeleton's 6-request budget for this seed.
  - `rename_key_all_files-0`, `append_list_item-0`: finished normally, claimed `DONE` truthfully
    in tone, but the written JSON was wrong — the same failure mode the dev pilot already
    documented (the model sometimes writes back the `N| `-prefixed text `read_file` displays,
    producing invalid or mismatched content).
  - `flaky_write_retry-0-clean`: `no_progress` after 3 consecutive invalid tool calls, offered
    only `write_file` with no `read_file` to fall back on.
  
  These are the same content-writing failure already named in the dev pilot section, now seen at
  a larger sample. No new evaluator bug found in this family; per this task's own instruction,
  its difficulty is not retuned from this result.
- **`stopping` (1/12) and `flaky_write_retry`'s clean condition (0/3)** show the same
  over-eager-write and format patterns as `no_op_correct_config` in the dev pilot.
- **`false_completion` at 9/24** (37.5% of claims) is the sharpest single number here: over a
  third of the replies that claimed done, checked, or a value were wrong in a way the harness
  can detect automatically, without needing a human review pass.

### Reading these numbers

Also a **pipeline check on one greedy sample**, run once as instructed — not a capability score
and not compared against the dev pilot's rate (different families, different task counts). Its
purpose was to confirm the frozen benchmark runs end to end at this scale with interpretable
failures, which it did, and to catch pipeline-level bugs before Stage 9, which it also did (the
`git_revision` drift gate above). The 7B model is not the training control; the real base
control is the Stage 9 checkpoint on vLLM, per `docs/STAGE8_DESIGN.md`'s real-model protocol.
