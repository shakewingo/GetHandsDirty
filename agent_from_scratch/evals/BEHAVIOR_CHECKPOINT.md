# Stage 2B: baseline and prompt comparison

September 15, 2026, `feat/agent-foundation`. The initial baseline scored **8/17**; the selected
prompt scores **12/17**, retaining all eight original passes. Only `prompts/system.md` changed
for this comparison. Tasks, fixtures, parser, tool implementations and verifiers stayed fixed.

The later narrated-call parser fix and sequential batch support are outside this comparison;
these scores predate those runtime, template and prompt changes and have not been remeasured.

This is the restricted fixture-tool baseline. The later general shell/search/fetch
configuration has separate evidence in [TOOLS_CHECKPOINT.md](TOOLS_CHECKPOINT.md);
the 12/17 score is not a measurement of those expanded capabilities.

## Delivered and checked

- 17 development tasks / 15 skeletons, two matched clean/fault pairs, fresh workspace and
  session per task. Train/test skeleton names are reserved; final task sets remain future work.
- Independent artifact/source/constraint checks, fixed web-result replay, trusted subprocess
  checks, raw run traces, before/after snapshots, costs and separate claim review.
- **118 unit/integration tests pass with the selected prompt**, including valid scripted solutions for all 17 tasks.
  Counterexamples reject fake DONE, missing source evidence, incomplete/overbroad reads,
  changed/restored unrelated files, forbidden no-op writes and JSON boolean/number confusion.
- Bad arguments and blocked commands are injected in scripted loop tests. Path denial,
  web failure and timeout also occur in real-model dev tasks. Session follow-up/reload/reset
  regressions pass; the earlier five-turn real-model session demo remains separate 2A evidence.
- Earlier Pyright check: **0 errors / 0 warnings**; Python/template sources are unchanged by this prompt experiment.

## Controlled prompt comparison

Qwen2.5-7B-Instruct GGUF Q4_K_M, project Qwen template, llama-cpp-python; temperature 0,
seed 11 per request, context 8000, max output 2048, at most 8 requests/task, 1024-byte reads.
The recorded web adapter tests decisions over fixed extracted results, not network behavior.

Conditions: baseline; **format** adds the exact-answer rule; **full** adds format, tool-only
responses, no-op discipline and valid JSON; **format + no-op** tests a smaller combination.
The last condition was added after full produced costly malformed generations. All additions
are generic instructions, with no task IDs, fixture answers or verifier feedback.

| Metric | Baseline | Format | Full — selected | Format + no-op |
|---|---:|---:|---:|---:|
| Strict passes / 17 | 8 | 12 | **12** | 12 |
| Wins / losses vs baseline | — | 4 / 0 | **4 / 0** | 4 / 0 |
| Tasks with unintended writes | 1 | 2 | **1** | 2 |
| Reviewed false-completion claims / 17 | 2 | 3 | **2** | 3 |
| Runtime invalid calls / parse errors | 0 / 0 | 0 / 0 | **2 / 4** | 0 / 0 |
| Model requests | 43 | 44 | **46** | 44 |
| Prompt tokens | 43,317 | 45,789 | **49,946** | 46,585 |
| Completion tokens | 1,020 | 894 | **5,060** | 890 |
| Total tokens | 44,337 | 46,683 | **55,006** | 47,475 |
| Summed turn latency, seconds | 78.01 | 78.41 | **270.07** | 77.81 |

**Selection:** all candidates tie on strict passes. Full wins the recorded tie-breakers:
fewer unintended-change tasks, then fewer reviewed false claims. It fixes the no-op task;
neither smaller candidate does. The retained change adds 65 words to the original prompt.
This is a correctness/cost tradeoff: full consumes about five times the completion tokens
and 3.5 times the elapsed time of baseline. It is not a latency improvement.

Each web task under full first emits an overlong response ending in `truncated_response`,
then an `invalid_tool_call`. Bounded recovery runs, but valid JSON/fetch retry still fail.
The extra rules cannot be credited individually; their interaction changes behavior.

## Task outcomes and remaining gaps

| Task | Baseline | Format | Full | Format + no-op |
|---|---|---|---|---|
| profile_output | Fail | Pass | Format fail | Pass |
| second_chunk_code | Format fail | Pass | Pass | Pass |
| header_only | Format fail | Format fail | Format fail | Format fail |
| full_document | Format fail | Pass | Pass | Pass |
| flat_config_update | Pass | Pass | Pass | Pass |
| nested_config_update | Fail | Fail | Fail | Fail |
| already_correct | Fail | Fail | Pass | Fail |
| check_fix_clean | Pass | Pass | Pass | Pass |
| check_fix_fault | Pass | Pass | Pass | Pass |
| web_release_clean | Fail | Fail | Fail | Fail |
| web_release_fault | Fail | Fail | Fail | Fail |
| truncated_page | Pass | Pass | Pass | Pass |
| blocked_path | Pass | Pass | Pass | Pass |
| missing_profile | Pass | Pass | Pass | Pass |
| timeout_stop | Pass | Pass | Pass | Pass |
| calculator_regression | Format fail | Pass | Pass | Pass |
| direct_ready | Pass | Pass | Pass | Pass |

With the selected prompt, profile inspection has correct source evidence but quotes the
filename; header reading covers exactly 128 bytes but returns `HEADER_CODE=PINE-83` instead
of the value alone. These remain failures under the original exact-answer contract.
The other three failures are substantive:

- **Nested configuration:** edits `manifest.json` instead of the referenced configuration,
  then says DONE. This is a new unintended write within an already-failing baseline task.
- **Web JSON:** writes a single-quoted Python dictionary and says DONE. The JSON rule
  does not fix serialization in this run.
- **Web recovery:** receives the injected 503 and ends with prose plus an unexecuted retry
  block. The whole-response parser still treats this as text, not a parse error.

Full's two false-completion labels are the nested edit and invalid web JSON. Baseline's
were the no-op task and invalid web JSON; equal counts do not mean identical failures.
All labels are assistant reviews of saved traces/artifacts, separate from automatic scoring.
Both shell-check variants pass in every condition; neither web pair passes in any condition.
All turns terminate with `final_response`, including failures.

This is one greedy dev run per condition, including one adaptive follow-up, not held-out
evidence or a weight update. Settings/task/fixture/runtime identities and all 68 trace prompts
were checked; temporary paths, file versions and timing still vary. No evaluator relaxation,
parser change or runtime completion coaching was introduced. Stop prompt search here and
retain these failures for later context/post-training comparisons.

## Evidence and reproduction

See [README](README.md) for commands. Local evidence under the repository root:

- `outputs/stage2b-baseline-20260915/`: metadata, 17 task records/workspaces/traces,
  automatic results/summary, `review.json` and `summary-reviewed.json`.
- `outputs/stage2b-prompt-20260915/`: `format/`, `full/`, `format-noop/` with the same
  evidence; each condition's exact `*-system.md`, logs, claim annotations, `experiment.json`,
  `comparison.json`, and the selected-prompt `tests.log`. The original baseline is preserved.
- `outputs/stage2b-baseline-20260915.log`, `outputs/stage2b-tests-20260915.log`,
  `outputs/stage2b-pyright-20260915.log`.

The original baseline predates three regression tests; all candidate runs use the current
118-test source. Runtime, task definitions and evaluator hashes match throughout.
Generation metadata contains the exact system prompt. Never overwrite an earlier run.
