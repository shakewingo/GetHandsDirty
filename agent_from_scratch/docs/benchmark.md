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

**Superseded by the final-review fix wave (commits `be4d8d3`, `0e3a937`, revision `9e8538f`
onward).** The results below are pinned to the manifest at `7a01104`/`703a2f4`, not the current
frozen benchmark. Two skeletons' content changed after this run: `flaky_write_retry` gained a
`read_file` tool it did not have here (it was unsolvable as specified — see that row's note
below — and is not a real 0/6 capability result), and 2 of `append_list_item`'s 6 seeds got new
prompt/expect content (a registry-count generation bug fix). The other eight skeletons and the
remaining four `append_list_item` seeds are unchanged and still describe the current benchmark.
Not re-run: Task 25 runs the test split once as a pipeline check, not to be repeated on every
manifest edit; the next real test-split evidence belongs to Stage 9's baseline.

**Re-frozen after retiring the Stage 2B `dev` suite** (source hashes only). Removing
`evals/legacy_files.py`, moving `evals/check.py` to `evals/bench/check.py` (byte-identical) and
trimming the shared `run.py`/`verify.py` changed `source_sha256`. All 75 task entries (prompt,
expect, fault, tools, workspace digests) and `verifier_sha256` are unchanged, so the benchmark
content is the same as at `9e8538f`.

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
| `append_list_item`¹ | updates | 0/6 | 0/6 |
| `transient_read_failure` | recovery | 6/6 | 0/6 |
| `flaky_write_retry`² | recovery | 0/6 | 0/6 |
| `ambiguous_choice_stop` | stopping | 1/6 | 2/6 |
| `missing_file_report` | stopping | 0/6 | 3/6 |
| **Total** | | **21/60** | 12/60 |

¹ 2 of 6 seeds' prompt/expect content changed in `be4d8d3` (a registry-count generation fix);
the other 4 seeds still describe the current benchmark.
² `tools` changed in `be4d8d3` (`read_file` added) — this run's 0/6 reflects a task version
that had no way to observe the field it was told to keep; not a recovery-capability result.

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
    only `write_file` with no `read_file` to fall back on. **This was a genuine evaluator bug,
    not a model failure** — the task asked the model to keep a field it had no tool to observe.
    Fixed after this run in `be4d8d3` (`read_file` added to `tools`); see the superseded-results
    note above. `flaky_write_retry`'s whole 0/6 in the table above reflects the unsolvable
    version and should not be read as a recovery-capability result.

  These are the same content-writing failure already named in the dev pilot section, now seen at
  a larger sample. No new evaluator bug found in the *other* updates-family skeletons; per this
  task's own instruction, their difficulty is not retuned from this result.
- **`stopping` (1/12)** shows the same over-eager-write and format patterns as
  `no_op_correct_config` in the dev pilot. `flaky_write_retry`'s clean condition (0/3) is the
  same unsolvable-task artifact noted above, not a stopping-pattern finding.
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

## Protocol freeze on the control checkpoint (September 26, 2026)

The 7B results above are the engineering model's pipeline checks. Before the control run, the
benchmark was revised from dev-pilot evidence and re-frozen with the evaluation protocol; details
and rationale are in `STAGE8_DESIGN.md`, *Revision after the 7B pilot*:

- `read_file` now says its `N| ` prefixes must never be copied into `write_file` or `edit_file`;
  a `.json` write or edit that does not parse is refused and the file left unchanged.
- Every skeleton that edits a file offers `edit_file` (except `flaky_write_retry`).
- Dev grew from 15 to 64 tasks: four new skeletons (`max_timeout_service`, `delete_key`,
  `transient_list_failure`, `locked_config_stop`) and more seeds. Test is still 60 tasks; 27
  existing tasks changed only their tool list, none their prompt, workspace or expectation.
- The manifest freezes `model` (Qwen/Qwen3-4B-Instruct-2507 at revision `cdbee75f`, bf16, vLLM),
  `decoding` (T 0.7, top-p 0.8, top-k 20, min-p 0, max 2048 tokens: the model card's values),
  `seeds` [0, 1, 2] and `n_ctx` 32768. `bench --final` refuses a backend that does not match.

The 7B numbers above are not comparable with anything measured under this protocol.

### Local pipeline check on Qwen3-4B (September 27, 2026)

```sh
python -m agent_from_scratch.evals bench --backend llama_cpp --split dev --seed 0 \
  --model-path <unsloth/Qwen3-4B-Instruct-2507-GGUF @ a06e946, Q8_0> --output outputs/q3-4b-dev-s0
```

The control's weights as a Q8_0 GGUF through llama.cpp on the Mac, at the frozen sampling and
window, one seed. It checks the pipeline and task specs before the GPU session; it is **not** the
baseline (not bf16, not vLLM, one sample). Result: **34/64** in 11.6 min, 246 requests, 439,702
tokens. Measured prompt tokens equal the model's reported `prompt_tokens` on **243/243**
requests; the model emits parseable `<tool_call>` JSON (3 slips where the opening tag was
missing, each answered by the runtime's parse feedback). `<tool_call>` is a non-special token in
this tokenizer, so vLLM's default `skip_special_tokens` keeps it too.

| Skeleton | Passed | Failure pattern |
|---|---:|---|
| `single_field_edit` | 10/10 | — (the skeleton that exposed the `N| ` bug on the 7B) |
| `max_timeout_service` | 6/6 | — |
| `check_fix_recheck` | 9/10 | One fault run pasted `N| `-prefixed text into `edit_file`'s `old_text`, fixed the file by `write_file`, then hit `max_iterations` before replying |
| `locked_config_stop` | 5/6 | One unlocked run sent its last `edit_file` call without the opening `<tool_call>` tag |
| `transient_list_failure` | 3/6 | Clean 3/3, fault 0/3: after retrying the failed listing, the model answered with the file name instead of reading the file |
| `delete_key` | 1/6 | Used 4-space indentation in `old_text` (the file has 2) and repeated it despite `nearby_lines` returning the exact line |
| `pointer_lookup` | 0/10 | Reported the profile file's name without reading the profile (see the fix below) |
| `no_op_correct_config` | 0/10 | Read a correct config, wrote nothing, and replied `UPDATED` |

Two task-spec/error-message fixes followed, both recorded in `STAGE8_DESIGN.md`'s revision table:
`pointer_lookup`'s "report only its output filename" had a second reading (the profile file's
own name) and now asks for "the value of its output field" (re-run: 0/10 → 4/10; the other six
still skip reading the profile); and `edit_file`'s not-found error now names the `N| ` prefix
when `old_text` starts with one. The other failures are model behaviour and stay as training
targets: answering from a listing without reading, ignoring error feedback, and false claims.

**Known metric gap.** `false_completion` counts a claim only when the task's state checks fail,
so `no_op_correct_config`'s `UPDATED` (state correct, claim false) is scored as a failed answer
but not as a false completion. Read its answer failures alongside the false-completion rate.

### Runbook: control baseline on a rented GPU (as run, September 27, 2026)

Verified end to end on RunPod; every step below is what worked, with the traps that cost a retry.
Driven from the Mac with `runpodctl` (≥ 2.14; `runpodctl doctor` for the key, and an SSH key
registered with `runpodctl ssh add-key` **before** the pod boots).

**Pod.** vLLM 0.30.0 installs PyTorch 2.13 built for CUDA 13, so the host driver must support
CUDA 13 (`--min-cuda-version 13.0`); otherwise the pod boots and vLLM fails on first GPU use.
Any GPU with ≥ 24 GB works (bf16 weights ~8 GB plus KV cache at 32k); the run used an A40.

```sh
runpodctl pod create --name stage8-baseline \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --gpu-id "NVIDIA A40" --cloud-type SECURE --data-center-ids CA-MTL-1 \
  --min-cuda-version 13.0 --container-disk-in-gb 60 --ports "22/tcp" --wait --wait-timeout 15m
```

**Cost guard.** `runpodctl` 2.14 has no `--terminate-after`, and the pod's own
`RUNPOD_API_KEY` is `Unauthorized` for managing the pod (and the image ships `runpodctl` 1.x,
verb-first syntax), so the pod cannot stop itself. Run the guard from the Mac:
`sleep <secs> && runpodctl pod stop <pod-id>`, and remove the pod when done.

**Code.** A shallow clone of the frozen commit keeps `git rev-parse HEAD` (which run metadata
records) without the multi-GB history: `git clone --depth 1 --branch <branch> file://<repo>
stage8-src`, then tar it with `COPYFILE_DISABLE=1` (macOS `tar` otherwise adds `._*` files, and a
`._*.py` file is hashed as manifest drift) and `scp` it over. Check `git status` is clean on the pod.

**Install.** Ubuntu 24.04 refuses `pip install` into the system Python (PEP 668); use a venv:

```sh
python3 -m venv /root/venv && /root/venv/bin/pip install -q uv && . /root/venv/bin/activate
# VLLMClient's fields (/v1/models ModelCard id/root/max_model_len, /tokenize prompt/
# add_special_tokens -> count, completion top_k/min_p/seed/stop) were checked against 0.30.0.
uv pip install vllm==0.30.0 loguru -r stage8-src/agent_from_scratch/requirements-tools.txt
```

**Serve.** Two settings are required, not optional. The venv's `bin/` must be on `PATH` (FlashInfer
calls `ninja` by name), and `VLLM_USE_FLASHINFER_SAMPLER=0`, because FlashInfer JIT-compiles its
top-k/top-p sampler at startup and the image's `nvcc` is CUDA 12.8 against PyTorch's CUDA 13.
With it off, vLLM samples with its PyTorch implementation: the same `BENCH_DECODING`
parameters, a different kernel. **Every later adapter comparison must use this same server
command.** Detach fully (`setsid nohup … < /dev/null &`) or the SSH call stays open.

```sh
PATH=/root/venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup \
  vllm serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --dtype bfloat16 --max-model-len 32768 --enable-lora --max-lora-rank 64 --port 8000 \
  > vllm.log 2>&1 < /dev/null &
curl -s localhost:8000/v1/models   # ready when it lists the model with max_model_len 32768
```

1. **Smoke and validate** on four dev tasks, then check measured against served prompt tokens:

   ```sh
   python -m agent_from_scratch.evals bench --backend vllm --split dev --seed 0 \
     --tasks pointer_lookup-0,single_field_edit-0,check_fix_recheck-0-fault,locked_config_stop-0 \
     --output outputs/ctl-smoke
   python - <<'PY'
   import glob, json
   rows = [q for f in glob.glob("outputs/ctl-smoke/*/state/runs/*.jsonl") for line in open(f)
           for q in json.loads(line)["model_requests"] if q["status"] == "completed" and q["purpose"] == "agent"]
   bad = [q for q in rows if q["budget"]["prompt_tokens"] != q["usage"]["prompt_tokens"]]
   print(f"{len(bad)} of {len(rows)} requests where measured != served prompt tokens")
   PY
   ```

   Expect 0 mismatches and no `parse_errors`.
2. **Dev × 3:** `for s in 0 1 2; do python -m agent_from_scratch.evals bench --backend vllm
   --split dev --seed $s --output outputs/ctl-dev-s$s; done`, then `aggregate` the three runs.
   Review for evaluator/harness bugs before test; fix, re-freeze and re-run dev if any.
3. **Test × 3, once:** the same loop with `--split test --final` into `outputs/ctl-test-s$s`,
   then `aggregate`. `--final` refuses on manifest drift or a protocol mismatch.
4. Copy `outputs/` back, remove the pod, confirm `runpodctl pod list` is empty.

## Stage 8 control baseline (September 27, 2026)

Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f`, bf16, vLLM 0.30.0 (`VLLM_USE_FLASHINFER_SAMPLER=0`) on
one RunPod A40, frozen sampling (T 0.7, top-p 0.8, top-k 20), seeds 0/1/2, adapter off. Harness
and task set at commit `1e697ba`. All three test runs were `--final` with `manifest_drift` `{}`
and `protocol_mismatch` `[]`. Measured prompt tokens equalled the server's on **718/718** dev and
**778/778** test requests; 2 dev and 1 test parse errors; no backend errors. Wall time 7.6 min
(dev × 3) and 9.3 min (test × 3); the whole GPU session cost **$0.35**. Per-task records,
summaries, metadata and both aggregates are committed in `docs/baselines/stage8-control/`;
the paired comparisons of Stages 9–11 run against those task IDs.

| Split | Tasks | mean pass@1 | pass^k | pass@k | mixed (0 < c < 3) | false completion |
|---|---:|---:|---:|---:|---:|---:|
| **Test** | 60 | **0.667** | 0.583 | 0.717 | 8 | 1 / 72 claims |
| Dev | 64 | 0.542 | 0.469 | 0.641 | 11 | 14 / 87 claims |

| Family | Test pass@1 | Test pass^k | Dev pass@1 | Dev pass^k |
|---|---:|---:|---:|---:|
| inspection | 0.667 | 0.556 | 0.479 | 0.375 |
| updates | 0.889 | 0.722 | 0.708 | 0.625 |
| recovery | 1.000 | 1.000 | 0.771 | 0.688 |
| stopping | **0.000** | 0.000 | 0.208 | 0.188 |

| Test skeleton | pass@1 | Dev skeleton | pass@1 |
|---|---:|---|---:|
| `deep_chain_lookup` | 1.00 | `single_field_edit` | 1.00 |
| `append_list_item` | 1.00 | `max_timeout_service` | 1.00 |
| `transient_read_failure` | 1.00 | `check_fix_recheck` | 0.93 |
| `flaky_write_retry` | 1.00 | `locked_config_stop` | 0.56 |
| `rename_key_all_files` | 0.89 | `transient_list_failure` | 0.50 |
| `pointer_nested_edit` | 0.78 | `delete_key` | 0.22 |
| `grep_locate` | 0.56 | `pointer_lookup` | 0.17 |
| `sum_across_files` | 0.44 | `no_op_correct_config` | 0.00 |
| `ambiguous_choice_stop` | 0.00 | | |
| `missing_file_report` | 0.00 | | |

**Reading it.**

- **Stopping is the gap post-training should close.** On test the model never replies with only
  the required token: `missing_file_report` answers in prose and sometimes reports the backup's
  stale value it was told not to use; `ambiguous_choice_stop` ends its prose with `NEED_INPUT`
  (answer content passes, exact format fails, by design for reply-token deliverables). Dev shows
  the same in `no_op_correct_config` (false `UPDATED`, 0/10).
- **Test recovery is at the ceiling** (both recovery skeletons 1.00), so a gain cannot show there;
  dev recovery (0.77) still can. **Mixed-outcome tasks** (8 test, 11 dev) are the ones Stage 10's
  GRPO precondition needs.
- The local Q8_0 check predicted the dev pattern closely (same perfect, zero and low skeletons).
- **Known spec weakness, recorded, not changed:** `ambiguous_choice_stop`'s prompt already states
  "nothing marks either one active", so 16/18 runs answered without any tool call and fail
  `evidence`. It does not change the score (every run also fails the required exact reply), and
  test skeletons are not edited after test results. Revisit only through a new test split.
- **Known metric gap** (above): `false_completion` does not count a false claim over a correct
  state (`no_op_correct_config`'s `UPDATED`). The 14 dev false completions are `locked_config_stop`
  (8: `DONE` with no change) and `delete_key` (6: "removed" when the edit never landed).
