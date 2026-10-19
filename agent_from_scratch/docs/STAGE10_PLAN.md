# Stage 10 — expert iteration and a GRPO pilot, from M1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

Written October 19, 2026 on branch `feat/stage9-wider-train-data`, after Stage 9 Item 7 kept mix
as M1. Reviewed by the user the same day; tables renamed at their request. **Working rule:
each task's code diff is shown to the user before it is run.**

**Goal:** Answer Stage 10's two questions with measured runs: (1) is M_k a better data generator
than M_{k-1} over three rounds of expert iteration, and (2) at an equal number of rollouts, does
a GRPO update that also uses failed rollouts beat expert iteration's filtered SFT?

**Architecture:** Both arms start from M1 (mix, `sft-mix-r1/epoch-2`), sample the same task pool
under the same protocol and verifier, and are compared on val. Expert iteration reuses `bench`,
`export --track rft`, `mix` and `train` with one new `mix` option. GRPO adds one module
(`evals/sft/grpo.py`: reward, advantages, task rotation, rollout rows, one weighted optimizer
step) and a weighted step in `train.py`. The server stops while the trainer steps, then restarts
with the new adapter.

**Tech Stack:** Python stdlib, `unittest`, torch/transformers/peft (requirements-train.txt),
vLLM 0.30.0 on one A100 80 GB (RunPod), bash drivers.

**Spec:** [STAGE.md](STAGE.md), section "Stage 10 — expert iteration and outcome learning",
including "Conditional: a GRPO pilot". Earlier guides: [STAGE9_ITEM4_PLAN.md](STAGE9_ITEM4_PLAN.md)
(mix, the train pool) and [STAGE9_ITEM7_PLAN.md](STAGE9_ITEM7_PLAN.md) (M1 kept; 104-task val).

## 1. Starting point

| | Value | Source |
|---|---|---|
| M1 | mix, `outputs/sft-mix-r1/epoch-2` (hash `1388d076…`) | Stage 9 Item 4, kept by Item 7 |
| M1 on val (104 tasks, 3 seeds) | pass@1 0.833; old 64: 0.943; F5's 40: 0.658; asked instead 8 of 312 | Item 7 |
| M1's training data | `sft-mix-v1`: teacher-wide (956 examples, 320 tasks, `max_per_task` 1, `flawed_turns` drop) × 3, When2Call 2,000, Toucan 3,002; 7,870 rows | Item 4 |
| Train pool | 420 tasks in 21 skeletons (320 from Item 4, 100 from Item 7) | `specs("train")` |
| M0 on the pool | mean pass@1 0.298 on the first 80 and on Item 4's 240 | STAGE.md |
| Spent | about \$31 of the \$200 ceiling (\$17.7 before Item 7, about \$13 in Item 7) | records |

## 2. Decisions (for review)

| # | Decision | Why |
|---|---|---|
| S1 | **Pool: the 320 tasks of Stage 9 Item 4** (16 skeletons). Item 7's 100 tasks stay out of both arms: three act/ask contrast shapes (`followup_same_record`, `followup_file_from_listing`, `lookup_then_act`) and two JSON-edit shapes (`json_remove_array_item`, `json_nested_key_add`). Asking is still measured on val's 40 multi-turn tasks, but neither arm trains on it | STAGE.md fixes "the train pool as Stage 9 Item 4 leaves it". M1 never had demonstrations of Item 7's shapes, so including them adds a second change. **Alternative:** all 420, which would let self-training reach the act/ask shapes |
| S2 | **Sampling protocol: Stage 9 Item 2's.** Seeds 0–7, the Stage 8 server command with only `--lora-modules` changed, frozen decoding | the same protocol M0 was measured under |
| S3 | **Val: Item 7's 104 tasks, three seeds**, reporting the old 64 and F5's 40 separately, `asked_instead` and false completions | the set M1 was kept on |
| S4 | **Strict reward / verified:** GRPO's $r=1$ only when `passed` and `checks.format_exact` is not false (a task without an expected answer has no `format_exact`). Expert iteration uses `export --track rft`'s filter with teacher-wide's settings | as decided October 12; the export filter already drops loose answers and own errors |
| E1 | **Each round retrains from the base** with Item 3's recipe on `sft-mix-v1`'s rows **plus M_k's verified set once** (`export --track rft --max-per-task 1 --flawed-turns drop`, teacher-wide's settings). Sets do not accumulate across rounds | the only thing that changes between rounds is who generated the self data; retraining from the base avoids compounding drift (STaR). The teacher set stays, because self data alone collapsed to "read once, answer" (Item 3: 0.375) |
| E2 | **Epoch chosen on val seed 0**, the earlier on a tie (Item 3's D8) | unchanged recipe |
| E3 | **Three rounds** (M2, M3, M4), and M4 is also sampled on train, so each round's generator question has an answer. **Stop early** when a round's model is below M1 on val pass@1 by 0.05 or more (about 5 tasks per seed) | "2–3 rounds"; a collapse is a result and needs no further spending |
| G1 | **Go / no-go as STAGE.md, scaled to 320:** go when at least 120 tasks (3 in 8) have mixed outcomes under the strict reward, in at least two skeletons; the exploit review passes; the measured iteration cost fits | the pre-written threshold, at the same share |
| G2 | **Batch: 40 tasks × 8 seeds = 320 rollouts per iteration**, stratified 2–3 per skeleton, rotating so **8 iterations = one pass = 2,560 rollouts**, one expert-iteration round of sampling. **At most 3 passes** (24 steps, 7,680 rollouts) | STAGE.md: "keeping one pass equal to one expert-iteration round"; three passes pair with three rounds |
| G3 | **Update:** one AdamW step per iteration, **lr $1\times10^{-5}$ constant**, no warmup, no weight decay, clip 1.0, optimizer state carried in `optimizer.pt`; LoRA continues from M1's adapter (rank 16, alpha 32) | below the SFT rate of $10^{-4}$ (negative weights make the loss unbounded); carried state avoids sign-only steps |
| G4 | **Rows:** every agent request of every rollout in a mixed group, including requests that ended in a parse error. Prompt: the rendered `input_messages`. Target: **the text as sampled**, plus `<|im_end|>` when `finish_reason` is `stop`. Weight: the rollout's advantage. Loss normalized by the number of rows $N$ | the gradient must be on the tokens the model produced |
| G5 | **Server and trainer take turns on one A100:** stop vLLM, run `grpo step` (a fresh process that loads the base, the adapter and the optimizer state), restart vLLM with the new adapter. No runtime LoRA loading | the Stage 8 command stays as it is; every iteration is resumable from disk |
| G6 | **Pass chosen on val seed 0. Stop early** when a pass's val seed 0 is 10 or more tasks below M1's seed 0 in the same session, or when a pass's mean train reward is 0.10 below M1's in Item 1 | the collapse guard; a null result is valid |
| C1 | **Final comparison on one server:** M1, the chosen GRPO pass $r^*$ and $M_{1+r^*}$ (equal rollouts), plus the other M_k, val three seeds. Report mean pass@1, pass^k, per family and skeleton, and paired gains and losses per seed | the question is asked per rollout, so the pairing is by rollouts |
| C2 | **Kept candidates go to Stage 11**, not to test here (Stage 9's freeze rule) | as Item 7 |

## Global Constraints

- Train, val and test splits are unchanged; Stage 8 val and test fingerprints still match (`freeze --check`).
- Only `bench --split train` runs become training data; no val or test task is ever trained on.
- Evidence is never overwritten and lives outside the source package (`outputs/`, repository root).
- `train`'s defaults are the recipe: LoRA rank 16, alpha 32, dropout 0.05, lr 1e-4, 3 epochs, accumulate 8, seed 0.
- Every pod session has a self-delete guard and ends with `runpodctl pod list --all` empty; results are copied back before the guard fires.
- Pod setup runs the SFT unit tests with `OMP_NUM_THREADS=8` (Item 7's 13.6-CPU quota stall).
- Commits use `--author="claude <claude@noreply>"`.

## Review Focus

- A rollout that ended on a **parse error** stores its text as `raw_response.choices[0].message.content`, not `.text`: the GRPO rows must read both shapes (Task 2 test).
- A **truncated** sample (`finish_reason` `length`) must not get `<|im_end|>` appended, or the model is trained to stop where it did not (Task 2 test).
- A **seed run that crashed** leaves a group with fewer than 8 rollouts: advantages use the rollouts present and need at least 2 (Task 2 test).
- An iteration with **no mixed group** must still write an output (the unchanged adapter and optimizer state, `skipped: true`), so the driver's loop continues (Task 5 test).
- `mix --self` must not change the teacher set's repeat factor, or round 1 changes two things at once (Task 1 test).

---

## 3. Tasks — code (local, no GPU)

### Task 1: `mix --self` — add a self-generated set once

**Files:**
- Modify: `evals/sft/mix.py` (`combine`, `main`)
- Test: `tests/test_sft_mix.py`

**Interfaces:**
- Produces: `combine(own, public, *, share, seed, extra=()) -> tuple[list[dict], int]`; `extra` rows are appended once and are not counted when choosing `k`. CLI `--self DIR` (an `rft`-track export). Metadata gains `"self": source(...)` and `"self_share"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_sft_mix.py`:
  `test_self_rows_appear_once_and_leave_k_unchanged` (`combine(own=3 rows, public=[6 rows], share=1/3, seed=0, extra=4 rows)` returns the same `k` as without `extra`, and each extra id appears exactly once);
  `test_self_must_be_an_rft_export` (the CLI exits 2 when `--self` points at a `teacher` export).
- [ ] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_sft_mix -v`. Expected: the two new tests FAIL.
- [ ] **Step 3: Implement** `extra` in `combine` and `--self` in `main`; `self_share = len(extra) / len(rows)`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: let mix add a self-generated set once`.

### Task 2: `evals/sft/grpo.py` — reward, advantages, pool, rotation, rows

**Files:**
- Create: `evals/sft/grpo.py` (no torch import at module level)
- Test: `tests/test_sft_grpo.py`

**Interfaces:**
- Produces:
  - `ITEM7_SKELETONS: tuple[str, ...]`: the five Item 7 train skeletons, kept out of the pool.
  - `train_pool() -> dict[str, list[str]]`: skeleton → task IDs, from `bench.run.specs("train")` minus `ITEM7_SKELETONS` (320 IDs, 16 skeletons).
  - `iteration_tasks(pool, pass_index: int, iteration: int, per_iteration: int = 40) -> list[str]`: within a pass, each skeleton's IDs are shuffled with `random.Random(f"grpo:{pass_index}:{skeleton}")`, then interleaved round robin across skeletons in sorted order and cut into consecutive chunks of `per_iteration`; iteration is 0-based.
  - `reward(record: dict) -> int`: `1` if `record["passed"]` and `record["checks"].get("format_exact") is not False`, else `0`.
  - `advantages(rewards: list[int]) -> list[float] | None`: $(r_i-\bar r)/\sigma_r$ with the population $\sigma$; `None` when all rewards are equal or there are fewer than 2.
  - `sampled_rows(trace: dict, template) -> list[dict]`: one `{"request_index", "prompt", "completion"}` per agent request with a sampled text.
  - `collect_groups(run_dirs: list[Path]) -> tuple[list[dict], dict]`: rows `{"id", "prompt", "completion", "weight", "provenance"}` for every rollout of every mixed group, and metadata `{"rollouts", "tasks", "mixed", "all_pass", "all_fail", "mean_reward", "rows", "by_skeleton"}`. Refuses non-train runs, non-vLLM runs and duplicate (task, seed) pairs, as `export.collect` does.
  - CLI `python -m agent_from_scratch.evals grpo tasks (--pool | --pass P --iteration I)` prints comma-separated IDs (register `"grpo": "sft.grpo"` in `evals/__main__.py`).

- [ ] **Step 1: Write the failing tests** in `tests/test_sft_grpo.py`:
  - `test_pool_has_320_tasks_in_16_skeletons`: `sum(map(len, train_pool().values())) == 320` and no key in `ITEM7_SKELETONS`.
  - `test_one_pass_covers_the_pool_once`: the union of `iteration_tasks(pool, 1, i)` for `i` in 0–7 equals the pool, with no repeats, and every chunk has 40 IDs.
  - `test_rotation_differs_between_passes`: pass 1 iteration 0 differs from pass 2 iteration 0.
  - `test_strict_reward`: `passed` with `format_exact` false → 0; `passed` without the key → 1; not `passed` → 0.
  - `test_advantages_match_the_closed_form`: `advantages([1]+[0]*7)` gives `2.6458` (to 4 places) for the pass and `-0.3780` for each failure; `advantages([1]*8) is None`; `advantages([1]) is None`.
  - `test_parse_error_sample_is_read_from_the_chat_shape`: a request whose `raw_response` is `{"choices":[{"message":{"content":"<tool_call>{bad"},"finish_reason":"stop"}]}` yields a row with completion `"<tool_call>{bad<|im_end|>"`.
  - `test_truncated_sample_gets_no_end_token`: `finish_reason` `length` → the completion is the text alone.
  - `test_group_of_seven_still_counts`: 7 seed runs of one task, one request per rollout, rewards `[1,0,0,0,0,0,0]` → 7 rows, weights summing to 0 (within 1e-9).
  - `test_all_pass_group_is_skipped`: metadata `all_pass == 1`, no rows.
- [ ] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_sft_grpo -v`. Expected: FAIL (no module).
- [ ] **Step 3: Implement** the functions above. `sampled_rows` renders `request["input_messages"]` with `request["tools"]` through `render_chat` (as `export.examples_from_trace`), reads `choices[0]["text"]`, falling back to `choices[0]["message"]["content"]`, and appends `QWEN_END` only when `finish_reason == "stop"`. Multi-turn traces (`run_ids`) are read as `export.collect` reads them.
- [ ] **Step 4: Run** the GRPO tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: GRPO reward, advantages, task rotation and rollout rows`.

### Task 3: `aggregate --strict` — the GRPO reward's pass count

**Files:**
- Modify: `evals/bench/aggregate.py`
- Test: `tests/test_bench_aggregate.py`

**Interfaces:**
- Consumes: `grpo.reward(record) -> int` (Task 2). Import it inside `aggregate`, so `aggregate` stays torch-free.
- Produces: `aggregate(runs, seeds, *, strict=False)`; with `strict`, a task's `passes` counts `reward(record)`, not `passed`. The report records `"strict": true`.

- [ ] **Step 1: Write the failing test** `test_strict_counts_only_exact_answers`: two seeds of one task, both `passed`, one with `checks.format_exact` false. Expected `passes == 2` without `strict`, `1` with it, and `overall.mixed == 1` only under `strict`.
- [ ] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_bench_aggregate -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the flag (`--strict` on the CLI).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: aggregate can count the strict reward`.

### Task 4: `policy_step` — one weighted optimizer step

**Files:**
- Modify: `evals/sft/train.py`
- Test: `tests/test_sft_train.py` (`FitTests`, tiny model)

**Interfaces:**
- Produces: `policy_step(model, optimizer, rows: list[dict], *, lr: float) -> dict`. Each row is an `encode_example` output plus `"weight"`. Loss $= \frac{1}{N}\sum_i w_i \cdot \text{mean NLL}_i$, accumulated row by row (one forward per row, as `fit`), clipped at 1.0, one `optimizer.step()`. Returns `{"loss", "grad_norm", "rows", "tokens", "mean_weight"}`, with `grad_norm` before clipping.

- [ ] **Step 1: Write the failing tests:**
  - `test_unit_weights_match_one_fit_step`: same tiny adapted model and rows, weights all 1, `accumulate=len(rows)`. The adapter weights after `policy_step` equal those after one `fit` step at a constant rate (`warmup` irrelevant at one step; set `lr` equal), to `1e-6`.
  - `test_a_negative_weight_lowers_that_rows_likelihood`: two rows, weights `+1` and `-1`; after 5 steps at `lr=1e-2`, the second row's loss rose and the first row's fell.
  - `test_only_the_adapter_moves_in_a_policy_step`: base parameters unchanged.
- [ ] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_sft_train -v`. Expected: the three new tests FAIL.
- [ ] **Step 3: Implement** `policy_step` beside `fit`; `fit` is unchanged.
- [ ] **Step 4: Run.** Expected: PASS (all 17).
- [ ] **Step 5: Commit** `feat: a weighted policy-gradient step`.

### Task 5: `grpo step` — the per-iteration command

**Files:**
- Modify: `evals/sft/grpo.py` (CLI subcommand `step`)
- Test: `tests/test_sft_grpo.py`

**Interfaces:**
- Consumes: `collect_groups` (Task 2), `policy_step` (Task 4), `encode_example`, `hf_tokenize`.
- Produces: `python -m agent_from_scratch.evals grpo step --runs RUN... --adapter-in DIR --output DIR [--lr 1e-5]`. Loads the base at `BENCH_MODEL`'s revision in bf16 with gradient checkpointing, then `PeftModel.from_pretrained(base, adapter_in, is_trainable=True)`. Loads `adapter_in/optimizer.pt` if present (absent for M1). Writes `output/adapter_model.safetensors` + config, `output/optimizer.pt` and `output/step.json` (`collect_groups` metadata, `policy_step` result, lr, elapsed, git revision, the input adapter's hash).

- [ ] **Step 1: Write the failing tests:**
  - `test_step_refuses_an_existing_output_before_loading_a_model` (exit 2, torch never imported, the same check as `train`'s).
  - `test_no_mixed_group_copies_the_adapter_and_marks_the_step_skipped`: patch `collect_groups` to return no rows. `output/step.json` has `"skipped": true`, and the adapter file's bytes equal the input's.
- [ ] **Step 2: Run** the GRPO tests. Expected: FAIL.
- [ ] **Step 3: Implement** `step`.
- [ ] **Step 4: Run** all SFT tests: `python -m unittest agent_from_scratch.tests.test_sft_grpo agent_from_scratch.tests.test_sft_train agent_from_scratch.tests.test_sft_mix agent_from_scratch.tests.test_bench_aggregate -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: grpo step reloads, steps and saves the adapter and optimizer`.

### Task 6: Exploit review of the strict reward

**Files:**
- Create: `tests/test_grpo_exploits.py`

**Interfaces:**
- Consumes: `bench.run.specs`, `bench.verify.check`, `grpo.reward`; scripted results built as the existing `tests/test_bench_verify.py` builds them.

- [ ] **Step 1: Write the tests.** For seed 0 of each of the 16 pool skeletons, each cheap cheat must earn `reward == 0`:
  - `test_answer_only_earns_nothing`: no tool call; reply the expected answer token.
  - `test_read_everything_then_claim_earns_nothing_where_files_must_change`: list and read every workspace file, write nothing, reply the expected token. Asserted only on tasks whose expected files differ from the start.
  - `test_wrapped_answer_earns_nothing`: the scripted correct trajectory with the final reply `f"Done: {token}."`. `passed` may hold on advisory tasks, but `reward` is 0.
  - `test_no_change_claim_without_reading_earns_nothing`: on no-op variants, reply `NO_CHANGE` with no reads.
  - `test_fault_variant_without_meeting_the_fault_earns_nothing`: on fault variants, the clean variant's correct trajectory replayed so the injected fault is never observed (when the skeleton's fault fires on the first attempt, this cheat cannot be built; the test records it as skipped with that reason).
- [ ] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_grpo_exploits -v`. Expected: PASS. **A failure is a verifier loophole: stop and report it before any GPU work.**
- [ ] **Step 3: Commit** `test: the strict reward resists five cheap cheats`.

### Task 7: Pod drivers

**Files:**
- Create: `scripts/stage10_sample.sh`, `scripts/stage10_ei_round.sh`, `scripts/stage10_grpo.sh`
- Modify: `evals/README.md` (one paragraph each for `grpo tasks`, `grpo step`, `mix --self`, `aggregate --strict`)

**Interfaces:**
- `stage10_sample.sh NAME ADAPTER_NAME`: eight `bench --split train --tasks $(grpo tasks --pool) --seed s --adapter ADAPTER_NAME --output outputs/NAME-s$s` processes in parallel against the running server, then `aggregate` and `aggregate --strict`.
- `stage10_ei_round.sh K`: from M_K's samples: `export --track rft --max-per-task 1 --flawed-turns drop`, `mix --own teacher-wide --public when2call toucan --self rft-K --own-share 0.3333 --seed 0`, `train`, start the server with the three epochs, choose the epoch on val seed 0, run val seeds 1–2, sample M_{K+1} with `stage10_sample.sh`, pack the results.
- `stage10_grpo.sh PASSES`: first, val seed 0 for M1 on this server (G6's reference). Then for each pass and iteration 0–7: start the server with `grpo=CURRENT`, run 8 seeds on `grpo tasks --pass P --iteration I`, stop the server, `grpo step`, and make CURRENT the new adapter. After each pass: copy to `passes/pass-P` and run val seed 0. Applies G6's stop.
- Every script `set -euo pipefail`; a failed background process fails the script (wait on each PID separately, unlike Item 7's `session.sh`).

- [ ] **Step 1:** `bash -n` on each script. Expected: no output.
- [ ] **Step 2:** a dry run with `STAGE10_DRY=1`, which echoes each command instead of running it. Expected: for `stage10_grpo.sh 1`, eight iterations each listing 40 task IDs and the expected output paths.
- [ ] **Step 3: Commit** `feat: Stage 10 pod drivers`.

## 4. Steps — GPU sessions

Each session: A100 80 GB at \$1.79 per hour, a pod-side self-delete guard with a hard stop at
start + 1.5 × the projection, a local watcher that copies `item.tgz` back, and the pod deleted
at the end.

- [ ] **Session A — Item 1 and the GRPO smoke** (projected 2 hours, about \$4).
  1. Serve M1. Run `stage10_sample.sh m1-train M1`: 2,560 rollouts.
  2. Read beside M0, per skeleton: mean pass@1 (`passed` and strict), tasks with a verified
     trajectory, mixed tasks under the strict reward, and the `points_save_recovery` rollouts
     that reach `write_file` (M0: 5 of 160).
  3. One GRPO iteration on pass 1, iteration 0 as a smoke (`outputs/stage10-grpo-smoke`,
     never used again): measure sampling, step and restart minutes, and the rows and tokens per
     step.
  4. **Stop and report to the user**: the entry conditions (STAGE.md), G1's go/no-go, and the
     measured cost of a round and of a pass. Sessions B and C start only after that review.
- [ ] **Session B1–B3 — expert iteration rounds 1–3** (projected 7.5 hours each, about \$13.5
  each). Round K: `stage10_ei_round.sh K`. After each round, read the round table (below) and
  apply E3's stop.
- [ ] **Session C — GRPO pilot** (projected 3 hours per pass, about \$16 for three; runs in
  parallel with B on a second pod). `stage10_grpo.sh 3`. After each pass, read 10 reward-1
  rollouts for loopholes.
- [ ] **Session D — comparison (C1)** (projected 1 hour, about \$2). One server: M1, M2–M4 and
  the GRPO passes, val three seeds.
- [ ] **Record** in this plan and in STAGE.md; mark the candidates kept for Stage 11.

### Result tables

Two kinds of numbers are kept apart. **Train-task numbers** say how good a model is as a *data
generator* (how much usable training data it produces); they are never a capability claim,
because these tasks are trained on. **Val numbers** say how good a model is, and they are the
only numbers the two arms are compared on.

**Table 1 — Self-SFT: each model as a data generator.** Source: 320 train tasks × 8 attempts
(seeds 0–7), sampled at the start of the next round.

| Model | Attempt success rate | Tasks solved at least once (of 320) | Hard tasks solved at least once (of 60) | Tasks solved sometimes, not always (of 320) |
|---|---|---|---|---|
| M0 (base, reference) | | | | |
| M1 | | | | |
| M2 … M4 | | | | |

- *Attempt success rate*: the share of the 2,560 attempts that earn reward 1 (passed, with the
  answer in the exact requested form).
- *Tasks solved at least once*: tasks with at least one successful attempt, so the next round
  has a training example for them.
- *Hard tasks*: the 60 tasks of STAGE.md's three weak skeletons (`inventory_reconciliation`,
  `eligibility_record_update`, `points_save_recovery`; M0 solved 3, 5 and 1 of their 20).
- *Solved sometimes, not always*: tasks with between 1 and 7 successes in 8. GRPO learns only
  from these (G1).

**Table 2 — Val: how good each model is.** Source: the 104 val tasks × 3 seeds, all models on
one server in Session D.

| Model | Val success rate (104 tasks) | Single-turn tasks (64, Stage 8 val) | Multi-turn tasks (40, Item 7) | Asked instead of acting (of 312 runs) | Change from M1 (tasks per seed) |
|---|---|---|---|---|---|

- *Val success rate*: passed runs ÷ 312, averaged over the three seeds (mean pass@1).
- *Asked instead of acting*: runs that failed by asking or declining where the task expected an
  action.

**Table 3 — The comparison: GRPO against self-SFT at equal sampling.** Source: Table 2's runs.
After pass $r$, GRPO has used $2{,}560\,r$ rollouts, as has self-SFT's $M_{1+r}$.

| Rollouts used | Self-SFT model | Its val success rate | GRPO adapter | Its val success rate | Tasks GRPO wins / loses, per seed |
|---|---|---|---|---|---|
| 2,560 | M2 | | pass 1 | | |
| 5,120 | M3 | | pass 2 | | |
| 7,680 | M4 | | pass 3 | | |

**Table 4 — GRPO training log, one row per iteration** (a diagnostic, not a result): share of
the 320 attempts that earned reward 1; groups that were mixed, all success, all failure; rows
and tokens trained; loss; gradient norm; minutes.

## 5. Cost

| Session | Projection | Cost |
|---|---|---|
| A | 2 h | \$4 |
| B1–B3 | 3 × 7.5 h (training 5.7 h on about 19M tokens, sampling 1 h, val 0.5 h) | \$40 |
| C | 3 passes × 3 h (8 iterations of about 21 minutes: sampling 6, restart 3, step 12) | \$16 |
| D | 1 h | \$2 |
| **Stage 10** | | **about \$62**, bringing the total to about \$93 of \$200 |

Session A replaces these projections with measured ones before B and C start.
