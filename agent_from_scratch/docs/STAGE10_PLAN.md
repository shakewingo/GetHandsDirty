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
| Train pool | 420 tasks in 21 skeletons (320 from Item 4, 100 from Item 7); 20 tasks per skeleton | `specs("train")` |
| M0 on the pool | mean pass@1 0.298 on the first 80 and on Item 4's 240 | STAGE.md |
| Spent | about \$31 of the \$200 ceiling (\$17.7 before Item 7, about \$13 in Item 7) | records |

## 2. Decisions (for review)

| # | Decision | Why |
|---|---|---|
| S1 | **Pool: the whole train split, 420 tasks in 21 skeletons**, held fixed across rounds and passes. Item 7's 100 tasks (two multi-turn act/ask shapes, `followup_same_record` and `followup_file_from_listing`; one single-turn act/ask shape, `lookup_then_act`; two JSON-edit shapes) are reported as their own rows | decided with the user October 19, departing from STAGE.md's "as Stage 9 Item 4 leaves it". Val's 40 multi-turn tasks test act versus ask; without these shapes our own pool has no practice for it. M1's teacher demonstrations never covered them, so any gain on them comes from self-generated data alone. M1's data keeps When2Call's original share; Item 7's cut does not enter Stage 10 |
| S2 | **Sampling protocol: Stage 9 Item 2's.** Seeds 0–7, the Stage 8 server command with only `--lora-modules` changed, frozen decoding | the same protocol M0 was measured under |
| S3 | **Val: Item 7's 104 tasks, three seeds**, reporting the old 64 and F5's 40 separately, `asked_instead` and false completions | the set M1 was kept on |
| S4 | **Strict reward / verified:** GRPO's $r=1$ only when `passed` and `checks.format_exact` is not false (a task without an expected answer has no `format_exact`). Expert iteration uses `export --track rft`'s filter with teacher-wide's settings | as decided October 12; the export filter already drops loose answers and own errors |
| E1 | **Each round retrains from the base** with Item 3's recipe on `sft-mix-v1`'s rows **plus M_k's verified set once** (`export --track rft --max-per-task 1 --flawed-turns drop`, teacher-wide's settings). Sets do not accumulate across rounds | the only thing that changes between rounds is who generated the self data; retraining from the base avoids compounding drift (STaR). The teacher set stays, because self data alone collapsed to "read once, answer" (Item 3: 0.375) |
| E2 | **Epoch chosen on val seed 0**, the earlier on a tie (Item 3's D8) | unchanged recipe |
| E3 | **Three rounds** (M2, M3, M4), and M4 is also sampled on train, so each round's generator question has an answer. **Stop early** when a round's model is below M1 on val pass@1 by 0.05 or more (about 5 tasks per seed) | "2–3 rounds"; a collapse is a result and needs no further spending |
| G1 | **Go / no-go as STAGE.md, scaled to 420:** go when at least 158 tasks (3 in 8) have mixed outcomes under the strict reward, in at least two skeletons; the exploit review passes; the measured iteration cost fits | the pre-written threshold, at the same share |
| G2 | **Batch: 42 tasks × 8 seeds = 336 rollouts per iteration**, 2 per skeleton, rotating so **10 iterations = one pass = 3,360 rollouts**, one expert-iteration round of sampling. **At most 3 passes** (30 steps, 10,080 rollouts) | STAGE.md: "keeping one pass equal to one expert-iteration round"; three passes pair with three rounds |
| A1 | **Ask-side check before B and C:** in Session A, count M1's successes on the ask variants of the three Item 7 act/ask train shapes. If they are under 5% of those attempts, stop and review with the user | with no ask-side successes, self-SFT collects only act-side data and pushes toward acting, and GRPO's ask-side groups all fail and give no signal |
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

- [x] **Step 1: Write the failing tests** in `tests/test_sft_mix.py`:
  `test_self_rows_appear_once_and_leave_k_unchanged` (`combine(own=3 rows, public=[6 rows], share=1/3, seed=0, extra=4 rows)` returns the same `k` as without `extra`, and each extra id appears exactly once);
  `test_self_must_be_an_rft_export` (the CLI exits 2 when `--self` points at a `teacher` export).
- [x] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_sft_mix -v`. Expected: the two new tests FAIL.
- [x] **Step 3: Implement** `extra` in `combine` and `--self` in `main`; `self_share = len(extra) / len(rows)`.
- [x] **Step 4: Run** the same command. Expected: PASS.
- [x] **Step 5: Commit** `feat: let mix add a self-generated set once`.

### Task 2: `evals/sft/grpo.py` — reward, advantages, rotation, rows

**Files:**
- Create: `evals/sft/grpo.py` (no torch import at module level)
- Test: `tests/test_sft_grpo.py`

**Interfaces:**
- Produces:
  - `train_pool() -> dict[str, list[str]]`: skeleton → task IDs, every task of `bench.run.specs("train")` (420 IDs, 21 skeletons).
  - `iteration_tasks(pool, pass_index: int, iteration: int, per_iteration: int = 42) -> list[str]`: within a pass, each skeleton's IDs are shuffled with `random.Random(f"grpo:{pass_index}:{skeleton}")`, then interleaved round robin across skeletons in sorted order and cut into consecutive chunks of `per_iteration`; iteration is 0-based.
  - `reward(record: dict) -> int`: `1` if `record["passed"]` and `record["checks"].get("format_exact") is not False`, else `0`.
  - `advantages(rewards: list[int]) -> list[float] | None`: $(r_i-\bar r)/\sigma_r$ with the population $\sigma$; `None` when all rewards are equal or there are fewer than 2.
  - `sampled_rows(trace: dict, template) -> list[dict]`: one `{"request_index", "prompt", "completion"}` per agent request with a sampled text.
  - `collect_groups(run_dirs: list[Path]) -> tuple[list[dict], dict]`: rows `{"id", "prompt", "completion", "weight", "provenance"}` for every rollout of every mixed group, and metadata `{"rollouts", "tasks", "mixed", "all_pass", "all_fail", "mean_reward", "rows", "by_skeleton"}`. Refuses non-train runs, non-vLLM runs and duplicate (task, seed) pairs, as `export.collect` does.
  - CLI `python -m agent_from_scratch.evals grpo tasks (--pool | --pass P --iteration I)` prints comma-separated IDs (register `"grpo": "sft.grpo"` in `evals/__main__.py`).

- [x] **Step 1: Write the failing tests** in `tests/test_sft_grpo.py`:
  - `test_pool_is_the_whole_train_split`: 420 IDs in 21 skeletons, 20 each.
  - `test_one_pass_covers_the_pool_once`: the union of `iteration_tasks(pool, 1, i)` for `i` in 0–9 equals the pool, with no repeats; every chunk has 42 IDs, 2 from each skeleton.
  - `test_rotation_differs_between_passes`: pass 1 iteration 0 differs from pass 2 iteration 0.
  - `test_strict_reward`: `passed` with `format_exact` false → 0; `passed` without the key → 1; not `passed` → 0.
  - `test_advantages_match_the_closed_form`: `advantages([1]+[0]*7)` gives `2.6458` (to 4 places) for the pass and `-0.3780` for each failure; `advantages([1]*8) is None`; `advantages([1]) is None`.
  - `test_parse_error_sample_is_read_from_the_chat_shape`: a request whose `raw_response` is `{"choices":[{"message":{"content":"<tool_call>{bad"},"finish_reason":"stop"}]}` yields a row with completion `"<tool_call>{bad<|im_end|>"`.
  - `test_truncated_sample_gets_no_end_token`: `finish_reason` `length` → the completion is the text alone.
  - `test_group_of_seven_still_counts`: 7 seed runs of one task, one request per rollout, rewards `[1,0,0,0,0,0,0]` → 7 rows, weights summing to 0 (within 1e-9).
  - `test_all_pass_group_is_skipped`: metadata `all_pass == 1`, no rows.
- [x] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_sft_grpo -v`. Expected: FAIL (no module).
- [x] **Step 3: Implement** the functions above. `sampled_rows` renders `request["input_messages"]` with `request["tools"]` through `render_chat` (as `export.examples_from_trace`), reads `choices[0]["text"]`, falling back to `choices[0]["message"]["content"]`, and appends `QWEN_END` only when `finish_reason == "stop"`. Multi-turn traces (`run_ids`) are read as `export.collect` reads them.
- [x] **Step 4: Run** the GRPO tests. Expected: PASS.
- [x] **Step 5: Commit** `feat: GRPO reward, advantages, task rotation and rollout rows`.

### Task 3: `aggregate --strict` — the GRPO reward's pass count

**Files:**
- Modify: `evals/bench/aggregate.py`
- Test: `tests/test_bench_aggregate.py`

**Interfaces:**
- Consumes: `grpo.reward(record) -> int` (Task 2). Import it inside `aggregate`, so `aggregate` stays torch-free.
- Produces: `aggregate(runs, seeds, *, strict=False)`; with `strict`, a task's `passes` counts `reward(record)`, not `passed`. The report records `"strict": true`.

- [x] **Step 1: Write the failing test** `test_strict_counts_only_exact_answers`: two seeds of one task, both `passed`, one with `checks.format_exact` false. Expected `passes == 2` without `strict`, `1` with it, and `overall.mixed == 1` only under `strict`.
- [x] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_bench_aggregate -v`. Expected: FAIL.
- [x] **Step 3: Implement** the flag (`--strict` on the CLI).
- [x] **Step 4: Run.** Expected: PASS.
- [x] **Step 5: Commit** `feat: aggregate can count the strict reward`.

### Task 4: `policy_step` — one weighted optimizer step

**Files:**
- Modify: `evals/sft/train.py`
- Test: `tests/test_sft_train.py` (`FitTests`, tiny model)

**Interfaces:**
- Produces: `policy_step(model, optimizer, rows: list[dict], *, lr: float) -> dict`. Each row is an `encode_example` output plus `"weight"`. Loss $= \frac{1}{N}\sum_i w_i \cdot \text{mean NLL}_i$, accumulated row by row (one forward per row, as `fit`), clipped at 1.0, one `optimizer.step()`. Returns `{"loss", "grad_norm", "rows", "tokens", "mean_weight"}`, with `grad_norm` before clipping.

- [x] **Step 1: Write the failing tests:**
  - `test_unit_weights_match_one_fit_step`: same tiny adapted model and rows, weights all 1, `accumulate=len(rows)`. The adapter weights after `policy_step` equal those after one `fit` step at a constant rate (`warmup` irrelevant at one step; set `lr` equal), to `1e-6`.
  - `test_a_negative_weight_lowers_that_rows_likelihood`: two rows, weights `+1` and `-1`; after 5 steps at `lr=1e-2`, the second row's loss rose and the first row's fell.
  - `test_only_the_adapter_moves_in_a_policy_step`: base parameters unchanged.
- [x] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_sft_train -v`. Expected: the three new tests FAIL.
- [x] **Step 3: Implement** `policy_step` beside `fit`; `fit` is unchanged.
- [x] **Step 4: Run.** Expected: PASS (all 17).
- [x] **Step 5: Commit** `feat: a weighted policy-gradient step`.

### Task 5: `grpo step` — the per-iteration command

**Files:**
- Modify: `evals/sft/grpo.py` (CLI subcommand `step`)
- Modify: `evals/sft/tensorize.py`: `encode_example(..., require_end=True)`. GRPO passes `False`, because a sample cut off at the length limit has no stop token (found while writing this task: without it, one truncated rollout would crash the step)
- Test: `tests/test_sft_grpo.py` (also `test_saved_moments_land_on_the_same_parameters_after_a_reload`), `tests/test_sft_tensorize.py`

**Interfaces:**
- Consumes: `collect_groups` (Task 2), `policy_step` (Task 4), `encode_example`, `hf_tokenize`.
- Produces: `python -m agent_from_scratch.evals grpo step --runs RUN... --adapter-in DIR --output DIR [--lr 1e-5]`. Loads the base at `BENCH_MODEL`'s revision in bf16 with gradient checkpointing, then `PeftModel.from_pretrained(base, adapter_in, is_trainable=True)`. Loads `adapter_in/optimizer.pt` if present (absent for M1). Writes `output/adapter_model.safetensors` + config, `output/optimizer.pt` and `output/step.json` (`collect_groups` metadata, `policy_step` result, lr, elapsed, git revision, the input adapter's hash).

- [x] **Step 1: Write the failing tests:**
  - `test_step_refuses_an_existing_output_before_loading_a_model` (exit 2, torch never imported, the same check as `train`'s).
  - `test_no_mixed_group_copies_the_adapter_and_marks_the_step_skipped`: patch `collect_groups` to return no rows. `output/step.json` has `"skipped": true`, and the adapter file's bytes equal the input's.
- [x] **Step 2: Run** the GRPO tests. Expected: FAIL.
- [x] **Step 3: Implement** `step`.
- [x] **Step 4: Run** all SFT tests: `python -m unittest agent_from_scratch.tests.test_sft_grpo agent_from_scratch.tests.test_sft_train agent_from_scratch.tests.test_sft_mix agent_from_scratch.tests.test_bench_aggregate -v`. Expected: PASS.
- [x] **Step 5: Commit** `feat: grpo step reloads, steps and saves the adapter and optimizer`.

### Task 6: Exploit review of the strict reward

**Files:**
- Create: `tests/test_grpo_exploits.py`

**Interfaces:**
- Consumes: `bench.run.specs`, `bench.verify.check`, `grpo.reward`; scripted results built as the existing `tests/test_bench_verify.py` builds them.

- [x] **Step 1: Write the tests.** On seeds 0 and 1 of each of the 21 pool skeletons (seed 1 is the other side of each act/ask and no-op pair), each cheat runs through `run_case` with a `ScriptedModel` and must earn `reward == 0`:
  - `test_the_correct_solution_earns_the_reward`: the baseline; every scripted correct run earns 1, so a cheat's 0 means something.
  - `test_answer_only_earns_nothing`: no tool call; reply the expected answer in every turn. On no-op and ask variants this is "reply NO_CHANGE (or the ask token) without reading".
  - `test_doing_everything_but_the_change_earns_nothing`: the correct run with every `write_file` and `edit_file` call removed, on tasks whose correct run changes a file.
  - `test_a_wrapped_answer_earns_nothing`: the correct run with the final reply `f"Done: {token}."`. `passed` may hold on advisory tasks, but `reward` is 0.
  - `test_a_fault_left_unhandled_earns_nothing`: on fault variants, the correct run without each repeat of an earlier identical call (the retry). A skeleton that recovers another way (searching, not retrying) has no such repeat and is left out.
- [x] **Step 2: Run** `python -m unittest agent_from_scratch.tests.test_grpo_exploits -v`. Expected: PASS. **A failure is a verifier loophole: stop and report it before any GPU work.**
- [x] **Step 3: Commit** `test: the strict reward resists four cheap cheats`.

### Task 7: Pod drivers

**Files:**
- Create: `scripts/stage10_lib.sh` (sourced: the Stage 8 server command, `serve`/`stop_server`, parallel `bench` that fails if any process fails, `sample_train`, `choose_epoch`, `STAGE10_DRY=1`)
- Create: `scripts/stage10_sample.sh`, `scripts/stage10_ei_round.sh`, `scripts/stage10_grpo.sh`
- Modify: `evals/README.md` (a Stage 10 section: `aggregate --strict`, `mix --self`, `grpo tasks`, `grpo step`, the drivers)

**Interfaces** (all run from the repository root with the venv active and no server running; outputs under `outputs/`, logs under `/root/logs`):
- `stage10_sample.sh NAME ADAPTER_DIR`: serves the adapter, runs 8 seeds of the train split in parallel (`NAME-train-s0..7`), then `aggregate` and `aggregate --strict` (`NAME-train-aggregate.json`, `NAME-train-strict.json`). Session A: `stage10_sample.sh s10-m1 outputs/sft-mix-r1/epoch-2`.
- `stage10_ei_round.sh K`: from `s10-mK-train-s0..7`: `export --track rft --max-per-task 1 --flawed-turns drop` → `s10-rft-mK`; `mix --own teacher-wide --public when2call toucan --self s10-rft-mK` (default share, as mix-v1) → `s10-sft-eiK`; `train` → `s10-m{K+1}-r1`; serve the three epochs, choose on val seed 0, val seeds 1–2 for that epoch; sample M_{K+1} (`s10-m{K+1}-train-*`); copy the chosen epoch to `s10-m{K+1}`.
- `stage10_grpo.sh PASSES`: M1's val seed 0 on this server first (G6's reference). Then for each pass and iteration 0–9: serve `grpo=CURRENT`, 8 seeds on `grpo tasks --pass P --iteration I`, stop the server, `grpo step` → `s10-grpo-pP-iI-step`, which becomes CURRENT. After each pass: copy to `s10-grpo-pass-P`, val seed 0, apply G6. Resumable: an iteration whose step directory exists is skipped. Session A's smoke: `STAGE10_PREFIX=s10-grpo-smoke STAGE10_ITERATIONS=1 STAGE10_VAL=0 stage10_grpo.sh 1`.
- Every script `set -euo pipefail`; a failed background process fails the script (each PID is waited on, unlike Item 7's `session.sh`). Setup, the self-delete guard and packing the results stay in each session's own wrapper, as in Stage 9.

- [x] **Step 1:** `bash -n` on each script. Expected: no output.
- [x] **Step 2:** dry runs with `STAGE10_DRY=1`. Expected: `stage10_grpo.sh 1` prints ten iterations, each a `serve`, eight `bench` commands with the same 42 task IDs, `stop_server` and a `grpo step` whose `--adapter-in` is the previous step's output; `stage10_ei_round.sh 1` prints export, mix, train, serve with three epochs, three val runs, two more, eight train runs and both aggregates.
- [x] **Step 3: Commit** `feat: Stage 10 pod drivers`.

## 4. Steps — GPU sessions

Each session: A100 80 GB at \$1.79 per hour, a pod-side self-delete guard with a hard stop at
start + 1.5 × the projection, a local watcher that copies `item.tgz` back, and the pod deleted
at the end.

- [x] **Session A — Item 1 and the GRPO smoke** (projected 2 hours, about \$4; ran 30 minutes, about \$0.90; record in section 6).
  1. Serve M1. Run `stage10_sample.sh m1-train M1`: 3,360 rollouts.
  2. Read beside M0, per skeleton: mean pass@1 (`passed` and strict), tasks with a verified
     trajectory, mixed tasks under the strict reward, and the `points_save_recovery` rollouts
     that reach `write_file` (M0: 5 of 160); for A1, M1's successes on the ask variants of the
     three Item 7 act/ask shapes.
  3. One GRPO iteration on pass 1, iteration 0 as a smoke (`outputs/stage10-grpo-smoke`,
     never used again): measure sampling, step and restart minutes, and the rows and tokens per
     step.
  4. **Stop and report to the user**: the entry conditions (STAGE.md), G1's go/no-go, A1, and the
     measured cost of a round and of a pass. Sessions B and C start only after that review.
- [ ] **Session B1–B3 — expert iteration rounds 1–3** (projected 8 hours each, about \$14.5
  each). Round K: `stage10_ei_round.sh K`. After each round, fill Tables 1 and 2 (below) and
  apply E3's stop.
- [ ] **Session C — GRPO pilot** (projected 3.7 hours per pass, about \$20 for three; runs in
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

**Table 1 — Self-SFT: each model as a data generator.** Source: 420 train tasks × 8 attempts
(seeds 0–7), sampled at the start of the next round. Each model gets three rows: all 420 tasks,
the 320 of Stage 9 Item 4, and the 100 of Item 7.

| Model | Tasks | Attempt success rate | Tasks solved at least once | Hard tasks solved at least once (of 60) | Tasks solved sometimes, not always |
|---|---|---|---|---|---|
| M0 (base, reference) | 320 (Item 7's not sampled) | | | | |
| M1 | 420 / 320 / 100 | | | | |
| M2 … M4 | 420 / 320 / 100 | | | | |

- *Attempt success rate*: the share of the 3,360 attempts that earn reward 1 (passed, with the
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
After pass $r$, GRPO has used $3{,}360\,r$ rollouts, as has self-SFT's $M_{1+r}$.

| Rollouts used | Self-SFT model | Its val success rate | GRPO adapter | Its val success rate | Tasks GRPO wins / loses, per seed |
|---|---|---|---|---|---|
| 3,360 | M2 | | pass 1 | | |
| 6,720 | M3 | | pass 2 | | |
| 10,080 | M4 | | pass 3 | | |

**Table 4 — GRPO training log, one row per iteration** (a diagnostic, not a result): share of
the 336 attempts that earned reward 1; groups that were mixed, all success, all failure; rows
and tokens trained; loss; gradient norm; minutes.

## 5. Cost

| Session | Projection | Cost |
|---|---|---|
| A | 2 h | \$4 |
| B1–B3 | 3 × 8 h (training 6 h on about 20M tokens, sampling 1.3 h, val 0.5 h) | \$43 |
| C | 3 passes × 3.7 h (10 iterations of about 22 minutes: sampling 6, restart 3, step 13) | \$20 |
| D | 1 h | \$2 |
| **Stage 10** | | **about \$69**, bringing the total to about \$100 of \$200 |

Session A replaces these projections with measured ones before B and C start.

## 6. Record (October 19)

**Code.** Tasks 1–7 committed on `feat/stage10-ei-grpo` (`b466aff` … `f910f74`). The dry runs of
Task 7 caught two bash bugs before any GPU time (an empty array under `set -u` in bash 3.2, and a
`}` inside `${1:?…}` that ended the expansion early).

**Session A.** Pod `jdzjkbjxmecsdq`, one A100 80 GB, 16:58–17:26 UTC, about \$0.90. Setup
(Stage 9's versions, M1's hash `1388d076…`, 52 tests in 12 seconds with threads capped) took two
minutes; Item 1 (3,360 rollouts) and M1's val (three seeds) together took 20 minutes, far under
the projection. The session stopped at the gate, as A1 requires; no adapter was trained.

M1 on the train split, 8 seeds, strict reward:

| Tasks | Attempt success rate | Solved at least once | Solved sometimes, not always |
|---|---:|---:|---:|
| all 420 | 0.839 | 363 | 32 |
| Item 4's 320 | 0.999 | 320 | 1 |
| Item 7's 100 | 0.328 | 43 | 31 |

M1 on val, three seeds, on this server: 87, 82, 91 passed, pass@1 0.833, asked instead 9 of 312.
This reproduces Item 7's 0.833.

| Gate | Result |
|---|---|
| STAGE.md: M1's yield on the weak skeletons above M0's | yes: 1.000 on each of the three (M0: 0.044, 0.156, 0.006) |
| STAGE.md: `points_save_recovery` reaches a write | yes: 160 of 160 rollouts (M0: 5 of 160) |
| G1: at least 158 mixed tasks in at least two skeletons | **no-go**: 32 mixed, in 5 skeletons |
| A1: ask-side success at least 5% | **stop**: 1 of 240 (0.4%) |

How M1 fails Item 7's shapes. On the ask side of the two follow-up shapes it acts anyway
(`UPDATED` in 154 of 160). On `lookup_then_act` it replies `NEED_INPUT` without searching, on
both sides (0 of 160: on the act side the file exists, and on the ask side the answer lacks the
listing as evidence). On `json_remove_array_item` it claims `UPDATED` with the file unchanged, or
writes on a no-op (3 of 160 pass). The skill both arms would select is not in M1's samples: it
does not look before it decides.

**GRPO smoke** (one iteration on 42 tasks, never used again): 336 rollouts, mean reward 0.845,
2 mixed groups, 34 all pass, 6 all fail; 103 rows and 3,835 trained tokens; one step in 85
seconds (loss −0.013, gradient norm 0.78). The whole iteration took 4.3 minutes, sampling
included, against the 22 projected. The pipeline works end to end on the GPU.

**Decision pending with the user.** Both pre-registered stop conditions fired, so Sessions B and
C did not start.

## 7. Demonstrations first: mix-v3 (decided with the user, October 19)

Session A showed that M1 cannot be a data generator for what Stage 10 wants to improve: it
solves Item 4's 320 tasks every time, and on Item 7's shapes it lacks the skill (looking before
deciding) that either arm would select. STAGE.md's answer to "the weak skeletons are still
nearly all-fail" is to add demonstrations, then iterate. The teacher's demonstrations of Item 7's
100 tasks already exist (teacher-wide-v2); mix never saw them.

| # | Decision | Why |
|---|---|---|
| H1 | **mix-v3 = mix's data with teacher-wide replaced by teacher-wide-v2** (379 more examples, covering Item 7's train tasks), repeated 3 times as teacher-wide was in mix (`--own-share 0.445` gives $k = 3$), and When2Call and Toucan exactly as in mix: the original When2Call, not Item 7's capped one. 9,007 rows | one change from mix: the demonstrations. Unlike mix-v2, the When2Call ask rows stay, so mix-v3 against mix-v2 also separates Item 7's two changes |
| H2 | **Item 3's recipe**; the epoch chosen on val seed 0, the earlier on a tie | unchanged |
| H3 | **M1 choice, F8's rule:** mix-v3 replaces mix as M1 if its val pass@1 (three seeds) is not lower than mix's, both on one server. Otherwise M1 stays mix, and Stage 10 closes as a no-go (mix failed G1 and A1) | as Item 7 chose |
| H4 | **Gates on the new M1's Item 1:** STAGE.md's entry conditions and A1 (at least 5% ask-side success) gate both arms; **G1 is re-based to the 100 tasks that can move:** at least 38 mixed tasks (3 in 8 of 100), in at least two skeletons. It gates GRPO only. If A1 or an entry condition fails, Stage 10 closes as a no-go | Item 4's 320 tasks are saturated (M1: 0.999), so 158 of 420 cannot be met by a model that keeps them solved. Fixed here, before mix-v3 is sampled |
| H5 | **If the gates pass, Sessions B and C run as planned from the new M1.** Each self-SFT round's data is mix-v3's (teacher-wide-v2 ×3, When2Call, Toucan) plus M_k's verified set once; GRPO starts from mix-v3's adapter | the arms are unchanged; only M1 and its data change |

**Code.** `stage10_ei_round.sh` passes `--own-share` from `OWN_SHARE` (default 1/3, as before),
so a round built on teacher-wide-v2 keeps $k = 3$ (`TEACHER_WIDE=…-teacher-wide-v2 OWN_SHARE=0.445`).

**Steps.**

- [x] Build `sft-mix-v3` locally: `mix --own teacher-wide-v2 --public when2call toucan --own-share 0.445 --seed 0`; check $k = 3$ and 9,007 rows. Built as `outputs/stage10-sft-mix-v3`: 9,007 rows, $k = 3$, own share 0.4447.
- [x] **Session E** (one A100; record below): train mix-v3; serve mix and mix-v3's three epochs; choose the epoch on val seed 0; val seeds 1–2 for that epoch and seeds 0–2 for mix; apply H3. If mix-v3 becomes M1: Item 1 on it (420 tasks × 8 seeds), then wait at the gate for H4's reading. On go, the self-SFT rounds run on the same pod and the GRPO pilot on a second pod. Projected: 6.5 hours to the gate, about \$12.
- [ ] Record H3 and H4 here and in STAGE.md.

**Cost, revised with Session A's measurements** (an iteration took 4.3 minutes, not 22): Session
E to the gate about \$12; Sessions B about \$43; C about \$6 (three passes of about 50 minutes);
D about \$2. Stage 10 about \$64, bringing the total to about \$95 of \$200.

### Record of Session E (October 20–21)

**Two failed starts on a faulty host.** Pod `2xl3p7d3sxxuqd` (A100 SXM): training crashed in
epoch 2 twice, with different GPU errors at different points (`CUBLAS_STATUS_EXECUTION_FAILED`
at 04:15, `unspecified launch failure` at 08:53, October 20), and the host rebooted after each.
RunPod reported a network outage on that server. The crash logs are kept in
`outputs/stage10-20261020-session-e/crashes/`. The two runs cost about \$15. Lesson: the pod's
own `runpodctl` has no API key, so the pod-side self-delete guard never worked; the guard must
run on the Mac.

**The run.** Pod `69e7txgnniygbp`, from 08:56 UTC, October 20. mix-v3 trained 09:22–15:27 (about
2 hours per epoch). H3 on one server, val three seeds:

| | Passed by seed | pass@1 | Asked instead (of 312) |
|---|---|---:|---:|
| mix | 89, 87, 91 | 0.856 | 5 |
| mix-v3, epoch 2 (seed 0 by epoch: 95, 97, 93) | 97, 97, 96 | **0.929** | 0 |

**mix-v3 becomes M1.** Item 1 on it (420 tasks × 8 seeds, strict reward): attempt success rate
0.996, every task solved at least once, 9 mixed (7 of Item 4's 320, 2 of Item 7's 100). The ask
side of the three act/ask shapes: 240 of 240 (mix: 1 of 240).

**H4.** Entry conditions pass (the weak skeletons at 1.00, 1.00 and 0.94; `points_save_recovery`
writes in 160 of 160). A1 passes. **G1 fails: 2 mixed tasks among Item 7's 100 against 38, so the
GRPO pilot is a no-go** (STAGE.md: "M1 passes nearly everything on train: add tasks instead").
Self-SFT runs (Session B, rounds 1–3 from October 21, 02:56 UTC, on the same pod). Note for the
reading: M1's train yield is already at the ceiling, so the rounds measure whether
self-distillation holds or drifts on val, not whether yield grows.

The gate waited three hours on October 20 with no reading (the Mac-side watcher and guard had
stopped with the session) and the session ended at 18:59; the pod stood idle until 02:56 the next
day, about \$14.

## 8. Session B stopped; harder tasks next (October 21)

Round 1 of Session B started at 02:56 UTC on Session E's pod and was stopped about 20 minutes in,
at the user's decision: with M1 at 0.996 on train and 0.929 on val, the rounds could not show a
gain the seed noise would not hide. The pod was deleted after its logs were copied
(`outputs/stage10-20261021-session-b/`); `runpodctl pod list --all` is empty. The next step is
[STAGE10_HARD_TASKS_PLAN.md](STAGE10_HARD_TASKS_PLAN.md): shapes calibrated so that M1 succeeds
about half the time, then both arms on them.

## 9. Result (October 22)

Run on the harder pool of [STAGE10_HARD_TASKS_PLAN.md](STAGE10_HARD_TASKS_PLAN.md) (section 7 holds
the tables), with mix-v3 as M1. At every equal number of rollouts, self-SFT beat the GRPO pilot on
val. The best model, M4 (three self-SFT rounds), reached val 0.852 against M1's 0.752: the 40 hard
val tasks 0.225 → 0.617, the old 104 kept (0.955 → 0.942), false completions 33 → 3. GRPO raised its
train reward but lost on val (0.718–0.731), forgetting part of the old 104. As data generators the
models improve once and then flatten (pool success 0.58 → 0.74 → 0.75 → 0.77). Stage 10 is
complete; M4 goes to Stage 11 with M1 as the reference.
