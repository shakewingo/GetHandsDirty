# Results: post-training a 4B tool-using agent (Stages 9–11)

What the adapters trained in Stages 9 and 10 do on tests they were never chosen on. Every number
here comes from a run directory listed under *Evidence*; nothing below was selected on these
results. Plan: [STAGE11_PLAN.md](STAGE11_PLAN.md); Stage 9's own report: [STAGE.md](STAGE.md),
"Item 6 complete".

## Summary

- **Supervised fine-tuning is the gain.** On 140 test tasks of 14 shapes that training never
  saw, LoRA SFT on verified teacher trajectories lifts Qwen3-4B-Instruct from 0.419 to 0.724
  pass@1, and adding public tool-use data to 0.774 (Stage 9, reported then and unchanged here).
- **Stage 10's val gain does not transfer.** Three rounds of self-SFT (M4) raised val from 0.752
  to 0.852 over their starting point (mix-v3). On test M4 scores 0.745 against mix-v3's 0.738
  (39 tasks won, 36 lost; 95% interval of the difference −0.043 to +0.055), and on the new hard
  test 0.471 against 0.492. Neither difference is distinguishable from zero.
- **No Stage 10 or Item 7 candidate beats Stage 9's mix on test** (0.714–0.745 against 0.774),
  though that comparison crosses sessions (below).
- **Outside our harness** the later candidates trade the other way: on the local BFCL adaptation
  M4 scores 0.395 against mix's 0.285 and mix-v3's 0.325 (M4 over mix-v3: 26 entries won, 12
  lost, p = 0.034); When2Call is flat (judge mean 0.70–0.72).

This is a small local benchmark. None of it is a claim about general agent ability.

## The arms

| Arm | What it is | Chosen on |
|---|---|---|
| base | Qwen3-4B-Instruct-2507 (bf16, vLLM) | — |
| teacher-v3 | LoRA SFT on 255 DeepSeek trajectories that passed the verifier | val (Stage 9 Item 3) |
| mix | teacher data ×3, 2,000 When2Call and 3,002 Toucan rows | val (Stage 9 Item 4) |
| mix-v2 | mix's data with act/ask contrast pairs (Stage 9 Item 7) | kept for Stage 11 |
| mix-v3 | mix with the teacher's act/ask demonstrations; Stage 10's starting model, M1 | val (Stage 10) |
| M4 | three rounds of self-SFT from mix-v3 (each adds M_k's own verified rollouts) | val (Stage 10: 0.852) |

mix-v2, mix-v3 and M4 come after a change motivated by BFCL (Item 7's finding that mix asked
instead of acting), so their BFCL numbers are labelled that way. The GRPO adapters lost to
self-SFT on val and are not candidates.

## Protocol

- **Test**: 140 generated tasks on 14 shapes no training set contains, frozen with fingerprints.
  Session I wrote a new freeze from a clean clone of `637c310` (`dataset --check`: 640 of 640
  scripted correct runs pass, 0 of 640 answer-only and 0 of 640 wrong-answer cheats pass) and
  stopped unless its 140 test tasks matched Stage 9's freeze: they did, field for field.
- **Hard test**: 80 tasks on four shapes in formats no train or val shape uses (an ordered INI
  patch, a first-match fee table, a ledger audit with a failed read, a support dialogue), at the
  level calibrated on M1 with calibration seeds only (Session H). First run in Stage 11, so every
  arm is run on it.
- Every test and hard-test run: `bench --backend vllm --final --manifest`, seeds 0, 1 and 2,
  sampled decoding (temperature 0.7, top-p 0.8, top-k 20), no manifest drift, no protocol
  mismatch, no limit override. Each arm once.
- **Sessions.** Stage 9's base, teacher-v3 and mix ran on test on October 16 (one server). Session
  I (October 22, one A40, one server process with all five adapters) ran mix-v2, mix-v3 and M4 on
  test, all six arms on the hard test, and the external test for the three new arms.
- **Comparisons**: paired by task and seed; wins and losses, McNemar's exact p over the discordant
  pairs, and a 95% bootstrap interval of the pass@1 difference that resamples tasks with all three
  seeds of a task together (10,000 resamples, seed 0). Computed by
  `python -m agent_from_scratch.evals report`.

## Test (140 tasks, three seeds)

| | base | teacher-v3 | mix | mix-v2 | mix-v3 | M4 |
|---|---|---|---|---|---|---|
| session | S9 | S9 | S9 | S11 | S11 | S11 |
| **mean pass@1 / pass^3** | 0.419 / 0.36 | 0.724 / 0.64 | **0.774 / 0.71** | 0.714 / 0.66 | 0.738 / 0.61 | 0.745 / 0.66 |
| passed per seed (of 140) | 57, 58, 61 | 101, 106, 97 | 105, 111, 109 | 100, 102, 98 | 109, 99, 102 | 106, 101, 106 |
| Stage 8's 60 test tasks | 0.683 | 0.700 | 0.728 | 0.606 | **0.778** | 0.722 |
| Stage 9's 80 test tasks | 0.221 | 0.742 | **0.808** | 0.796 | 0.708 | 0.762 |
| false completions (of 420 runs) | 31 | 5 | 1 | 10 | 0 | 3 |
| val (144 tasks, Stage 10) | — | — | — | — | 0.752 | 0.852 |

By family, mean pass@1:

| | base | teacher-v3 | mix | mix-v2 | mix-v3 | M4 |
|---|---:|---:|---:|---:|---:|---:|
| inspection (38) | 0.439 | 0.851 | 0.833 | 0.825 | 0.719 | **0.895** |
| recovery (32) | 0.688 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| stopping (32) | 0.094 | 0.208 | **0.438** | 0.302 | 0.396 | 0.354 |
| updates (38) | 0.447 | 0.798 | 0.807 | 0.711 | **0.825** | 0.711 |

Paired comparisons (B against A; 420 task-seed pairs):

| B against A | Sessions | B wins | B loses | McNemar p | B − A, 95% interval |
|---|---|---:|---:|---:|---|
| M4 vs mix-v3 | same | 39 | 36 | 0.82 | +0.007 [−0.043, +0.055] |
| mix-v3 vs mix-v2 | same | 42 | 32 | 0.30 | +0.024 [−0.024, +0.074] |
| mix-v2 vs mix | across | 16 | 41 | 0.001 | −0.060 [−0.105, −0.017] |
| mix-v3 vs mix | across | 26 | 41 | 0.086 | −0.036 [−0.083, +0.014] |
| M4 vs mix | across | 21 | 33 | 0.13 | −0.029 [−0.069, +0.012] |

Test pass@1 by shape (6 tasks per Stage 8 shape, 20 per Stage 9 shape, three seeds):

| Shape | base | teacher-v3 | mix | mix-v2 | mix-v3 | M4 |
|---|---:|---:|---:|---:|---:|---:|
| `ambiguous_choice_stop` | 0.00 | 0.61 | 1.00 | 1.00 | 1.00 | 0.72 |
| `append_list_item` | 1.00 | 0.11 | 0.33 | 0.28 | 0.94 | 0.50 |
| `backup_record_lookup` | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `deep_chain_lookup` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `flaky_write_retry` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `grep_locate` | 0.56 | 1.00 | 0.72 | 0.56 | 0.83 | 1.00 |
| `merge_record_requests` | 0.00 | 0.92 | 1.00 | 1.00 | 0.97 | 0.90 |
| `missing_file_report` | 0.00 | 0.00 | 0.56 | 0.00 | 0.61 | 0.67 |
| `pointer_nested_edit` | 0.94 | 0.94 | 0.44 | 0.67 | 0.56 | 0.28 |
| `rename_key_all_files` | 0.89 | 0.94 | 1.00 | 0.22 | 0.50 | 0.72 |
| `sum_across_files` | 0.44 | 0.39 | 0.22 | 0.33 | 0.33 | 0.33 |
| `three_source_join` | 0.23 | 0.90 | 1.00 | 1.00 | 0.72 | 1.00 |
| `transient_read_failure` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_complete_total` | 0.15 | 0.15 | 0.23 | 0.18 | 0.15 | 0.15 |

## Hard test (80 tasks, three seeds, all arms in one session)

| | base | teacher-v3 | mix | mix-v2 | mix-v3 | M4 |
|---|---:|---:|---:|---:|---:|---:|
| **mean pass@1 / pass^3** | 0.275 / 0.12 | 0.267 / 0.15 | 0.429 / 0.25 | 0.442 / 0.34 | **0.492** / 0.33 | 0.471 / **0.34** |
| passed per seed (of 80) | 29, 18, 19 | 25, 19, 20 | 36, 31, 36 | 36, 38, 32 | 40, 35, 43 | 40, 38, 35 |
| `config_patch_plan` (L1) | 0.100 | 0.550 | 0.517 | 0.500 | **0.667** | 0.450 |
| `shipping_fee_table` (L1) | **0.433** | 0.033 | 0.167 | 0.167 | 0.367 | 0.250 |
| `ledger_audit_fault` (L2) | 0.000 | 0.100 | 0.600 | 0.583 | 0.433 | **0.683** |
| `support_dialogue` (L4) | **0.567** | 0.383 | 0.433 | 0.517 | 0.500 | 0.500 |
| false completions (of 240) | 101 | 156 | 106 | 100 | 82 | 91 |

| B against A (240 pairs) | B wins | B loses | McNemar p | B − A, 95% interval |
|---|---:|---:|---:|---|
| M4 vs mix-v3 | 36 | 41 | 0.65 | −0.021 [−0.113, +0.071] |
| mix-v3 vs mix-v2 | 34 | 22 | 0.14 | +0.050 [−0.021, +0.121] |
| M4 vs mix | 35 | 25 | 0.25 | +0.042 [−0.033, +0.117] |
| mix vs base | 82 | 45 | 0.001 | +0.154 [+0.021, +0.287] |

The hard test separates trained from untrained models only weakly (mix over base, p = 0.001), and
no trained arm from another. Each shape ranks the arms differently, and the base model is best on
two of the four shapes. Every arm claims a finished task it did not finish in a third or more of
its runs, so these shapes still mostly measure where all six models fail.

## External test

| | base | teacher-v3 | mix | mix-v2 | mix-v3 | M4 |
|---|---|---|---|---|---|---|
| session | S9 | S9 | S9 | S11 | S11 | S11 |
| When2Call MCQ accuracy (normalised) | 0.456 (0.500) | 0.486 (0.535) | 0.694 (0.675)* | 0.699 (0.660)* | 0.691 (0.671)* | 0.675 (0.658)* |
| When2Call hallucination rate | 0.225 | 0.233 | 0.027* | 0.035* | 0.039* | 0.035* |
| When2Call judge, mean of three kinds | 0.533 | 0.500 | 0.697* | 0.720* | 0.713* | 0.723* |
| judge: tool call / ask for info / cannot answer | 0.93 / 0.21 / 0.46 | 0.98 / 0.18 / 0.34 | 0.56 / 0.86 / 0.67 | 0.79 / 0.73 / 0.64 | 0.61 / 0.88 / 0.65 | 0.59 / 0.86 / 0.72 |
| BFCL multi-turn base (local adaptation) | 0.360 | **0.445** | 0.285 | 0.385† | 0.325† | 0.395† |

\* In-distribution: these adapters trained on When2Call's `train_pref` rows, from the same
generator. † After a BFCL-motivated change. When2Call's judge is DeepSeek (`deepseek-flash`, both
sessions), not the repository's judge; the alias may name a newer release on October 22 than on
October 16. BFCL, paired by entry within Session I: M4 over mix-v3 26 won and 12 lost (p = 0.034);
mix-v3 over mix-v2 12 won and 24 lost (p = 0.065). mix-v2's BFCL score was 0.360 in Item 7's
diagnostic and is 0.385 here, both greedy: the same adapter differs by 0.025 across server
sessions.

## What the evidence shows

- **The SFT gain is real; the later gains are not visible on held-out tests.** Every trained arm
  beats the base on test by about 0.3 (0.295 to 0.355). Among the trained arms, no difference on test or the hard
  test is distinguishable from zero within a session.
- **Val overstated Stage 10.** M4 was chosen on val among seven candidates after three rounds that
  were themselves not monotone on val (0.785, 0.729, 0.852); the best of several noisy val scores
  is biased upwards. On test M4's 0.745 is close to mix-v3's 0.738. Stage 10's own record already
  warned that part of the val gain might be retraining variance; the test supports that reading.
- **Each later change moves skills around rather than adding them.** mix-v3 recovers the JSON-list
  edit Stage 9 lost (`append_list_item` 0.33 → 0.94) but loses ground on Stage 9's own test shapes
  (0.808 → 0.708); M4 is the best on inspection (0.895) and on `missing_file_report` (0.67) but
  falls on `ambiguous_choice_stop` (1.00 → 0.72) and `pointer_nested_edit` (0.28). With 6 tasks per
  Stage 8 shape, single-shape differences are noisy.
- **Outside the harness, mix-v2 and M4 ask less where they should act.** mix asked for
  information it already had in BFCL's multi-turn conversations (Stage 9); mix-v2 and M4 recover
  about two thirds of the gap to teacher-v3 (0.385 and 0.395, against mix's 0.285 and teacher-v3's
  0.445). On When2Call only mix-v2 shows the trade: more correct tool calls (0.79 against mix's
  0.56) and fewer correct requests for information (0.73 against 0.86); mix-v3 and M4 score close
  to mix.
- **Across sessions, the later candidates sit below mix on test** (−0.029 to −0.060). Only
  mix-v2's interval excludes zero, and across-session comparisons carry server-session variance
  that one run per arm cannot measure (BFCL shows 0.025 for a greedy run).

## Limitations

- One training run per candidate; epochs and rounds were chosen on val.
- The test is 140 tasks on 14 shapes; the hard test 80 tasks on 4 shapes, with levels calibrated
  on M1 (mix-v3), which may favour it. `ledger_audit_fault`'s levels are not monotone: its
  mismatch share is sparsest at the chosen L2 and its higher levels exceed the 8-calls-per-reply
  limit ([STAGE11_PLAN.md](STAGE11_PLAN.md), Session H record).
- Comparisons with Stage 9's base, teacher-v3 and mix on test cross server sessions.
- BFCL is a local adaptation (our template and decoding, one category), not comparable with the
  leaderboard. When2Call scores are in-distribution for every trained arm except teacher-v3.
- The agent's harness (tools, limits, prompts) is fixed; a different harness could rank the
  arms differently.

## Cost

| Item | Cost |
|---|---|
| Stage 9 (all items) | about \$17.7 |
| Stage 10 (sessions A–G) | about \$100 |
| Stage 11 Session H (calibration, A40) | about \$1.30, of which about \$0.95 idle before setup |
| Stage 11 Session I (final runs, A40) | about \$8.80, of which about \$6.70 idle after the session ended: the Mac's lid closed on battery and froze the cost guard |
| When2Call judging (DeepSeek) and Task 4 / notebook API calls | a few dollars |

The whole project stayed well inside the \$200 ceiling.

## Evidence

- Session I: `outputs/stage11-20261022-session-i/` (the freeze `outputs/s11-data`, test runs
  `s11-test-*`, hard-test runs `s11-hard-*`, external runs `s11-ext-*` including the judge,
  logs and the timeline).
- Session H: `outputs/stage11-20261022-session-h/` (calibration runs, report and chosen levels).
- Stage 9: `outputs/stage9-item6-20261016-session/` (test runs `test-*`, external `ext-*`).
- Stage 10: `outputs/stage10-20261021-session-fg/` (val comparison `s10g-compare`).
- To recompute the tables: [evals/README.md](../evals/README.md), "Reproduce Stage 11".
