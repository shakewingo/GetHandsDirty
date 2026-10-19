# Stage 9 Item 7 — mix-v2: act when the answer is at hand, ask only when it is not

Written October 18, 2026 on branch `feat/stage9-wider-train-data` at `a47053c`, after Stage 9's
final test. Approved by the user the same day ("execute per the plan"). Earlier guides:
[STAGE9_ITEM4_PLAN.md](STAGE9_ITEM4_PLAN.md) (data and mix), [STAGE9_ITEM5_6_PLAN.md](STAGE9_ITEM5_6_PLAN.md)
(test and external test). The code is delegated to AI at the user's explicit request.

## 1. Why this item exists

Stage 9's test put mix first (0.774 against teacher-v3's 0.724 and the base's 0.419), but the
same session found three weaknesses that val could not see:

| Weakness | Evidence | Cause |
|---|---|---|
| **Mix asks instead of acting in multi-turn work** | BFCL multi-turn 0.285, below the base's 0.360 and teacher-v3's 0.445. Mix makes no call in 147 of 734 user turns (base 26, teacher-v3 0); 136 of those are follow-up questions about information the conversation already holds | When2Call's "ask when a parameter is missing" rows (646 of 2,000) are single questions; no training row shows information carried from an earlier turn or obtainable with a tool |
| **Targeted edits loop on JSON lists** | Test `append_list_item`: base 1.00, teacher-v3 0.11, mix 0.33; mix also falls on `pointer_nested_edit` (0.94 → 0.44) | The teacher's demonstrations prefer `edit_file`; on multi-line JSON the model copies `read_file`'s line prefixes into `old_text` or writes invalid JSON, and loops to the request limit |
| **Stopping stays weak** | Test stopping 0.438 for mix | Few stop decisions in the train pool, and none that contrast "stop" with "act" on the same records |

**Val is blind to the first one.** Every val task is a single turn whose information is all in
the workspace, so the set every choice is made on cannot show a model that asks instead of
acting. Stage 10's expert iteration samples the same kind of task, so it would neither see nor
fix the habit, and could entrench it. This item fixes the data and the val set first.

## 2. Goal and rules

**Goal.** A new candidate, mix-v2, trained with Item 3's recipe on data that teaches *when* to
act and *when* to ask, chosen on a val set extended with multi-turn tasks. M1 for Stage 10
becomes whichever of mix and mix-v2 wins on that val set.

**Rules that stay.**

- Stage 9's test and external numbers stand as reported. STAGE.md forbids adding a candidate
  after Item 6 has run, so mix-v2 is not reported on test here: it is evaluated once in
  Stage 11, under a new freeze, beside every other final candidate.
- Test and the When2Call and BFCL test sets are never trained on, and no new shape mirrors a
  val or test shape (the Item 4 rule, with the closest shape recorded).
- BFCL has been looked at. Any BFCL score of a model trained after it carries the label
  "after a BFCL-motivated change".

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| F1 | **Diagnostic first**: run BFCL again for base, teacher-v3 and mix with our `system.md` as the system message (Stage 9 used Qwen's default line). Diagnostic only; the reported numbers stand | every training row carried our system prompt; if mix's asking mostly disappears under it, the habit is tied to the prompt, and F6's cut can be milder |
| F2 | **Multi-turn tasks**: a task may carry follow-up user turns; the runner plays them in one session (the agent already takes `history`), the verifier scores the final state and the evidence across all turns, and `export` writes one example per model request of every turn | the BFCL failure is about information from an earlier turn; it must be trainable and measurable in our harness |
| F3 | **Contrast pairs for act versus ask**, built like clean/fault pairs: the same records, with the needed fact either present (in an earlier turn, a file, or a tool's result) or genuinely absent | teaches the decision, not a bias toward acting or toward asking |
| F4 | **JSON-structure edit shapes** that do not mirror `append_list_item` or `pointer_nested_edit`: removing an array element, adding a key in a nested object | demonstrations of edits that keep valid JSON |
| F5 | **A multi-turn val set**: new shapes, separate from F3's, about 40 tasks; `bench_summary` reports an **asked-instead-of-acting rate** (a final reply that asks or declines on a task whose answer is to act) | so the choice between mix and mix-v2 can see the failure |
| F6 | **Rebalance When2Call**: keep its calls and its declines, keep follow-up questions at about 10% of its rows (from about 32%) | F3 supplies the "ask" signal with a contrast; F1 may change the share |
| F7 | Item 3's recipe, epoch on val seed 0, mix proportions as in Item 4 otherwise | one change at a time: the data |
| F8 | **Choice**: val (old 64 plus F5's tasks), three seeds, base, teacher-v3, mix and mix-v2 on one server. M1 is mix-v2 if its val pass@1 is not lower than mix's and its asked-instead-of-acting rate is lower | a better multi-turn model that loses on the old val is not a clear win; both must hold |

## 4. Steps

- [x] **1. F1 diagnostic.** `bfcl --system-prompt ours`; one A40 session with the mix and
  teacher-v3 adapters; base, teacher-v3 and mix. Read the no-call turns of each.
- [x] **2. F2 multi-turn tasks**, test first: `Task.followups`, the runner, the verifier, export.
- [x] **3. New train shapes** (F3, F4), about 8 skeletons, each through the Item 1 gates, with
  the closest val and test shape recorded.
- [x] **4. F5 val shapes** and the asked-instead-of-acting rate; the Stage 8 val tasks and all
  test tasks unchanged (fingerprints).
- [x] **5. Teacher pass** on the new train tasks; export teacher-wide-v2 (teacher-wide plus the
  new shapes); check teacher-wide's 956 examples are inside.
- [x] **6. F6**: When2Call rebalanced; `sft-mix-v2` built as Item 4 built mix.
- [x] **7. Commit; one A100 session**: train mix-v2; serve base, teacher-v3, mix, mix-v2; epoch
  on val seed 0; val with three seeds; M0 and M1 are not sampled here (Stage 10 Item 1).
- [x] **8. Choose M1 (F8)** and record in STAGE.md and this plan.

## 5. Cost

Projected: the diagnostic about \$0.50; the teacher on about 160 new tasks under \$0.50; the
A100 session about 7 hours, about \$13. About \$15 in all, inside the \$200 ceiling (about
\$17.7 spent).

## 6. Record (October 19)

**Step 1 — F1 diagnostic.** One A40 (pod `agzp4hcz42844u`, about 45 minutes). BFCL multi-turn
base with our `system.md` as the system message:

| | Qwen default line (Stage 9) | our `system.md` | Turns without a call (of 734) |
|---|---:|---:|---|
| base | 0.360 | 0.395 | 26 → 32 |
| teacher-v3 | 0.445 | 0.450 | 0 → 5 |
| mix | 0.285 | 0.290 | 147 → 125 |

Our prompt hardly changes mix: the asking habit is in the weights, not tied to the prompt. F6's
cut stands as planned.

**Step 2 — multi-turn tasks.** `Task.followups`; the runner plays them in one session, each with
the conversation so far as history; the last turn's messages carry every turn's tool rows to
the verifier; request counts, usage and time add up over turns (`turn_metrics`); the task
finishes normally only if every turn does; the manifest records a hash of the follow-ups; and
`export` reads one trace per turn (`run_ids`) and marks each example with its `turn`. A
single-turn task's record is unchanged. Tests: `tests/test_multiturn.py`.

**Steps 3–4 — the shapes.** All pass the Item 1 gates; every earlier task is unchanged
(manifest comparison with Item 6's), and the Stage 8 freeze still matches. Act/ask pairs share
their records (seeds 2k and 2k+1), so each pair differs only in whether the fact is established.

| Shape | Split | Act / ask | Closest val or test shape | How it differs |
|---|---|---|---|---|
| `followup_same_record` | train | turn 1 names one customer, or only counts customers in a city; turn 2 changes "that customer's" phone, or replies NEED_INPUT | test `merge_record_requests` | the referent comes from an earlier turn, not from files |
| `followup_file_from_listing` | train | turn 1 names the newest report, or two reports; turn 2 appends to "that report" | val `transient_list_failure` | a listing establishes the referent; a write follows in a later turn |
| `lookup_then_act` | train | the customer's preferred day is in a file the prompt does not name, or absent | test `backup_record_lookup` | the needed file must be found by listing; absence means asking, not a fallback |
| `json_remove_array_item` | train | remove an item from a nested array, or NO_CHANGE when it is not there | test `append_list_item` | removal, not append, in a nested object with two arrays |
| `json_nested_key_add` | train | add a key in a nested object, or NO_CHANGE when it already holds | test `pointer_nested_edit` | adds a key; no pointer file |
| `followup_move_meeting` | val | turn 1 names the largest meeting, or counts meetings; turn 2 moves "it", or UNSURE | — | a val shape of the act/ask pattern, in another domain and token |
| `followup_carry_value` | val | turn 1 computes an amount; turn 2 writes "that amount" | — | acting is the only right answer; asking is a failure |

Train grows to 420 tasks and val to 104. Each run record gains `asked_instead`: a failed run
whose final reply asks or declines where acting was expected; the run summary counts it (in
`bench/run.py`, since `bench/verify.py` is part of the Stage 8 freeze). `write_file` already
refuses invalid JSON in a `.json` file, which is why mix's `append_list_item` failures were a
refused write followed by a false `DONE`.

**Step 5 — teacher.** A 10-task smoke passed 10. The full pass on the 100 new tasks, in four
processes: 98 passed. Both failures were ask variants of `followup_file_from_listing` where the
teacher took "that report" to mean the first of two; the verifier correctly rejected them. A
retry at high effort covered one. teacher-wide-v2: 1,335 examples covering 419 of 420 tasks
(107 from second turns), teacher-wide's 956 among them.

**Step 6 — data.** `public --cap question=200` on When2Call `train_pref` (seed 0): 912 calls,
888 declines, 200 questions (649 question rows skipped). `sft-mix-v2`: teacher-wide-v2 twice,
the capped When2Call set and Item 4's Toucan set; 7,672 rows, own share 0.348, 16.9M tokens,
longest row 29,549, every trained token decodes back to its completion. Commits `3126cb4` and
`2bed774`.

**Step 7 — session.** Pod `ng1771y6ej0syd`, one A100 80 GB in US-MD-1, \$1.79 per
hour, from 04:02 UTC October 19, at commit `2bed774`. Setup installed the pinned versions and
saw the A100, then stalled for 50 minutes in its last step, the SFT unit tests, with the GPU
idle. The container reports the host's 128 cores but has a quota of 13.6 CPUs, so torch ran
`test_sft_train`'s tiny CPU model on 128 threads that spun and were throttled (28,828 throttled
periods). With the threads capped at 4 the same 33 tests pass in 11 seconds. The stalled tests
were stopped without tripping the failure marker, and the session was restarted at 04:54 with
`OMP_NUM_THREADS=8`. The test now caps its own threads. The guard's hard stop moved from 14:02 to
16:00 UTC to keep the planned margin; the stall cost about \$1.50. Mix-v2 trained in 5.3 hours
(04:54 to about 10:14), 2,877 steps, mean loss by epoch 0.528, 0.378, 0.300 (mix: 0.515, 0.368,
0.291). The server was up and the epoch runs on val seed 0 had started by 10:16. Val seed 0 by
epoch: 51, 79, 85, so epoch 3. The val runs and BFCL finished at 10:39; the results were copied
back, the pod was deleted at about 10:43 and `runpodctl pod list --all` is empty. The session
cost about \$12.

Val, three seeds, 104 tasks (the old 64 and F5's 40); pass@1 is the mean over seeds:

| | Passed by seed | pass@1 | Old 64 | F5's 40 | Asked instead (of 312) | False completions |
|---|---|---:|---:|---:|---:|---:|
| base | 61, 64, 67 | 0.615 | 0.542 | 0.733 | 3 | 25 |
| teacher-v3 | 85, 89, 87 | 0.837 | 0.922 | 0.700 | 0 | 37 |
| mix | 86, 84, 90 | 0.833 | 0.943 | 0.658 | 8 | 26 |
| mix-v2 (epoch 3) | 85, 86, 86 | 0.824 | 0.901 | 0.700 | 0 | 18 |

Mix's old-64 score reproduces Item 4's 0.943. Against mix, mix-v2 loses on the old val in
`no_op_correct_config` (23 of 30 against 30) and `delete_key` (8 of 18 against 11), and gains
in `locked_config_stop` (17 against 14) and on F5's shapes (84 of 120 against 79).

**What the F5 shapes show.** Mix-v2 never asks where it should act: on the act variant of
`followup_move_meeting` its 6 failures are wrong moves, where mix's 9 are requests for which
meeting was meant. But no model passes the ask variant (0 of 30 for each): mix-v2 moves a
meeting anyway in 26 of 30, teacher-v3 in all 30, and the base asks in prose ("I cannot tell
which…") instead of the required `UNSURE`. The act/ask pairs in train moved mix-v2 toward
acting, not toward the decision.

**BFCL, after a BFCL-motivated change** (diagnostic; Qwen's default line, as in Stage 9):
mix-v2 0.360, with 48 of 734 turns without a call (mix 0.285 and 147; base 0.360 and 26;
teacher-v3 0.445 and 0).

**Step 8 — choice (F8).** Mix-v2's asked-instead-of-acting count is lower (0 against 8), but
its val pass@1 is lower too (0.824 against 0.833), and F8 needs both. **M1 stays mix.** The
gap is one task per seed and inside mix's own seed spread (84 to 90); F8 asks for "not lower",
so the rule decides it as written. Teacher-v3 (0.837) was not a candidate under F8.
