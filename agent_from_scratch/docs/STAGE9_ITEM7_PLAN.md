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

- [ ] **1. F1 diagnostic.** `bfcl --system-prompt ours`; one A40 session with the mix and
  teacher-v3 adapters; base, teacher-v3 and mix. Read the no-call turns of each.
- [ ] **2. F2 multi-turn tasks**, test first: `Task.followups`, the runner, the verifier, export.
- [ ] **3. New train shapes** (F3, F4), about 8 skeletons, each through the Item 1 gates, with
  the closest val and test shape recorded.
- [ ] **4. F5 val shapes** and the asked-instead-of-acting rate; the Stage 8 val tasks and all
  test tasks unchanged (fingerprints).
- [ ] **5. Teacher pass** on the new train tasks; export teacher-wide-v2 (teacher-wide plus the
  new shapes); check teacher-wide's 956 examples are inside.
- [ ] **6. F6**: When2Call rebalanced; `sft-mix-v2` built as Item 4 built mix.
- [ ] **7. Commit; one A100 session**: train mix-v2; serve base, teacher-v3, mix, mix-v2; epoch
  on val seed 0; val with three seeds; M0 and M1 are not sampled here (Stage 10 Item 1).
- [ ] **8. Choose M1 (F8)** and record in STAGE.md and this plan.

## 5. Cost

Projected: the diagnostic about \$0.50; the teacher on about 160 new tasks under \$0.50; the
A100 session about 7 hours, about \$13. About \$15 in all, inside the \$200 ceiling (about
\$17.7 spent).
