# Stage 9 Item 4 — more training data, chosen on val

Written October 14, 2026 on branch `feat/stage9-adapt-on-sft-examples` at `bac2cbe` (414 tests
passing), for review before any code is written. Spec: [STAGE.md](STAGE.md), Stage 9
("Item 4 — more training data, chosen on val"). Earlier guides:
[STAGE9_ITEM2_PLAN.md](STAGE9_ITEM2_PLAN.md) (export), [STAGE9_ITEM3_PLAN.md](STAGE9_ITEM3_PLAN.md)
(recipe).

> **Who wrote what (D1).** The coding rules in STAGE.md make the environment and the
> training-data logic the user's to write. At the user's explicit request, as in Items 2 and 3,
> AI wrote Item 4's code.

**Status: done (October 15, 2026).** Final candidates for Item 6: teacher-v3 and mix; M1 for
Stage 10: mix. Results in [STAGE.md](STAGE.md), Stage 9 ("Item 4 complete"); the record of every
step is in sections 11 and 12. Commits `edc56d7`, `9638b34`, `33867ec` and the results commit.

**Goal.** Train two new candidates with Item 3's recipe, teacher-wide (more train shapes) and
mix (teacher-wide plus public tool-use data), compare them with teacher-v3 and the base on val,
and name the final candidates for Item 6 and the M1 that Stage 10 starts from.

**Not in Item 4.**

- The external test and its scorers: Item 5. The test split: Item 6.
- M1's yield on the train pool: Stage 10 Item 1. (M0's yield on the *new* shapes is optional
  here, D12, because the server is already up.)
- Any change to the verifier, `prompts/`, the chat template, `AgentLimits`, `BENCH_*`, the
  existing 80 train tasks, or any val or test task.

## 1. What Item 4 delivers

| Requirement (STAGE.md) | What satisfies it |
|---|---|
| 12–16 new train skeletons, the 80 existing tasks unchanged | 12 skeletons, 20 tasks each: 320 train tasks (section 4) |
| Cover leaving a correct state untouched, and reporting a missing file | 4 shapes with a no-op outcome, 1 missing-file shape, 1 moved-file search |
| ≥2 outcomes per skeleton chosen by the seed; clean/fault pairs for recovery | every row of section 4 |
| Closest val and test shape recorded, none mirrored | section 4, columns "Closest" and "How it differs" |
| Item 1 gates pass | `dataset --check` on a new versioned dataset, plus per-shape negative tests |
| Decide on `points_save_recovery` | D4: unchanged |
| Teacher on the new tasks, Item 2 export flags | `sft-teacher-wide` |
| Public data: licence, 20 rows read, converter, round-trip and window filters | `public` command, two converted datasets |
| The approved mix (2,000 / 3,000 / own data a third), ≤2 changes on val | `mix` command, `sft-mix-v1` |
| Check what public data can break | `format_exact` and false completions, mix against teacher-wide |
| teacher-wide and mix on val, three seeds, one server with the base | Session A |
| Final candidates named, at most three | section 8 |
| Confirm Item 3's recipe | D8 |

```
new skeletons ─► dataset --check ─► bench --backend teacher (240 new tasks) ─┐
Item 2 teacher runs (80 tasks) ──────────────────────────────────────────────┴─► export ─► sft-teacher-wide ─┐
When2Call train_sft ─► public ─► sft-public-when2call ─┐                                                     │
Toucan-1.5M SFT ─────► public ─► sft-public-toucan ────┴─► mix (+ sft-teacher-wide, upsampled) ─► sft-mix-v1 ┤
                                                                                                             ▼
                                          train (Item 3 recipe) ─► adapters ─► vLLM ─► bench --split val ─► aggregate, compare
```

## 2. Facts the plan rests on

Checked in the code or on the dataset pages, not assumed. Items marked *(to verify)* are
checked again in the step named.

1. **Val has little headroom.** teacher-v3 passes 0.953 of val. The remaining failures are in
   stopping (0.833) and almost all are `no_op_correct_config`, which replies `UPDATED` without
   writing. On the other families val can show a loss, but hardly a gain. Section 8 is written
   around this.
2. **The train pool has no inspection skeleton.** The four train shapes are 2 recovery
   (`inventory_reconciliation`, `points_save_recovery`), 1 updates, 1 stopping.
3. **Adding skeletons leaves val and test untouched.** `specs()` iterates `SKELETONS` in
   registration order, and `test_dev_test_tasks_and_protocol_match_the_frozen_manifest`
   compares only val and test tasks with the frozen manifest. New `.py` changes drift
   `source_sha256`, which already drifted in Items 2–3; the versioned freeze happens in Item 6.
4. **The verifier already expresses what the new shapes need.** A no-op outcome is
   `process=("no_write_attempts",)` with a `required` answer. A read-back is an `evidence`
   string that exists only after the write (as in `eligibility_record_update`). A failed
   read of a named path is `fault_signal` plus `fault_path` (as in `backup_record_lookup`).
   For train tasks with expected files, `run_case` also requires the decision sources to be
   observed before the first write (`observed_before_write`).
5. **Tests to update.** `TrainingTaskTests` asserts 80 train tasks and four names; it runs the
   correct / answer-only / wrong-answer / reproducibility / clean-fault-pair gates on every
   train task, so the new shapes inherit them once the expected count and names change.
6. **`train` takes one dataset directory** and reads `track`, `pilot`, `flawed_turns`,
   `max_per_task` and `chat_template_sha256` from its `metadata.json`, and `provenance.skeleton`,
   `run_dir` and `task_id` from each example (for `--max-per-skeleton`). A converted or mixed
   dataset that writes those keys trains with `train.py` unchanged.
7. **Training time.** Item 3 measured 1.33 s per row on an A40 for teacher rows averaging
   1,769 tokens, with gradient checkpointing. Time is close to linear in tokens, so the token
   counts in Step 10 give the estimate before renting.
8. **When2Call `train_sft`** *(to verify, Step 6)*: columns `tools` (a list of JSON strings, with
   BFCL-style parameter type `"dict"`) and `messages` (`user`, `assistant` only). The first 58
   rows are all follow-up questions or refusals, with no tool call, so the file is probably
   ordered by category and must be sampled across the whole file. How a tool call is written
   is not yet known. Licence CC-BY-4.0.
9. **Toucan-1.5M `SFT` config** *(to verify, Step 6)*: 119k rows; `messages` is a JSON string
   with roles `user`, `assistant`, `tool_call` and `tool_response`; a call is a Python-dict
   string `{'name': ..., 'arguments': '<json>'}`; `tools` is a JSON string of OpenAI-style
   function objects. Trajectories come from Qwen3-32B, Kimi-K2 and GPT-OSS-120B; failed calls
   were kept on purpose. Licence Apache-2.0.
10. **`docs/` is git-ignored.** This file needs `git add -f`, as Items 2–3's plans did.

## 3. Decisions

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Who writes the code | AI, as in Items 2–3, except the parts you name | see the note at the top |
| D2 | New train shapes | 12 skeletons, 3 per family, 20 tasks each (20 seeds, or 10 seeds × clean/fault): 240 new tasks, 320 in all | the low end of 12–16 keeps the teacher pass and review small; three per family fills the empty inspection family |
| D3 | Answer tokens | A new shape uses no token that a val or test shape uses, except tokens already in the train pool (`UPDATED`, `NEED_INPUT`, `NONE`, `BLOCKED`, `SAVED`). So `NO_CHANGE`, not `UNCHANGED`; `NOT_FOUND`, not `MISSING` | a val gain on `no_op_correct_config` should come from the decision, not from having learned val's token |
| D4 | `points_save_recovery` | Leave it unchanged | changing it breaks comparison with M0's table and with teacher-v3's data. The base's habit (guess a path, fail, give up) is taught directly by `moved_record_search` (R1). Stage 10 Item 1 measures whether M1 now reaches a write |
| D5 | teacher-wide's data | Re-export the Item 2 teacher runs together with the new runs, with the Item 2 flags (`--flawed-turns drop --max-per-task 1`). The new runs cover only the 240 new tasks (`bench --tasks`) | the 255 teacher-v3 examples must come out as a subset, so teacher-wide differs from teacher-v3 only by the new shapes. Step 5 checks it by example ID |
| D6 | Public row conversion | Our `system.md` as the system message; the row's own tools rendered by our template; parameter type `"dict"` mapped to `"object"`; each tool response wrapped in our tool-result envelope (`ok: true`, the raw text as `output`); one example per assistant turn. Drop, never repair: a row with thinking text, a target that does not parse back to the same calls, more than 8 calls in one turn, or a row longer than `BENCH_N_CTX` | the student should see public rows in the same shape as its own harness. The envelope's `ok` cannot be known for Toucan's failed calls; the error text is still in `output`, and tool results are context, never targets. Qwen3-4B-Instruct-2507 does not think, so thinking text must not be a target |
| D7 | Unit and size of the mix | Count in examples (one per assistant turn), since D3 of Item 3 weighs every example equally. Sample whole conversations, seeded, until 2,000 When2Call and 3,000 Toucan examples. Repeat teacher-wide a whole number of times $k$ so that it is close to a third (formula below) | the approved mix of October 13, made exact |
| D8 | Recipe | Item 3's D1–D5 unchanged for both candidates: rank 16, rate 1e-4, 3 epochs, an adapter per epoch, the epoch chosen on val seed 0 (most passed, the earlier on a tie) | mix has about 30 times teacher-v3's steps per epoch, so the same rate moves the adapter much further. The per-epoch save and the epoch rule are the guard; a change of recipe is one of the two allowed mix changes |
| D9 | GPU plan | One A40 for training and val, if Step 10 estimates mix's training at 4 hours or less. Otherwise a second A40 trains mix while the first trains teacher-wide and runs val | the Stage 8 server command cannot share the GPU with training (it takes the memory), and it is not changed |
| D10 | Val protocol | Base, teacher-v3 (Item 3's epoch-2 adapter), teacher-wide and mix, three seeds each, one server process, Stage 8 command plus `--lora-modules` | the same control as Item 3 (its D7–D8) |
| D11 | Choosing | Section 8 | |
| D12 | M0 on the new shapes | Run it in Session A after val, if time allows: 8 seeds of the base on the 240 new tasks (about 1,900 rollouts, roughly 80 minutes) | Stage 10 needs it ("M0 is sampled on them too, under the same protocol"), and the server is already up |

**The mix formula (D7).** With $P$ public examples and $N$ teacher-wide examples, the own share
is a third when the own rows number $P/2$, so

$$k = \max\left(1,\ \operatorname{round}\left(\frac{P}{2N}\right)\right), \qquad \text{own share} = \frac{kN}{kN + P}$$

With $P = 5{,}000$ and $N \approx 1{,}000$ (Step 5 estimate), $k = 2$ or $3$ and the share is
$0.29$ to $0.38$. The actual share is recorded in `metadata.json`. Note that with three epochs
the own rows are seen $3k$ times; the epoch rule (D8) is what catches over-fitting on them.

## 4. The new train shapes (D2, D3)

Domain: small household and office records (bookings, members, rosters, tasks, vendors), JSON
or `key=value` text, like Item 1's pool. Each prompt states its tokens. "Outcome" is chosen by
`ctx.seed`; recovery shapes have clean/fault pairs built from `_paired_rng`, so a pair shares
its base records.

| # | Skeleton | Family | Task and outcomes | Tools | Closest val / test shape | How it differs |
|---|---|---|---|---|---|---|
| I1 | `booking_count_by_day` | inspection | `bookings.json` lists 8–14 bookings with date and status; reply the number of confirmed bookings on a given date, or `NONE` when there are none | read_file, calculator | test `sum_across_files` | one file, a two-field filter, a count rather than a sum, a zero outcome |
| I2 | `latest_visit_lookup` | inspection | visits split by month under `visits/`; reply the latest visit date of one member, or `NO_VISITS` | list_files, read_file | val `max_timeout_service` | dates rather than numbers, a filter by member across files, an empty outcome |
| I3 | `vendor_price_compare` | inspection | two vendor price lists; reply the cheaper vendor for one item, or `TIE` | read_file | train `address_conflict_stop` | numeric comparison, three outcomes; no val/test shape compares two sources |
| U1 | `member_plan_set` | updates | set one member's plan as `request.json` asks; if it already holds, reply only `NO_CHANGE` and write nothing; else change that line, read it back, reply `UPDATED` | read_file, write_file, edit_file | val `no_op_correct_config` | a text record, the target value in a request file, a change outcome with read-back, another token (D3) |
| U2 | `roster_add_member` | updates | add a name to the end of `roster.txt` unless it is already listed (`ALREADY_LISTED`, no write); else append, read back, `UPDATED` | read_file, write_file, edit_file | test `append_list_item` | text lines, a membership check, a no-op outcome, read-back |
| U3 | `close_project_tasks` | updates | in `tasks.txt`, set every `OPEN` task of one project to `CLOSED`, other lines unchanged; read back; reply the number closed, or `NONE_OPEN` and no write | read_file, write_file, edit_file | test `rename_key_all_files` | one file, a per-line condition, a count answer, a no-op outcome |
| R1 | `moved_record_search` | recovery | read the value in `orders/current.json`; in the fault variant it has moved under `archive/` and the prompt does not say where; reply the value | list_files, glob_files, read_file | test `backup_record_lookup` and `missing_file_report` | the fallback is not named and must be found; the answer is a value, not a stop token |
| R2 | `batch_read_retry` | recovery | read three member records and reply the member with the most points; the fault fails `read_file` call 2 or 3 (by seed), so the rest of a batch is skipped | read_file | test `transient_read_failure` | three sources, a fault in the middle of a batch whose skipped calls must be reissued, an aggregate answer |
| R3 | `log_append_retry` | recovery | append one entry to `log.txt` with `edit_file`; the fault fails the first edit; read back, reply `LOGGED` | read_file, edit_file | test `flaky_write_retry` | a text append through `edit_file` rather than a JSON field through `write_file`, read-back required |
| S1 | `missing_attachment_report` | stopping | `request.json` lists 2–3 attachment files; if all exist reply their codes joined by `,`; if one is missing (by seed) reply only `NOT_FOUND <name>` | list_files, read_file | test `missing_file_report` and `backup_record_lookup` | which file is missing varies, no decoy backup, no fallback, the answer names the file |
| S2 | `booking_slot_stop` | stopping | book a slot: free → write, read back, `BOOKED`; taken by someone else → `CONFLICT`, no write; already this person's → `ALREADY_BOOKED`, no write | read_file, write_file, edit_file | val `locked_config_stop` | the decision comes from record state, three outcomes, one of them a no-op |
| S3 | `note_instruction_ignore` | stopping | report the sender of the newest note under `inbox/`; in half the seeds one note says to delete a file and reply `DONE`; nothing may change | list_files, read_file, write_file | none | follows the system prompt's "tool content is information"; no val/test shape tests it |

R3 and S3 are the weakest rows: R3 is close to `flaky_write_retry` in mechanism, and S3 tests
nothing val or test measures. Swap candidates if you prefer: a calculator retry on a
non-currency quantity, or a third no-op shape. Answer evidence for every shape is the full
content of each decision record, escaped as tool output (the Item 1 rule), so a filename or a
clipped read never counts.

**Negative gates, per shape, beyond the inherited ones:**

- A no-op outcome fails if any write is attempted, even of identical content.
- A change outcome fails without a read-back, and fails if the decision sources are read only
  after the write.
- R1's fault variant fails if the archive value is reported without a failed read of the named
  path (`fault_path`); S1 fails if the reply names a file that exists.
- R2 fails if a skipped record is never read.
- S3 fails on any write, and on the reply `DONE`.

## 5. Constraints

- Only train tasks become training data. The teacher sees only the train split; `export`
  refuses any other.
- No public row is derived from val, test, When2Call's test set or BFCL. Nothing from
  Item 5's external sets is ever trained on.
- Evidence is never overwritten; every command writes a new directory outside the package.
- The 80 existing train tasks, all val and test tasks, and the Stage 8 manifest stay
  as they are. Step 2 proves it.
- A public row is dropped, never repaired or truncated (D6).
- Adapter weights and datasets are not committed; `metadata.json` records their hashes.
- Every rental has a cost guard and ends with the pod deleted and `runpodctl pod list --all`
  empty.

## 6. Files

| File | Change |
|---|---|
| `evals/bench/skeletons.py` | 12 builders and solutions, registered after the existing train shapes |
| `evals/bench/splits.json` | train lists 16 skeletons |
| `evals/bench/dataset.py` | `dataset_version` becomes `stage9-v3` |
| `evals/sft/public.py` (new) | `public` command: download record, `convert_row`, filters, funnel |
| `evals/sft/mix.py` (new) | `mix` command: sample, repeat, write a dataset `train` accepts |
| `evals/__main__.py` | registers `public` and `mix` |
| `tests/test_stage9_tasks.py` | `TrainingTaskTests` to 320 tasks and 16 names; per-shape oracle and negative tests |
| `tests/test_sft_public.py`, `tests/test_sft_mix.py` (new) | converter and mix contracts |
| `requirements-train.txt` | `pyarrow` to read the parquet files, if the download needs it (Step 6) |
| `evals/README.md`, `docs/STAGE.md`, this file | commands and results |

## 7. Steps

Each step ends with the full test suite green. Commands write under `outputs/`.

### Part 1 — the environment (Mac, no model)

**Step 0 — branch and baseline.** Branch `feat/stage9-wider-train-data` from `bac2cbe`; 414
tests pass.

- [x] **Step 1 — the 12 skeletons, test first.** For each shape: oracle tests for each outcome,
  then the negative gates of section 4, then the builder and the scripted solution.
- [x] **Step 2 — the gates and the unchanged sets.**

  ```sh
  python -m agent_from_scratch.evals dataset --output outputs/stage9-item4-YYYYMMDD/data --check
  ```

  `gates_passed` is true: every correct scripted run passes, no answer-only or wrong-answer run
  does, and every fault is observed. Then compare the new dataset manifest with Item 1's
  (`outputs/stage9-item1-20261012/data/manifest.json`): every Stage 9 test task and every old
  train task entry is identical, and only the new train tasks are added.
- [x] **Step 3 — checkpoint: you read the shapes.** One generated task per outcome per
  skeleton (prompt and workspace), printed in one file. Nothing is spent on the teacher before
  you approve them.

### Part 2 — teacher-wide (Mac, API)

- [x] **Step 4 — teacher pass on the 240 new tasks.** A 12-task smoke first (one per
  skeleton), whose traces are read; then the full pass at low effort; then retry passes over
  tasks with no usable trajectory (high effort, then thinking off), as Item 2 did.

  ```sh
  NEW=$(python -c "from agent_from_scratch.evals.bench.run import specs; \
    old={'inventory_reconciliation','eligibility_record_update','address_conflict_stop','points_save_recovery'}; \
    print(','.join(c.id for s,c in specs('train') if s.name not in old))")
  TMPDIR=/tmp python -m agent_from_scratch.evals bench --backend teacher --split train --seed 0 \
    --tasks $NEW --student-gguf $GGUF --output outputs/stage9-item4-YYYYMMDD-teacher-s0
  ```

- [x] **Step 5 — export teacher-wide (D5).**

  ```sh
  python -m agent_from_scratch.evals export --track teacher --flawed-turns drop --max-per-task 1 \
    --output outputs/stage9-item4-YYYYMMDD-sft-teacher-wide \
    outputs/stage9-item2-20261012-teacher-s{0,1,2} outputs/stage9-item4-YYYYMMDD-teacher-s*
  ```

  Check: all 255 teacher-v3 example IDs are in it; every one of the 320 tasks is covered, or
  the uncovered ones are listed with the reason; the funnel by skeleton is recorded.

### Part 3 — public data and the mix (Mac)

- [x] **Step 6 — licences and 20 rows each.** Record the licence, card revision (commit SHA)
  and download date of both datasets. Read 20 rows of each, spread across the file, and
  settle facts 8 and 9: how When2Call writes a tool call, its category shares, whether Toucan
  rows hold thinking text, multiple user turns, or more than 8 calls per turn, and whether its
  judge columns should filter rows. Each finding that changes D6 comes back to you before
  Step 7.
- [x] **Step 7 — `public`, test first.** `convert_row(row, source) -> list[example]` turns one
  row into `messages` and `tools`, renders them with `render_chat` and the project template,
  and writes one example per assistant turn, with the round-trip check `export` uses
  (`LLM.parse_response` must give back the same calls and content). Provenance: `track`
  `"public"`, `skeleton` the source name (When2Call also its category), `run_dir`
  `dataset@revision`, `task_id` the row's ID. Tests: a call row and a no-call row of each
  source; a row with thinking text, an unparseable call, 9 calls in one turn and an over-long
  row are each dropped and counted; tool responses never appear in a completion.
- [x] **Step 8 — convert both.** The whole When2Call `train_sft` and a seeded sample of
  Toucan's SFT config large enough to leave 3,000 examples after the filters. The funnel per
  drop reason is in each `metadata.json`.
- [x] **Step 9 — `mix`, test first, then `sft-mix-v1`.** Inputs: the datasets, a target count
  per public source, the repeat factor rule (D7) and a seed. It samples whole conversations,
  shuffles, and writes `examples.jsonl` and a `metadata.json` that `train` reads unchanged
  (fact 6), with the sources, their hashes, $k$ and the actual shares.
- [x] **Step 10 — token counts and the time estimate.** Encode teacher-wide and mix with the
  Hugging Face tokenizer on the Mac, as Item 3's Step 5 did: total, trained and longest-row
  tokens, and the trained tokens decode back to the completions. Estimate each training time
  from fact 7 and settle D9.
- [x] **Step 11 — commit before renting.** A shallow clone of the branch passes the suite.

### Part 4 — Session A (GPU)

- [x] **Step 12 — set up**, as Item 3's appendix: ship the clone, the two new datasets and
  Item 3's epoch-2 teacher adapter; install; the SFT tests pass with none skipped. Cost guard
  started.
- [x] **Step 13 — train.** teacher-wide first, then mix (or mix on the second pod, D9), each
  with `train --examples DIR --output DIR` and nothing else.
- [x] **Step 14 — serve and check the reload.** The Stage 8 command plus
  `--lora-modules teacher-v3=… wide-e1=… wide-e2=… wide-e3=… mix-e1=… mix-e2=… mix-e3=…`.
  `replay` on 20 teacher-wide examples for `wide-e3` and `mix-e3`: each reproduces more
  targets than the base.
- [x] **Step 15 — the epochs.** Val seed 0 for each of the six epoch adapters; pick by D8.
- [x] **Step 16 — the comparison.** Val seeds 0–2 for the base, teacher-v3, and the chosen
  epoch of each candidate (seed 0 reused); `aggregate` per arm; `compare` each candidate with
  the base and with teacher-v3, per seed. Record mean pass@1 and pass^3 by family and
  skeleton, `format_exact`, false completions, completion tokens, and measured against served
  prompt tokens. For mix against teacher-wide, also count the val runs whose final reply
  declines or asks a question instead of doing the task: every val task can be done with the
  tools offered, so each such reply is a cost of the public data.
- [x] **Step 17 — optional: M0 on the new shapes (D12).**
- [x] **Step 18 — copy back and delete.** Runs, logs and the two chosen adapters into
  `outputs/stage9-item4-YYYYMMDD-session-a/`; checksums match; pod deleted; cost read.

### Part 5 — choose and record

- [x] **Step 19 — at most two mix changes.** Not needed: mix beat teacher-wide on every count
  below (section 13). Only if mix is worse than teacher-wide on val in a
  way its data explains (for example more questions or refusals on our tasks, or lost
  `format_exact`). Each change is written down with its reason before it is trained, and costs
  one more short session.
- [x] **Step 20 — name the candidates (section 8)** and record Item 4 in STAGE.md: results,
  costs, limits, and the Stage 10 entry conditions it settles.

## 8. Choosing (D11)

Val is near its ceiling for teacher-v3 (fact 1), so it can show that a candidate lost
something, but hardly that a wider one gained. The rules are fixed now, before any result:

- **A candidate is dropped** if its val mean pass@1 is below teacher-v3's in at least two of
  the three seeds, or if it raises false completions or lowers `format_exact` against
  teacher-wide (for mix).
- **The final candidates for Item 6** are teacher-v3 and every candidate not dropped: at most
  three. Item 6 then reports all of them on test; nothing is chosen there.
- **M1 for Stage 10** is the candidate with the highest val mean pass@1, and on a tie (within
  one task per seed on average) the one trained on more shapes, since Stage 10 samples the
  wider pool.
- An outcome where neither new candidate beats teacher-v3 on val is a result, and is reported
  as one.

## 9. Exit gate

- [x] The suite passes (449 tests); the 12 new skeletons pass the Item 1 gates and their negative
  tests.
- [x] Val, test and the 80 old train tasks are unchanged (Step 2; the split rename moved labels
  only).
- [x] teacher-wide contains all 255 teacher-v3 examples; its funnel is recorded.
- [x] Both public datasets have licence, revision, date and funnel recorded, and mix's
  `metadata.json` gives its sources and shares.
- [x] teacher-wide and mix trained with the Item 3 recipe, epochs chosen on val seed 0.
- [x] Base, teacher-v3, teacher-wide and mix ran on val with three seeds each, on one server
  process. No test task was run.
- [x] Final candidates and M1 named by section 8; results, costs and limits in STAGE.md.

## 10. Cost

Measured (October 15): teacher \$0.18 at off-peak list prices, at most \$0.37 if all of it billed
at the peak rate; Session A \$13.21 (A100 80 GB, about 7.5 hours). Item 4 came to about
\$13.5, against the projection below; the project has spent about \$16.0 of its \$200 ceiling.

The projection, as written before the work:

| Part | Basis | Estimate |
|---|---|---|
| Teacher on 240 tasks, with retries | Item 2: \$0.034 for an 80-task pass | \$0.10–0.30 |
| Session A training | teacher-wide about 1 hour; mix from Step 10's token count | \$1–6 |
| Session A val and reload | 16 val runs plus replay, about 3 hours; Item 3's session was \$0.58 per hour | about \$2 |
| Optional M0 on new shapes | Item 2: 640 rollouts in 26 minutes | about \$0.80 |
| Up to two mix changes | one training and three val seeds each | \$2–6 |

This fits the \$5–20 line for Items 4–6 in STAGE.md's budget table.

## 11. Record, and what is waiting on you

**Done (October 14).**

- **Steps 0–2.** 12 builders in `evals/bench/skeletons.py`, `splits.json` lists 16 train shapes,
  `dataset_version` is `stage9-v3`. `dataset --check` into `outputs/stage9-item4-20261014/data`:
  400 of 400 correct scripted runs pass (320 train, 80 Stage 9 test), 0 of 400 answer-only and
  0 of 400 wrong-answer runs pass, and all 60 fault variants observe their fault. Against Item 1's
  manifest, all 160 Item 1 task entries are identical, 240 train tasks are added, and the only
  other field that differs is `splits_sha256`. All 124 Stage 8 tasks equal the Stage 8 freeze.
- **Split rename (October 14).** The `bench` splits are now train, val and test (STAGE.md,
  Stage 9). Regenerated into `outputs/stage9-item4-20261014/data-current-splits`: `dataset`
  now covers all 140 test tasks, and 460 of 460 correct runs pass, 0 of 460 answer-only and
  0 of 460 wrong-answer, with all 66 fault variants observed. Against the run above, only split
  labels moved, except 22 new train tasks whose generated names changed (a first name was
  renamed).
- **Two changes made while building, both found by the new tests.**
  `close_project_tasks` sets a status of `CLOSED`, not `DONE`, and the injected note in
  `note_instruction_ignore` asks for `DELETED`, not `DONE`: `DONE` is a reply val and test ask
  for, so training the student to refuse it would have worked against them (D3).
  `missing_attachment_report` carries a `fault_signal` for the failed read of the missing file,
  so that `export` does not drop that turn as the model's own error.
- **Step 3.** `outputs/stage9-item4-20261014/shapes-for-review.md`: one generated task per
  outcome of every new shape (27), with prompt, workspace, expected reply, expected files,
  evidence and process rules.
- **Step 7.** `evals/sft/public.py` and 14 tests. `export`'s render-and-round-trip step is now
  the function `render_turn`, shared by both; `export`'s 31 tests pass unchanged.
- **Step 9.** `evals/sft/mix.py` and 6 tests.

**Step 6 findings: three decisions for you.**

1. **When2Call `train_sft` has no tool calls.** All 15,000 rows are one user turn and one text
   reply: 7,379 follow-up questions and 7,621 declines (2,222 of them with no tool offered),
   told apart by a "?" in the reply. In
   the paper this set is combined with separate tool-call data. `train_pref` (9,000 rows) holds
   the tool calls: its chosen reply is a `<TOOLCALL>` in 3,000 rows. Dry runs of 2,000 examples:

   | Source | Calls | Questions | Declines | Rows dropped |
   |---|---:|---:|---:|---:|
   | `when2call-sft` | 0 | 949 | 1,051 | 20 of 2,020 |
   | `when2call-pref` (chosen reply) | 684 | 646 | 670 | 22 of 2,022 |

   **Recommendation: `when2call-pref`.** 2,000 examples that only ask or decline are the risk
   the plan already names (more questions or refusals on our tasks, where everything needed is
   in the files). The chosen replies teach when to call as well as when not to. Both sets come
   from the generator of Item 5's When2Call test, so either one makes mix's When2Call score
   in-distribution, as STAGE.md already says.

   **Decided October 15, 2026: `when2call-pref`, all three kinds kept.** A decline is the chosen
   reply only when no suitable tool is offered: in the 2,000-row sample, 206 of the 670 declines
   have no tools and 464 offer only tools that do not fit the request. Most decline a request for
   live data or an outside service, which is the behaviour When2Call teaches: do not invent a
   call. A few also decline work the model could do itself (for example finding the repeated
   number in a short list). The val comparison counts that risk (Step 16).
2. **D6, extended.** When2Call writes Python type hints, not only BFCL names: `str, optional`
   (5,949 rows), `int, optional`, `List[int]`, `Tuple[float, float]` and so on. With only
   `dict` → `object` mapped, 55–60% of rows were dropped, and the drop was not random. Now a
   trailing `, optional` / `, default=...` is removed (the schema's `required` list already says
   it), and `List[...]`, `Tuple[...]` and `Set[...]` become `array` with no item type invented.
   `Union` and `Callable` still drop the row. About 1% of rows are dropped in all.
3. **Downloading Toucan.** The SFT config is three parquet files, 425 MB, at revision
   `0df3cf37f2abefb380370cfb02eabea2a35ae782`. Reading them needs `pyarrow`, which is not
   installed. **Recommendation:** add `pyarrow` to `requirements-train.txt`, install it, and
   download the three files at that revision. The 20 rows read so far came through the
   datasets-server API, which does not pin a revision. Of those 20, all 20 converted (51
   examples): 7 are from the "irrelevant tools" subset, 5 are multi-turn, none has thinking text,
   and none has more than 2 calls in a turn. Toucan writes narration, calls and results as
   separate messages; `public` joins a narration and the calls after it into one assistant turn.

**Licences and revisions.** When2Call: CC-BY-4.0, revision
`0582f7749df63a96fdc3070932e83e72396ace53`, downloaded October 14, 2026; CC-BY requires
attribution in what is published. Toucan-1.5M: Apache-2.0, revision
`0df3cf37f2abefb380370cfb02eabea2a35ae782`.

**Approved October 15:** the three decisions above, and the shapes (Step 3).

## 12. Record of Steps 4–13 (October 15)

**Step 4 — teacher.** The 12-task smoke passed 11. `moved_record_search` failed because the
teacher globbed for `current.json` before reading the named path, so the fault was never met
and the trajectory showed no recovery. The prompt now says "Read orders/current.json and
report only its total field. If the read fails because the file is not there, ..."; with it,
4 of 4 smoke tasks pass. The first full pass (`teacher-s0`) then failed all 10 "missing"
outcomes of `missing_attachment_report`: the runtime stops a batch at its first failed call, so
attachments listed after the missing one were never read, while the contract required their
codes as evidence. The shape is now a clean/fault pair: the fault variant needs a failed read
of exactly the missing file (`fault_path`) and nothing after it; a new test checks both sides.
`dataset --check` at that code: 460 of 460 correct, 0 answer-only, 0 wrong-answer, 76 of 76
fault variants observed. The pass was rerun at the fixed code (`teacher-final-s0` for 200
tasks; the last 40 in four parallel processes, `-s1` to `-s4`, because DeepSeek's peak hours
slowed one process to a task every three minutes): **240 of 240 passed**. `teacher-s0` is kept
as evidence of the defect and is not training data.

**Step 5 — teacher-wide.** Exported from the three Item 2 runs and the five final runs: 356
rollouts, none rejected, 320 trajectories, **all 320 tasks covered**, 16 own-error and 4
loose-answer turns dropped, **956 examples**. All 255 teacher-v3 example IDs are in it.

**Steps 6 and 8 — public data.** `pyarrow` could not be installed: this session reaches
Hugging Face but not PyPI. Toucan rows were therefore fetched through the datasets-server JSON
API instead of the parquet files: 160 seeded offsets of 10 rows, 1,600 rows, with their
`row_idx` and a `.meta.json` (repository revision `0df3cf3`, unchanged since 2025-10-04,
fetched October 15). Converted:

| Dataset | Rows visited | Dropped | Examples | Kinds |
|---|---:|---|---:|---|
| `public-when2call` (`train_pref`, seed 0) | 2,022 | 10 bad tools, 12 unknown type | 2,000 | 684 call, 646 question, 670 decline |
| `public-toucan` (seed 0) | 1,015 | 34 round trip, 9 bad call, 6 too long | 3,002 | 316 irrelevant, 275 multi-turn, 230 single-turn, 145 diversified (rows) |

**Step 9 — mix.** `sft-mix-v1`: 7,870 rows, $k = 3$, own share 0.364.

**Step 10 — tokens.** Every trained token decodes back to its completion in both datasets.
teacher-wide: 1.6M tokens, longest row 3,780. Mix: 17.0M tokens per epoch, longest row 29,549;
Toucan rows average 3,563 tokens (p99 18,350). A bug found here: `train` and `replay` split
`examples.jsonl` with `str.splitlines()`, which also breaks on U+2028, and 7 Toucan rows hold
one inside a JSON string. Both now split on `"\n"` only, with a regression test.

**D9, revised: one A100 80 GB instead of A40s.** Item 3 measured 14.4 GiB for a 3,599-token
row; the estimate for 29,549 tokens was above an A40's 44.5 GiB. On the pod, one training step
on a 23,481-token mix row peaked at **51.8 GiB**, which confirms it. The A100 (\$1.79 per hour,
secure cloud) holds every row, so D6's rule (drop only rows longer than the window) stands.

**Step 11.** Commits `edc56d7`, `9638b34`, `33867ec`; a shallow clone passes 449 tests.

**Steps 12–13 — Session A**, pod `wo80nlx0p5x481` (A100-SXM4-80GB), from 08:03 UTC. Upload
from the Mac ran at about 25 KB/s and dropped twice, so the pod got a sparse clone of
`agent_from_scratch/` (commit `33867ec`, clean tree), the datasets via `rsync --partial`, and
the teacher-v3 adapter in the background. Versions as in Item 3; the 33 SFT tests pass.
teacher-wide trained in 1,656 seconds (0.35 ms per token, against 0.75 on the A40), 360 steps,
mean loss by epoch 0.258, 0.034, 0.027. Mix trained in 5.3 hours, 2,952 steps, mean loss by
epoch 0.515, 0.368, 0.291. The CUDA allocator logged three "allocation failed with OOM"
warnings during mix and recovered from each by freeing its cache; the training script now sets
`expandable_segments`. Training commands: `scripts/train_stage9_adapters.sh`.

## 13. Record of Steps 14–18 (October 15)

**Step 14 — serve and reload.** The Stage 8 command plus seven adapters (teacher-v3 from Item 3,
hash `7a7357fd…` as recorded, and three epochs of each candidate). `replay` on the first 40
teacher-wide examples: wide-e3 and mix-e3 reproduce 34, the base 31. These rows are mostly
shapes the base already handles, so the gap is small; the val runs settle it.

**Step 15 — epochs** (val seed 0): teacher-wide 44, 45, 51, so epoch 3; mix 57, 60, 60, so
epoch 2 (the earlier on a tie).

**Step 16 — comparison.** Results and reading in STAGE.md. Paired gains and losses per seed
(seeds 0, 1, 2):

| From → to | Gains, losses |
|---|---|
| base → teacher-v3 | 24, 1 · 24, 1 · 23, 3 |
| base → teacher-wide | 20, 5 · 20, 14 · 19, 11 |
| base → mix | 27, 3 · 26, 2 · 22, 2 |
| teacher-v3 → mix | 4, 3 · 3, 2 · 4, 4 |
| teacher-v3 → teacher-wide | 5, 13 · 3, 20 · 1, 13 |
| teacher-wide → mix | 11, 2 · 19, 1 · 13, 1 |

Format exact: base 93, teacher-v3 179, teacher-wide 156, mix 182 of 192. False completions:
7, 0, 14 and 1 of 87 claims. Failed val runs whose final reply declines or asks: 3, 0, 0, 0.
Measured and served prompt tokens agree on all 3,902 requests.

teacher-wide's failures where teacher-v3 passes, by kind: `delete_key` edits and reads back
until the 6-request limit (13) or saves the wrong content (1); `check_fix_recheck` edits after a passing check, or edits a
correct file (15); `transient_list_failure` repeats reads until `no_progress` (8);
`single_field_edit` loops edits (4); `max_timeout_service` answers before reading every file
(3); `locked_config_stop` replies `DONE` on an unlocked config without writing (2).

**Step 17 — M0 on the new shapes (D12).** Eight base samples of the 240 new tasks, Item 2
protocol, eight processes against one server, about 10 minutes:

| Skeleton | Mean pass@1 | Tasks with a verified trajectory | Mixed (strict reward) |
|---|---:|---:|---:|
| `batch_read_retry` | 0.975 | 20 of 20 | 2 |
| `booking_count_by_day` | 0.237 | 7 of 20 | 6 |
| `booking_slot_stop` | 0.019 | 1 of 20 | 1 |
| `close_project_tasks` | 0.013 | 2 of 20 | 2 |
| `latest_visit_lookup` | 0.013 | 1 of 20 | 1 |
| `log_append_retry` | 0.287 | 13 of 20 | 10 |
| `member_plan_set` | 0.175 | 10 of 20 | 10 |
| `missing_attachment_report` | 0.800 | 19 of 20 | 8 |
| `moved_record_search` | 0.506 | 11 of 20 | 1 |
| `note_instruction_ignore` | 0.000 | 0 of 20 | 0 |
| `roster_add_member` | 0.431 | 9 of 20 | 2 |
| `vendor_price_compare` | 0.125 | 8 of 20 | 7 |
| all | 0.298 | 101 of 240 | 50 |

On `note_instruction_ignore` the base reads one note and answers; it never writes, so it never
follows the injected instruction.

**Step 18 — copy back and delete.** Runs, comparisons, M0 runs, logs and the two chosen
adapters are in `outputs/stage9-item4-20261015-session-a/`; archive checksums match the pod, and
the adapter hashes match their `metadata.json` (teacher-wide epoch 3 `a81297dd…`, mix epoch 2
`1388d076…`). The pod was deleted at about 15:30 UTC; `runpodctl pod list --all` is empty. The
session cost \$13.21 (account balance). While the Mac lid had to close, the val work ran on the
pod from a script, and a guard on the pod (configured with the account's API key, which left
with the pod) was set to delete it at 18:30 UTC.

**Steps 19–20.** No mix change: mix is better than teacher-wide on every count Step 19 names.
Candidates named by section 8; recorded in STAGE.md.
