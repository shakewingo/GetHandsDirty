# Stage 9 Items 5 and 6 — the external test, and the final test once

Written October 16, 2026 on branch `feat/stage9-wider-train-data` at `fd4fb1e`, and executed the
same day at the user's request, without a review pause. Spec: [STAGE.md](STAGE.md), Stage 9
("Item 5" and "Item 6"). The code was delegated to AI at the user's explicit request.

**Status: done (October 16, 2026).** Results and reading in [STAGE.md](STAGE.md), Stage 9 ("Item 5
complete", "Item 6 complete", "Stage 9 complete"); the session record is section 4 below.

**Goal.** Build the two external scorers, check them on the base model (Item 5), then freeze the
tree once and run the base and the final candidates, teacher-v3 and mix, on test and on the
external test (Item 6). Nothing is chosen on these numbers.

## 1. Facts

1. **When2Call's MCQ scorer** (repository `NVIDIA/When2Call` at `ecc8d42`,
   `evaluation/mcq/lm_eval_harness`) is lm-eval-harness's `multiple_choice` task. Each of the
   four candidate replies (direct, tool call, request for info, cannot answer) is scored by its
   log-likelihood after the prompt. Metrics: `acc` (argmax of the summed log-likelihood),
   `acc_norm` (each divided by the reply's length in characters), macro F1 over the predicted
   kinds, and a hallucination rate: on rows with no tools whose answer is "cannot answer", the
   share where the tool call wins. The Qwen variant (`process_docs_qwen2_5`) writes the Qwen
   tool system prompt by hand and offers the tool call as bare JSON. 3,652 rows: 1,295 tool
   call, 1,062 request for info, 1,295 cannot answer; 258 offer no tools.
2. **When2Call's LLM-as-a-judge** (300 rows, 100 per kind): the model answers each question
   with the tools offered and no system prompt (parameter types forced to string), and a judge
   model classifies the reply into the four kinds with a fixed prompt. The score is the share
   classified as the correct kind, per kind.
3. **BFCL multi-turn** (`ShishirPatil/gorilla` at `6ea5797`, `bfcl_eval`):
   `BFCL_v4_multi_turn_base` has 200 entries of 1–7 user turns over eight simulated backends
   (file system, math, messaging, posting, tickets, trading, travel, vehicle). The checker
   executes the model's calls and the ground-truth calls on fresh backend instances after each
   turn and requires the same backend state and every ground-truth result among the model's
   results. Calls are strings such as `cd(folder='document')`, run with `eval`.
4. **Uploads are slow** (about 25 KB per second from the Mac) and downloads fast. The pod
   fetches both public repositories itself, at the pinned commits.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| E1 | When2Call MCQ exactly as the Qwen variant writes it (prompt, choices, metrics), log-likelihoods from the Stage 8 vLLM server with the frozen command | "Score it the way the dataset's own evaluation does" |
| E2 | When2Call judge: the model's reply rendered with our template (which writes Qwen's default system line when none is given), the tools converted as their inference script does, greedy decoding; judged by DeepSeek (`deepseek-flash`, thinking off) with their judge prompt, on the Mac | the original judge model is not named in the repository; the judge is recorded |
| E3 | BFCL: only `multi_turn_base`, all 200 entries, labelled a local adaptation. The model runs in a plain chat loop with the entry's tools (not our harness), greedy, at most 20 steps per turn; its calls are built from the parsed JSON arguments with `repr` and run on BFCL's own backends; BFCL's own `multi_turn_checker` scores | external means outside our harness; base is BFCL's core category |
| E4 | The external test runs on the same arms as test: base, teacher-v3, mix | Item 6 |
| E5 | Test: `--split test --final --manifest`, three seeds per arm, one server process; the manifest is written on the pod from a clean clone of the final commit by `dataset --check` | Item 6 as written |
| E6 | One A40 (serving needs no 80 GB). The two adapters (about 250 MB) upload while Item 5's base-only checks run on a base-only server; the server is then restarted with both adapters, and every Item 6 run uses that one process | uploads take hours; Item 6 asks for one process |

## 3. Steps

- [x] **1.** `evals/external/when2call.py` and `evals/external/bfcl.py`, the commands `when2call`
  and `bfcl`, with tests that need no model.
- [x] **2.** Commit; a shallow clone passes the suite.
- [x] **3.** Rent; start the adapter upload; set up; fetch the two repositories at the pinned
  commits.
- [x] **4.** Item 5: base-only server; run each scorer on the base; read the outputs.
- [x] **5.** Item 6: `dataset --check` writes the manifest; restart the server with both
  adapters; test with three seeds per arm; the external test per arm.
- [x] **6.** Copy back, delete the pod, judge the When2Call replies on the Mac, record in
  STAGE.md.

## 4. Record (October 16)

**Step 1–2.** `evals/external/when2call.py` and `bfcl.py`, 13 tests; commit `e2e9fc8`; a clean
sparse clone passes 462 tests.

**Step 3.** Pod `ohqyh51by0em9q`, one A40 in EU-SE-1, \$0.59 per hour, from about 05:20 to 08:10
UTC: \$1.69. The pod fetched When2Call's test files (dataset revision `0582f77`) and BFCL (commit
`6ea5797`) itself. The installer resolved transformers 5.20.0; it was pinned back to 5.19.0, the
version Items 3–4 trained and served with. The two adapters (245 MB) uploaded in about ten
minutes this time, and their hashes match their training records (teacher-v3 `7a7357fd…`, mix
`1388d076…`).

**Step 4 — Item 5.** BFCL's ground-truth calls, fed through our tool conversion and its checker,
pass 200 of 200 (17 to 39 tools per entry). On a base-only server: When2Call MCQ accuracy 0.456,
acc_norm 0.500, macro F1 0.300, hallucination rate 0.225; the base answered 205 of the 300
judge questions with a call; BFCL 0.37. A bug cost twenty minutes: `item5.sh` ended with a bare
`wait`, which also waits for the background vLLM server and so never returns. `item6.sh` waits
for its own jobs only.

**Step 5 — Item 6.** `dataset --check` at `e2e9fc8`: 460 of 460, gates passed. One server
process with `teacher-v3` and `mix`. Test: 9 runs of 140 tasks, `--final --manifest`, no drift,
no protocol mismatch; measured and served prompt tokens agree on all 4,291 requests. External
test on the same three arms.

**Step 6.** Results copied back and checked (`item6.tgz`, sha256 `5d3ef4d4…`) by a job on the
Mac, which then deleted the pod; a guard on the pod would have removed it 30 minutes after the
work ended. The When2Call answers were judged on the Mac with DeepSeek; one run failed on a
dropped connection, so the judge now retries an API call up to five times.

**Diagnoses** (detail in STAGE.md): mix's BFCL drop is follow-up questions (136 of its 147
turns without a call) about information the conversation already holds, and the drop of both
adapters on `append_list_item` is a targeted `edit_file` habit that fails on multi-line JSON.
