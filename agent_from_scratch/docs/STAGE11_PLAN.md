# Stage 11 — final comparison and delivery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user reviews
> each task's diff before it runs). Steps use checkbox (`- [ ]`) syntax for tracking.

Written October 22, 2026 on branch `feat/stage10-ei-grpo` at `4dd35ed`, after Stage 10. **For the
user's review; nothing here runs before approval.**

**Goal:** Run every final candidate once on the frozen test, the hard test and the external test;
report the results with paired comparisons, uncertainty, costs and limitations in
`docs/results.md`; let the agent run on any OpenAI-compatible model; and deliver a reproducible
setup and a walkthrough notebook that follows the stages of the build.

**Architecture:** No new training. Three hard-test shapes are added and calibrated on M1 like
Stage 10's, then a new freeze covers test and the hard test. One serving session runs the new
candidates on test, all arms on the hard test, and the new candidates on the external test. A
small results module computes pass@1, paired wins and losses, McNemar p and bootstrap intervals
from the run directories, and writes the tables.

**Tech Stack:** as Stages 9–10 (vLLM 0.30.0 serving only; one A40 is enough), DeepSeek for the
When2Call judge, `unittest`.

**Spec:** [STAGE.md](STAGE.md), "Stage 11 — final comparison and delivery";
[STAGE9_ITEM5_6_PLAN.md](STAGE9_ITEM5_6_PLAN.md) (the test and external protocol);
[STAGE10_HARD_TASKS_PLAN.md](STAGE10_HARD_TASKS_PLAN.md) (the hard test and its shortfall).

## 1. Starting point

| Arm | What it is | Test (140) | External | Adapter |
|---|---|---|---|---|
| base | Qwen3-4B-Instruct-2507 | 0.419 (Stage 9) | Stage 9 | — |
| teacher-v3 | SFT on 255 teacher examples | 0.724 (Stage 9) | Stage 9 | `stage9-item3…/sft-teacher-r1/epoch-2` (`7a7357fd…`) |
| mix | Stage 9's M1 | 0.774 (Stage 9) | Stage 9 | `stage9-item4…/sft-mix-r1/epoch-2` (`1388d076…`) |
| mix-v2 | Item 7; deferred to Stage 11 by Item 7's rule | — | — | `stage9-item7…/sft-mix-v2-r1/epoch-3` |
| mix-v3 | Stage 10's M1 (mix + act/ask demonstrations) | — | — | `stage10-20261020-session-e2/…/sft-mix-v3-r1/epoch-2` (`f16d8ece…`) |
| **M4** | Stage 10's best: three self-SFT rounds from mix-v3 | — | — | `stage10-20261021-session-fg/s10g-round3/outputs/s10-m4` |

Val (144): mix-v3 0.752, M4 0.852 (Stage 10, one server). The hard test holds only
`support_dialogue` (20 tasks): C2 needed two shapes.

## 2. Decisions (for review)

| # | Decision | Why |
|---|---|---|
| F1 | **New candidates on test and the external test: mix-v2, mix-v3, M4.** Stage 9's numbers for base, teacher-v3 and mix stand as reported and are not run on test again. The GRPO adapters are not candidates (they lost to M1 on val) | STAGE.md: run once, after val selection; Stage 9's numbers stand. Item 7 deferred mix-v2 here |
| F2 | **The hard test is completed before anyone runs it:** three new hardtest shapes, one per skill family Stage 10 trained (an ordered text-config patch, a first-match fee table, a ledger audit with a mid-scan read failure), in domains and formats no train or val shape uses. Calibrated on M1 (mix-v3) with Stage 10's C1 on calibration seeds; C2 for the hard test: at least two of the four hardtest shapes kept | the hard test is where Stage 10's gains could show; one all-or-nothing shape cannot carry a result |
| F3 | **All six arms on the hard test** (base, teacher-v3, mix, mix-v2, mix-v3, M4), three seeds | nobody has run it, so no number exists to "stand"; every arm gets its first and only run |
| F4 | **One new freeze** written on the pod from a clean clone of the final commit (`dataset --check`, as Stage 9's E5), covering test and the hard test. The 140 test tasks must match Stage 9's freeze fingerprints, or the session stops | Stage 9's test protocol; the harness changed (new splits), so a new freeze, with the old tasks proven unchanged |
| F5 | **The external test as Stage 9 ran it** (When2Call MCQ, When2Call judge with DeepSeek on the Mac, BFCL multi-turn base), on the three new candidates. mix-v2, mix-v3 and M4 carry the label "after a BFCL-motivated change" | Stage 9's E1–E3; the label rule of Item 7 |
| F6 | **Paired comparisons:** M4 against mix-v3 (Stage 10's question) and mix-v3 against mix-v2, inside the session; each new candidate against Stage 9's recorded mix runs, flagged as across sessions. Each with wins and losses by task and seed, McNemar's exact p (the existing `compare`), and a 95% bootstrap interval of the pass@1 difference over tasks (10,000 resamples, seed 0) | STAGE.md asks for paired results and uncertainty; three seeds alone understate it |
| F7 | **A walkthrough notebook replaces the demo script** (decided with the user, October 22): `examples/walkthrough.ipynb`, one section per stage of the build, in the order a tutorial would take them — one turn with one tool; the agent loop over several turns; context management; evaluation from scratch; post-training data (SFT examples and the loss mask); expert iteration and GRPO (the strict reward and group advantages on recorded rollouts, and Stage 10's results). Sections 1–3 run live; 4–6 need no model or use recorded runs, since training needs a GPU. Durable memory is not shown (Stage 4 is deferred) | the repo is to serve both daily use and a from-scratch tutorial; a notebook is runnable and readable, and becomes the tutorial's backbone. The README and the tutorial's final form are decided later |
| F8 | **Any model, not only the local Qwen:** an OpenAI-compatible chat backend (`/v1/chat/completions`, native tool calls), set by `AGENT_BASE_URL`, `AGENT_MODEL` and `AGENT_API_KEY`; the REPL, the Telegram bot and the notebook use it when set and the local GGUF otherwise. It covers OpenAI, DeepSeek, OpenRouter, and local servers (Ollama, LM Studio, llama.cpp's server, vLLM). The benchmark protocol keeps its frozen backends | users should be able to run the agent 24/7 on the model they choose; the notebook should run in minutes without a multi-GB download |
| F9 | **Delivery:** `docs/results.md` (every table, costs, limitations; no claim beyond the evidence) and a "Reproduce" section in `evals/README.md` (setup, fixture reset and scripted gates, adapter reload, one test run) | STAGE.md: package setup, fixture reset/eval, adapter reload |

## Global Constraints

- No training; no new train, val or test task; the 664 Stage 8–9 tasks and Stage 10's 140 keep their fingerprints.
- Test and the hard test run only with `--final --manifest`, three seeds (0, 1, 2), one server process, each arm once.
- The hard test's levels are chosen on calibration seeds before any candidate runs on it.
- Stage 9's numbers are not replaced; new numbers sit beside them.
- Pod sessions: the cost guard on the Mac; the Mac kept awake with disk space free; setup with `OMP_NUM_THREADS=8`.
- Each task's diff is shown to the user before it runs; commits use `--author="claude <claude@noreply>"`.

## Review Focus

- A test-task fingerprint differing from Stage 9's freeze must stop the session before any test run (Task 3).
- The bootstrap interval must resample tasks (all seeds of a task together), not runs (Task 2 test).
- An arm missing a seed must be reported as missing, never averaged over fewer seeds silently (Task 2 test).
- The hard test's calibration must use only calibration seeds; a final hardtest seed in the calibration runs is an error (Task 1, `test_levels`).
- The When2Call judge must retry a dropped connection (it does since Stage 9) and record the judge model (Task 3).

---

## 3. Tasks — code (local, no GPU)

### Task 1: Three hard-test shapes

**Files:** modify `evals/bench/shapes_rules.py` (`config_patch_plan`, `shipping_fee_table`) and `evals/bench/shapes_states.py` (`ledger_audit_fault`), `evals/bench/splits.json` (`"shapes".hardtest`); tests in `tests/test_shapes_rules.py` and `tests/test_shapes_states.py`.

| Shape | Task | Level 1 → 5 | Closest train shape · how it differs |
|---|---|---|---|
| `config_patch_plan` | apply the N ordered steps of `patch.txt` to `service.ini` (sections, `key = value`); reply `PATCHED` | N = 2, 4, 6, 9, 12 | `ordered_migration` · an INI text file with sections, not JSON |
| `shipping_fee_table` | first-match fee rules (weight, zone, express), fee = base + rate × kg rounded down; write `fees.txt`; reply `PRICED` | (rules, orders) = (2,3) … (8,14) | `rule_table_apply` · a computed fee, not a looked-up value |
| `ledger_audit_fault` (clean/fault) | compare `bank/` statements with `ledger.csv`, write `audit.txt` with every mismatch; the fault variant fails one read mid-scan | K = 3, 5, 7, 9, 12 | `fault_amid_reconcile` · amounts with cents, two sources per line |

- [x] Tests (as Stage 10's): every level keeps the gates; difficulty grows; a skipped step, an off-by-one fee and an unretried read fail.
- [x] Run; the full suite passes; the 804 tasks of Stage 10's manifest keep their fingerprints.
- [x] Commit `feat: three hard-test shapes`.

### Task 2: Results module

**Files:** create `evals/report.py` (`python -m agent_from_scratch.evals report`; not `results.py`, which `.gitignore`'s `*result*` would hide); test `tests/test_report.py`.

**Interfaces — produces:** `pass_at_1(runs: list[list[dict]]) -> float`; `paired(a: list[list[dict]], b: list[list[dict]]) -> dict` (wins, losses, ties by task and seed; McNemar exact p over the discordant pairs); `bootstrap_diff(a, b, *, resamples=10000, seed=0) -> tuple[float, float]` (resamples task ids, keeping every seed of a task); `table(arms: dict[str, list[Path]], subsets: dict[str, set[str]]) -> str` (markdown). The command takes `--arm NAME=RUN,RUN,RUN` repeatedly and `--subset NAME=SKELETON,…`.

- [x] Tests: worked examples for each function; `test_bootstrap_resamples_tasks_not_runs`; `test_a_missing_seed_is_reported`; `test_mcnemar_matches_compare` (agrees with `compare` on the same runs).
- [x] Run; commit `feat: a results module for Stage 11`.

### Task 3: Session scripts

**Files:** create `scripts/stage11_calibrate.sh` (the hardtest shapes only: `bench --split calib --tasks` of those shapes, 8 seeds, then `levels`); `scripts/stage11_session.sh`:

1. `dataset --check` on a clean clone writes the freeze; a python check compares its 140 test tasks with Stage 9's freeze (shipped with the bundle) and stops on any difference.
2. One server: base, teacher-v3, mix, mix-v2, mix-v3, M4.
3. Test: mix-v2, mix-v3, M4 × seeds 0–2, `--final --manifest`.
4. Hard test: all six arms × seeds 0–2, `--final --manifest`.
5. External: `when2call mcq` and `when2call generate` and `bfcl`, for mix-v2, mix-v3, M4.
6. Each step packed for the copy-back.

- [x] `bash -n`; dry runs (`STAGE10_DRY=1`) list the expected runs (9 test, 18 hard test, 9 external).
- [x] Commit `feat: Stage 11 drivers`.
- [x] Follow-up (`4df8351`): `stage11_calibrate.sh` prints the hard test's C2 with the hardtest
  shapes `levels.json` already keeps (the `levels` report sees only the three new shapes, so its
  own C2 always fails there); `stage11_session.sh` refuses to freeze an uncommitted tree. Checked
  on the Mac: today's code reproduces Stage 9's 140 test fingerprints
  (`outputs/stage9-item6-20261016-session/outputs/stage9-item6-20261016-data/manifest.json`, the
  file to pass as `STAGE9_MANIFEST`).

### Task 4: An OpenAI-compatible chat backend

**Files:** modify `llm.py` (`ChatAPIClient`), `config.py` (the three variables), the REPL entry in `agent.py` and `bots/telegram_bot.py` (pick the backend); test `tests/test_chat_api_client.py` (a fake HTTP transport, no network).

**Interfaces — produces:** `ChatAPIClient(base_url, model, api_key, *, temperature, max_tokens)` with `LLM`'s interface (`generate`, `measure_context`, `settings`, `read_usage`). Tools are sent as OpenAI tool schemas and replies parsed from `tool_calls`; the context budget is estimated from characters (`count_method: "estimate"`) where the API has no token counter; `make_model() -> LLM | ChatAPIClient` chooses by whether `AGENT_BASE_URL` is set.

- [x] Tests: a tool-call reply becomes an `LLMResponse` with the same calls; a plain reply; an HTTP error is retried, then raised as the existing backend error; `make_model` picks each backend; the agent loop runs a two-tool task end to end on the fake transport.
- [x] Run against one real endpoint (DeepSeek, whose key Stage 9 already uses); commit `feat: any OpenAI-compatible model as the agent's backend`.
- [x] Beyond the plan: the fit rule (`context.context_blocker`, `window_share_ratio`) accepted
  only `"exact"` and would have blocked every request, so it now also accepts `"estimate"`. The
  estimate is characters × a token rate that starts at 1/3 and rises to the highest rate the API
  reports. A fourth, optional variable, `AGENT_N_CTX`, gives the window (default 32,768), since a
  local server's is often smaller. Neither changes the vLLM benchmark path.

### Task 5: The walkthrough notebook

**Files:** create `examples/walkthrough.ipynb` (outputs saved, so it reads on GitHub without running).

- [x] Six sections in stage order (F7), each a short explanation, the few calls that show the idea, and its output: (1) one turn, one tool; (2) the agent loop and a follow-up turn; (3) context management: a long session compacted, what was kept, the session resumed; (4) evaluation: a generated task, the scripted correct run passing and two cheats failing, the verifier's checks; (5) post-training data: one verified run exported as SFT examples, the loss mask shown on the tokens; (6) expert iteration and GRPO: the strict reward and advantages on recorded rollouts, and Stage 10's results table.
- [x] The first cell picks the model (F8): `AGENT_BASE_URL` set, or the local GGUF.
- [x] Run it top to bottom on the Mac with an API model; commit it with its outputs.

## 4. Steps — GPU sessions

- [ ] **Session H — hard-test calibration** (A40, about 30 minutes, about \$0.30): M1 on the three new shapes × 5 levels × 6 calibration tasks × 8 samples; C1 per shape; C2 for the hard test (at least 2 of 4). Commit `levels.json`; stop and review if C2 fails.
- [ ] **Session I — final runs** (A40, about 5 hours, about \$3): Task 3's session. Copy back, delete the pod. Judge the When2Call replies on the Mac (DeepSeek, about \$1).
- [ ] **Results:** `docs/results.md` from the results module; STAGE.md's Stage 11 marked done.

## 5. Cost

About \$5 in all (A40 at \$0.59 per hour, plus the judge), and a few cents of API calls for
Task 4's check and the notebook; within the \$200 ceiling.
