# Stage 10, part 2: harder tasks, so both arms have signal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user reviews
> each task's diff before it runs). Steps use checkbox (`- [ ]`) syntax for tracking.

Written October 21, 2026 on branch `feat/stage10-ei-grpo` at `8c7544e`, after Session E. **For
the user's review; nothing here runs before approval.**

**Goal:** A train pool and a val set on which M1 (mix-v3) succeeds about half the time, chosen by
rules fixed before any of it is sampled, then the two Stage 10 arms (self-SFT and the GRPO
pilot) run on that pool exactly as [STAGE10_PLAN.md](STAGE10_PLAN.md) designed them.

**Architecture:** New skeletons whose builder takes a difficulty `level` (1–5). A calibration
split samples M1 on every (shape, level) at seeds no other split uses; a fixed rule picks one
level per shape, recorded in `evals/bench/levels.json`; the train, val and hard-test shapes are
then registered at that level with fresh seeds. The arms reuse Stage 10's code, with the pool
restricted to the new train shapes.

**Tech Stack:** as Stage 10 (Python stdlib, `unittest`, vLLM 0.30.0 on one A100 80 GB).

**Spec:** [STAGE.md](STAGE.md) Stage 10, [STAGE10_PLAN.md](STAGE10_PLAN.md) sections 2 and 6–7.

## 1. Why

Session E (STAGE10_PLAN.md, section 7) made mix-v3 M1: val pass@1 0.929, and 0.996 on the 420
train tasks with every task solved. Self-improvement needs tasks the model solves *sometimes*:
GRPO learns only from tasks with mixed outcomes (9 of 420), and self-SFT can only raise a yield
that is below its ceiling. And a gain must be measurable: val is at 97 of 104, where seed noise
(±1–3 tasks) is as large as any likely effect.

What mix-v3 still fails on val points at the skills to stretch: the multi-turn act/ask decision
in an unfamiliar domain (`followup_move_meeting`, 9 of 60), leaving a correct state untouched
(`no_op_correct_config`, 6 of 30), and stopping on a lock (`locked_config_stop`, 3 of 18).

## 2. Decisions (for review)

| # | Decision | Why |
|---|---|---|
| K1 | **Difficulty is a level 1–5 of each new shape**, scaling what a 4B model finds hard: more records, longer rule chains, more turns, more mixed decisions per run. Each level keeps the shape's scripted solution and the Item 1 gates (correct passes; answer-only and wrong-answer fail) | one knob per shape makes "about half" reachable without designing a shape per difficulty |
| K2 | **Six train shapes, three val shapes, three hard-test shapes** (section 3). No new shape mirrors a val or test shape; each records its closest existing shape and how it differs (the Item 4 rule) | train and val must test the same skills with different mechanics, or val stops being held out |
| K3 | **Calibration on M1, on its own seeds** (100–105, six tasks per level, three act/ask pairs where a shape has them), 8 samples each, strict reward. Calibration runs are never training data (`export` refuses a non-train split) | levels are chosen on tasks that are then thrown away, so no final task is picked for being failed |
| K4 | **Level rule (C1):** per shape, the level whose strict pass@1 is in [0.25, 0.75] and closest to 0.5; the lower level on a tie. No level in band: the shape is dropped. **Minimum (C2):** 4 of 6 train, 2 of 3 val, 2 of 3 hard-test shapes kept, or stop and review | fixed before calibration runs; at G = 8, a task at 0.5 gives the largest GRPO signal |
| K5 | **Final tasks use fresh seeds** at the chosen level: train 20 tasks per shape (seeds 0–19), val 20 per shape, hard test 20 per shape | regression to the mean: tasks selected for failing would "improve" with no learning |
| K6 | **The pool for both arms is the new train shapes only** (about 120 tasks). The old 420 stay solved and stay out; self-SFT's data still carries mix-v3's full training set, which covers them | sampling saturated tasks costs time and adds nothing either arm can use |
| K7 | **Val = the 104 tasks plus the new val shapes** (about 60). The comparison (Tables 2–3 of STAGE10_PLAN.md) is read on the **hard val subset first**, the 104 as a forgetting check; E3 and G6 keep guarding the whole val | the hard subset is where M1 has headroom; the 104 show whether either arm costs old skills |
| K8 | **Gates on the final pool, decided by the pod itself** (Session G): STAGE.md's entry conditions, A1 on the act/ask train shape, and G1 at 3 in 8 of the pool (45 of 120) in two shapes. Pass: the session runs both arms; fail: it stops and packs | Session E lost 8 idle hours waiting for a reading by hand |
| K9 | **The hard test is built and frozen now, never run before Stage 11** (only its calibration seeds are sampled). Stage 11 reports it beside Stage 9's test under a new freeze | the frozen test is near ceiling for mix-v3 too; building it after training would let the results shape it |
| K10 | **Arms unchanged** (STAGE10_PLAN.md E1–E3, G2–G6, C1): self-SFT rounds from the base on mix-v3's data plus M_k's verified set; GRPO from mix-v3's adapter, one step per iteration of 12 tasks × 8 seeds, ten iterations a pass (960 rollouts, one round's sampling), at most three passes | the question is unchanged; only the pool is |

## 3. The shapes

All are `strict=True` (Stage 9 scoring). Answers are `required` unless noted. Levels are
(parameters at L1 → L5).

| Shape | Split | Task | Level | Closest existing shape · how it differs |
|---|---|---|---|---|
| `ordered_migration` | train | apply the N steps of `plan.txt` in order to `data.json` (rename a key, set a value, delete, scale a number, append to a list; later steps name keys earlier steps created); reply `MIGRATED` | N = 2, 4, 6, 9, 12 | test `rename_key_all_files` · one file, a sequence of different edits that depend on each other |
| `capped_allocation` | train | split the budget in `policy.txt` across the K requests in `requests.json` by the policy's rules, rounding down; write `allocations.txt` (`team=amount`, sorted); reply `ALLOCATED` | K = 3–8; rules: proportional; + cap; + floor; + priority team; + leftover to the largest unmet | test `unit_complete_total` · rule-driven division and rounding, written to a file |
| `long_edit_dialogue` | train | T turns of edits to `cart.txt` (`item=qty`), each turn able to refer to items named in any earlier turn; act (even seed): the last turn's referent is clear, reply `UPDATED`; ask (odd seed): it is ambiguous, change nothing, reply `NEED_INPUT` | T = 2, 3, 4, 5, 6 | train `followup_same_record` · up to six turns of cumulative state |
| `mixed_state_batch` | train | K account files and `target.txt`; each account is correct, wrong, or wrong and `locked: yes`. Fix the wrong unlocked ones, leave the rest untouched, reply the sorted ids of the locked wrong ones, or `NONE` | K = 3, 5, 7, 10, 14, locked share rising | val `locked_config_stop`, `no_op_correct_config` · one decision per record, partial action |
| `fault_amid_reconcile` | train, clean/fault | reconcile K stock files against `ledger.csv`, write `discrepancies.txt`; the fault variant fails one read in the middle of the scan | K = 3, 5, 7, 9, 12; the faulted read moves later | test `transient_read_failure` · the fault falls inside a long scan whose result depends on every file |
| `rule_table_apply` | train | apply the R ordered rules of `rules.txt` (first match wins, a default last) to the N customers of `customers.json`; write `discounts.txt`; reply `APPLIED` | (R, N) = (2,3), (3,5), (4,8), (6,10), (8,14) | train `eligibility_record_update` · an ordered rule table over many records |
| `transfer_to_targets` | val | warehouses' stock and targets; move surplus to deficit by the stated order (largest surplus to largest deficit, ties by name); write `transfers.txt` | warehouses 3–8, items 1–3 | train `capped_allocation` · two-sided matching, not one pool |
| `itinerary_dialogue` | val | T turns reordering and shifting the stops of `itinerary.txt`, act/ask on the last turn as above (`UNSURE` when ambiguous) | T = 2–6 | train `long_edit_dialogue` · another domain; edits move and re-date, not add and remove |
| `flag_rollout` | val | turn a flag on for the K services in `services/`; some are already on (leave them), some `frozen` (skip and report); one write fails once at L3–5 | K = 3–12 | train `mixed_state_batch` · three decisions plus a fault in one run |
| `payroll_rule_adjust` | hardtest | rule-driven pay adjustments over N employees with caps | as `capped_allocation` | train `capped_allocation`, `rule_table_apply` |
| `support_dialogue` | hardtest | multi-turn ticket edits, act/ask on the last turn | T = 2–6 | train `long_edit_dialogue` |
| `cert_renewal_batch` | hardtest | renew K certificates; some valid, some pinned, a read fault mid-batch | K = 3–12 | train `mixed_state_batch`, `fault_amid_reconcile` |

## Global Constraints

- The Stage 8 val and test tasks and all 420 train tasks keep their fingerprints (`freeze` manifest comparison); `bench/verify.py` is not changed.
- Calibration (`calib`) and hard-test (`hardtest`) runs are never training data; `hardtest` runs only with `--final`.
- Every new task passes the Item 1 gates at every level, and `tests/test_grpo_exploits.py` covers the new train shapes (it reads `specs("train")`).
- Pod sessions: the cost guard runs on the Mac (the pod's `runpodctl` has no API key); setup runs the tests with `OMP_NUM_THREADS=8`.
- Each task's diff is shown to the user before it runs; commits use `--author="claude <claude@noreply>"`.

## Review Focus

- A level must change only difficulty: at every level the scripted solution passes, and the answer-only and wrong-answer scripts fail (Tasks 2–5, `test_every_level_keeps_the_gates`).
- An act/ask pair at a level shares its records and differs only in the last turn's referent (Task 3, as Item 7's pair test).
- `levels.json` must be the only thing that decides a final shape's level; a missing entry leaves the shape unregistered rather than at a default (Task 1).
- C1 on a tie, and a shape with no level in band, must behave as K4 states (Task 6 tests).
- The pool restriction must cover rotation and sampling alike, or GRPO and self-SFT sample different pools (Task 6).

---

## 4. Tasks — code (local, no GPU)

### Task 1: Levels, the calibration split and the hard-test split

**Files:** modify `evals/bench/skeletons.py` (registration), `evals/bench/run.py` (`specs`, `--split`), `evals/bench/splits.json`; create `evals/bench/levels.json` (`{}` until Session F); test `tests/test_levels.py`.

**Interfaces — produces:** `Shape(name, family, split, recovery, build(rng, root, ctx, level), solution, seeds)`; `SHAPES: list[Shape]`; `CALIB_SEEDS = tuple(range(100, 106))`; `specs("calib")` yields one entry per (shape, level 1–5, calib seed) with skeleton name `f"{name}@L{level}"`; `specs("train" | "val" | "hardtest")` include a shape only if `levels.json` names its level; `bench --split calib|hardtest` (hardtest needs `--final`).

- [ ] **Step 1: Failing tests:** `test_calibration_covers_every_level_on_its_own_seeds` (ids like `ordered_migration@L3-100`, no seed below 100); `test_an_unleveled_shape_is_not_registered` (with `levels.json` empty, no new shape in train/val/hardtest); `test_a_leveled_shape_registers_at_that_level` (patched levels → 20 train tasks built at that level); `test_old_tasks_keep_their_fingerprints` (manifest entries of every pre-existing task unchanged); `test_hardtest_needs_final` (exit 2 without `--final`).
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS).
- [ ] **Step 5:** commit `feat: difficulty levels, a calibration split and a hard-test split`.

### Tasks 2–5: The shapes, in four groups

Each group: builders with `level`, solutions, and tests. Every group's test file runs
`test_every_level_keeps_the_gates` (for each level and two calibration seeds: correct passes,
answer-only fails, wrong-answer fails) and `test_difficulty_grows_with_level` (the level's size
parameter rises with the level), plus the shape-specific tests below.

- **Task 2 — rules and arithmetic:** `ordered_migration`, `capped_allocation`, `rule_table_apply`, `transfer_to_targets`, `payroll_rule_adjust`. Shape tests: a migration applied out of order fails `files`; an allocation off by one unit fails; a rule table applied last-match-wins fails.
- **Task 3 — dialogues:** `long_edit_dialogue`, `itinerary_dialogue`, `support_dialogue`. Shape tests: each act/ask pair shares its records (Item 7's pair test); acting on the ask side fails with `asked_instead` false and `unchanged` false; asking on the act side fails and counts `asked_instead`.
- **Task 4 — mixed states and faults:** `mixed_state_batch`, `fault_amid_reconcile`, `flag_rollout`, `cert_renewal_batch`. Shape tests: touching a locked or already-correct record fails `unchanged`; a fault variant that skips the failed file fails `files` or `evidence`.
- **Task 5 — exploit review extended:** `tests/test_grpo_exploits.py` already reads `specs("train")`; run it with a patched `levels.json` holding every train shape at L3 and L5, and fix any loophole before calibration.

Each: steps as Task 1 (failing tests, run, implement, run, commit `feat: <group> shapes`).

### Task 6: The level rule and the pool restriction

**Files:** create `evals/bench/levels.py` (the `levels` command); modify `evals/sft/grpo.py` (`train_pool(skeletons=None)`, `grpo tasks --skeletons`), `scripts/stage10_lib.sh` (`STAGE10_POOL` → `sample_train` passes `--tasks`), `evals/__main__.py`; tests `tests/test_levels.py`, `tests/test_sft_grpo.py`.

**Interfaces — produces:** `choose_level(rates: dict[int, float]) -> int | None` (C1); `python -m agent_from_scratch.evals levels RUN_DIR... --output FILE` (strict pass@1 per shape and level, the choice, and C2's verdict); `train_pool(skeletons: Iterable[str] | None = None)`; `iteration_tasks(..., per_iteration=2 × kept train shapes)`, so ten iterations still cover the pool once (12 for six shapes).

- [ ] **Tests:** `test_the_level_closest_to_half_wins` ({1: .9, 2: .6, 3: .3} → 2); `test_a_tie_takes_the_lower_level` ({2: .4, 3: .6} → 2); `test_no_level_in_band_drops_the_shape` ({1: .9, 2: .8} → None); `test_c2_needs_four_train_two_val_two_hardtest`; `test_pool_restricted_to_named_shapes` (120 IDs, 6 shapes); `test_rotation_on_the_restricted_pool` (10 iterations × 12, 2 per shape, all 120 once).
- [ ] Run, implement, run, commit `feat: the level rule and a restricted Stage 10 pool`.

### Task 7: Session scripts

**Files:** create `scripts/stage10_calibrate.sh` (serve M1, `bench --split calib` at 8 seeds in parallel, `levels`); create `scripts/stage10_session_g.sh` (Item 1 on the final pool and val at three seeds; K8's gate decided in the script; on pass: the GRPO pilot, then the three self-SFT rounds, then Session D's comparison on one server; each stage packed for the copy-back).

- [ ] `bash -n` both; dry runs with `STAGE10_DRY=1` (expected: calibration lists 12 shapes × 5 levels × 6 seeds; Session G lists the gate, 3 passes of 10 iterations, 3 rounds, and the final val runs).
- [ ] Commit `feat: calibration and Session G drivers`.

## 5. Steps — GPU sessions

- [ ] **Session F — calibration** (one A100, about 1 hour, about \$2): 360 calibration tasks × 8 samples. Then `levels.json` from C1, shown to the user as a diff, and C2. Commit; `freeze` records the new manifest (old fingerprints unchanged).
- [ ] **Session G — the arms** (one A100, about 25 hours, about \$45): Item 1 on the final pool; K8's gate; GRPO pilot (3 passes, about 2.5 hours); self-SFT rounds 1–3 (about 6.5 hours each); comparison on one server (M1, M2–M4, the GRPO passes; val three seeds). Copy-back after each stage; the Mac-side guard deletes the pod.
- [ ] **Record** in this plan, STAGE10_PLAN.md and STAGE.md: Tables 1–4 of STAGE10_PLAN.md, read on the hard val subset first.

## 6. Cost and budget

GPU about \$47 (F \$2, G \$45). The RunPod balance on October 21 is **\$25.49**, so Session G
needs a top-up of at least \$30 first. Spent on Stage 10 so far: Session A \$0.90, Session E about
\$50 including the faulty host's two runs (about \$15) and the idle gate (about \$14).

**Laptop.** Copy-back and the cost guard run on the Mac; while a session runs, the Mac must stay
awake, or the pod bills idle after it finishes. The alternative is a restricted RunPod API key on
the pod so it can delete itself, which only the user can create.
