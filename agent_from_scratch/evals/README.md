# Evaluating the agent

## Task data: train, val and test

Tasks are split by shape. Train has 320 tasks from 16 shapes (80 from Item 1's 4 shapes, 240
from the 12 Item 4 added). Val is Stage 8's 64 tasks. Test has 140 tasks: Stage 8's 60, and 80
on 4 shapes that Stage 9 reserved before any training data existed.
Generate initial workspaces and private contracts, optionally running real tool/verifier gates
with a scripted model (no model weights or API calls):

```sh
python -m agent_from_scratch.evals dataset --output outputs/NEW_STAGE9_DATA --check
python -m unittest agent_from_scratch.tests.test_stage9_tasks agent_from_scratch.tests.test_stage9_dataset -v
```

Each task's `workspace/` contains only its inputs; `task.json` beside it contains private
expectations. Do not send the contract or scripted solution to the actor/teacher. A successful
scripted gate is not a training demonstration or a real-model capability measurement.

For Item 2, the ordinary model runner accepts `bench --split train`. Test runs require
`--final --manifest outputs/NEW_STAGE9_DATA/manifest.json` and the frozen vLLM protocol. They
reject drift and limit overrides; use val for selection and reserve test results for the
formal base/adapter comparison. The generated manifest is separate
from the original `evals/bench/manifest.json`. Regeneration creates a new directory/version;
never overwrite earlier evidence or silently refresh a freeze to hide changes.

## Stage 9 training data

Two tracks collect rollouts on the train split, and `export` turns the verifier-passed ones into
`(prompt, completion)` examples, one per model request:

```sh
# Teacher track: an API model (DeepSeek) inside the same harness, tools and verifier.
# Needs DEEPSEEK_API_KEY; the student GGUF only supplies the tokenizer for the context budget.
TMPDIR=/tmp python -m agent_from_scratch.evals bench --backend teacher --split train --seed 0 \
  --student-gguf <Qwen3-4B-Instruct-2507 GGUF> --output outputs/NEW_TEACHER_RUN
# RFT track: N samples of the base checkpoint on vLLM, one run per seed (runbook: docs/benchmark.md).
python -m agent_from_scratch.evals bench --backend vllm --split train --seed 3 --output outputs/NEW_RFT_RUN_S3
# One dataset per track: examples.jsonl plus metadata.json (sources and the funnel).
python -m agent_from_scratch.evals export --track teacher --output outputs/NEW_SFT_TEACHER outputs/NEW_TEACHER_RUN
```

`bench --backend teacher` refuses every split but train, and `export` refuses every run that is
not on the train split, every scripted fixture, and a run from another backend than the track's
(unless `--pilot` marks the output as a pipeline check). `--flawed-turns drop` keeps a verified
rollout that holds the model's own failed tool call or a final answer that is not format-exact,
and leaves only that turn out; the default `reject` refuses the whole rollout. `--max-per-task K`
caps the distinct trajectories kept per task. Design, decisions and results:
[docs/STAGE9_ITEM2_PLAN.md](../docs/STAGE9_ITEM2_PLAN.md).

## Stage 9 public tool-use data

`public` turns rows of a public dataset into examples in `export`'s format, and `mix` combines
them with our own examples into one dataset that `train` reads unchanged:

```sh
# Our system prompt, the row's tools in our template, tool results in our tool-result envelope,
# one example per assistant turn. A row that cannot be converted exactly is dropped and counted.
python -m agent_from_scratch.evals public --source when2call-pref --input when2call_train_pref.jsonl \
  --revision SHA --student-gguf <Qwen3-4B-Instruct-2507 GGUF> --max-examples 2000 --output outputs/NEW_PUBLIC
# Our examples repeated k times, so that they come closest to a third of the rows.
python -m agent_from_scratch.evals mix --own outputs/NEW_SFT_TEACHER --public outputs/NEW_PUBLIC ... \
  --output outputs/NEW_MIX
```

Sources: `when2call-sft`, `when2call-pref` (the chosen reply) and `toucan`. Rows are visited in a
`--seed` order, whole rows at a time. Design and decisions:
[docs/STAGE9_ITEM4_PLAN.md](../docs/STAGE9_ITEM4_PLAN.md).

## Stage 9 external test

Two scorers outside our harness, run against the vLLM server like `bench` (`--adapter NAME`, or
none for the base). Nothing derived from either set is ever trained on.

```sh
# When2Call (dataset nvidia/When2Call, test/): the repository's own MCQ scoring for Qwen, and its
# LLM-as-a-judge path (answers on the server, then DeepSeek classifies them; needs DEEPSEEK_API_KEY).
python -m agent_from_scratch.evals when2call mcq --data when2call_test_mcq.jsonl --output outputs/NEW_MCQ
python -m agent_from_scratch.evals when2call generate --data when2call_test_llm_judge.jsonl --output outputs/NEW_GEN
python -m agent_from_scratch.evals when2call judge --responses outputs/NEW_GEN --output outputs/NEW_JUDGE
# BFCL multi-turn base, a local adaptation (never an official score): BFCL's backends and checker,
# from a checkout of ShishirPatil/gorilla.
python -m agent_from_scratch.evals bfcl --bfcl-root gorilla/berkeley-function-call-leaderboard --output outputs/NEW_BFCL
```

Design, decisions and the Stage 9 results: [docs/STAGE9_ITEM5_6_PLAN.md](../docs/STAGE9_ITEM5_6_PLAN.md).

## Stage 9 adapter training

`train` turns one exported dataset into LoRA adapters, and `replay` checks a served adapter
against the examples it was trained on:

```sh
# Needs requirements-train.txt (torch, transformers, peft). One adapter per epoch, in PEFT
# format, plus train_log.jsonl and metadata.json (dataset hash, selection, recipe, versions).
python -m agent_from_scratch.evals train --examples outputs/NEW_SFT_TEACHER --output outputs/NEW_ADAPTERS
# Against a vLLM server started with --lora-modules NAME=outputs/NEW_ADAPTERS/epoch-3:
python -m agent_from_scratch.evals replay --examples outputs/NEW_SFT_TEACHER --adapter NAME \
  --ids outputs/NEW_ADAPTERS/metadata.json --output outputs/NEW_REPLAY.json
python -m agent_from_scratch.evals bench --backend vllm --split val --seed 0 --adapter NAME --output outputs/NEW_VAL_RUN
```

The loss falls on the assistant's completion only; the prompt, with its tool results and
runtime feedback, is context. `train` refuses a pilot dataset without `--pilot`, a dataset
exported under another chat template, and a row longer than the window, which is never
truncated. `--max-per-skeleton N` caps the examples per skeleton in whole trajectories, and
`--limit N` with `--save-last` is the overfit smoke test. `replay` asks for a greedy
continuation of each prompt and reports the exact matches; omit `--adapter` for the base
model. Its rate is a fit measure on training prompts, never a capability result. Design and
decisions: [docs/STAGE9_ITEM3_PLAN.md](../docs/STAGE9_ITEM3_PLAN.md). How to run it on a GPU:
the next section.

### Training on a rented GPU (RunPod)

Training and every adapter evaluation need a CUDA GPU. We rent one RunPod pod per session and
drive it from the Mac with `runpodctl`. The 4B model in bf16 needs a GPU with at least 24 GB; we
use an A40 (48 GB). A session has six parts:

1. **Prepare.** `runpodctl` 2.14 or later with its API key (`runpodctl doctor`), and your SSH
   key added with `runpodctl ssh add-key` *before* the pod is created. Commit first: the pod
   runs a shallow clone of the branch's last commit.
2. **Rent.** `--min-cuda-version 13.0` is required, because vLLM 0.30.0 installs PyTorch for
   CUDA 13. If no instance is available, `runpodctl gpu list` shows where each GPU is in stock.
   The pod cannot stop itself, so start a cost guard on the Mac
   (`sleep 7200 && runpodctl pod stop POD_ID`) and copy results back well before it fires.

   ```sh
   runpodctl pod create --name NAME --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
     --gpu-id "NVIDIA A40" --cloud-type SECURE --min-cuda-version 13.0 \
     --container-disk-in-gb 60 --ports "22/tcp" --wait --wait-timeout 15m
   runpodctl ssh info POD_ID   # the ssh command: host, port and key
   ```

3. **Ship and install.** Copy a shallow clone (tar it with `COPYFILE_DISABLE=1` on macOS) and
   the exported datasets, which are not in git. On the pod, `chown -R root:root` the clone,
   then install into a venv. Do not install `requirements-train.txt` there: vLLM pins its own
   torch and transformers, and an adapter should be trained with the stack that serves it.

   ```sh
   uv pip install vllm==0.30.0 peft -r agent_from_scratch/requirements.txt
   python -m unittest agent_from_scratch.tests.test_sft_tensorize \
     agent_from_scratch.tests.test_sft_train agent_from_scratch.tests.test_sft_replay   # none skipped
   ```

4. **Train** before starting the server, which takes the GPU memory. Run the overfit smoke
   first (`train --limit 8 --epochs 40 --accumulate 4 --lr 3e-4 --save-last`, `last_loss` below
   0.05); its `elapsed_seconds` tells you how long the full runs will take. Start long runs with
   `setsid nohup ... &` so a dropped SSH session does not stop them. The three Stage 9
   candidates (teacher-v3, teacher-wide, mix) are trained, one after the other, by
   [`scripts/train_stage9_adapters.sh`](../scripts/train_stage9_adapters.sh); its header gives
   the GPU each needs and how long each takes.
5. **Serve and evaluate.** Use the Stage 8 server command unchanged and add one
   `--lora-modules NAME=ADAPTER_DIR` per adapter. Then `replay --adapter NAME` checks the reload,
   and `bench --backend vllm --adapter NAME` runs the evaluation.
6. **Copy back and delete.** Copy `outputs/` into a new local directory and check it, then
   `runpodctl pod delete POD_ID`, confirm `runpodctl pod list --all` is empty, and read the cost
   from the balance in `runpodctl user`.

The server command, the reasons for each setting and the traps that cost a retry are in the
[GPU runbook in docs/benchmark.md](../docs/benchmark.md#runbook-control-baseline-on-a-rented-gpu-as-run-september-27-2026).
The exact commands of each Stage 9 session are in its plan (Item 3: Step 8 and Appendix F of
[docs/STAGE9_ITEM3_PLAN.md](../docs/STAGE9_ITEM3_PLAN.md)).

## Stage 10 expert iteration and the GRPO pilot

Both arms start from M1 (mix) and sample the whole train split at 8 seeds; they are compared on
val only. Design and decisions: [docs/STAGE10_PLAN.md](../docs/STAGE10_PLAN.md).

```sh
# How good a data generator is a model? Both counts: `passed`, and GRPO's strict reward
# (passed, with the answer exactly as requested; a wrapped advisory answer counts as a failure).
python -m agent_from_scratch.evals aggregate outputs/RUN-s0 ... outputs/RUN-s7 --strict
# Self-SFT round: M1's data plus M_k's own verified rollouts, added once (the teacher set's
# repeat factor is unchanged).
python -m agent_from_scratch.evals mix --own TEACHER --public WHEN2CALL TOUCAN --self RFT_EXPORT --output DIR
# GRPO: the 42 task IDs of one iteration (2 per skeleton; 10 iterations cover the split once),
# then one optimizer step on that iteration's runs. The step writes the next adapter with the
# AdamW state beside it (optimizer.pt); with no mixed group it copies the adapter and says skipped.
python -m agent_from_scratch.evals grpo tasks --pass 1 --iteration 0
python -m agent_from_scratch.evals grpo step --runs outputs/IT-s0 ... outputs/IT-s7 \
  --adapter-in ADAPTER_DIR --output outputs/NEXT_STEP [--lr 1e-5]
```

A GRPO row is one model request of a rollout whose task had mixed outcomes: the prompt as sent
and the text as sampled (not the canonical rendering), weighted by the rollout's advantage
$A_i = (r_i - \bar r)/\sigma_r$ within the task's group. On a pod, the drivers in `scripts/` run
whole sessions: `stage10_sample.sh` (one model on the train split), `stage10_ei_round.sh K` (one
self-SFT round) and `stage10_grpo.sh PASSES` (the pilot, resumable); `STAGE10_DRY=1` prints their
commands without running them. `tests/test_grpo_exploits.py` checks that cheap cheats earn no
reward.

## Reproduce Stage 11

The final numbers are in [docs/results.md](../docs/results.md). To reproduce one of them:

1. **Setup** on a GPU pod (previous section; one A40): a clean clone of the commit, the adapters
   under `outputs/`, and `uv pip install vllm==0.30.0 peft -r agent_from_scratch/requirements.txt`.
2. **Reset the fixtures and run the scripted gates.** Every task's workspace is rebuilt from its
   seed, so there is no state to reset by hand. `dataset --check` writes the versioned freeze and
   runs each train and test task's correct script and two cheats through the agent loop:

   ```sh
   python -m agent_from_scratch.evals dataset --output outputs/NEW_DATA --check   # gates_passed: true
   ```
3. **Reload the adapters.** Start the Stage 8 server command with one
   `--lora-modules NAME=ADAPTER_DIR` per adapter (as `serve` in `scripts/stage10_lib.sh` does), then
   check a reload with `replay --adapter NAME` against that adapter's training examples.
4. **One test run**, under the freeze:

   ```sh
   python -m agent_from_scratch.evals bench --backend vllm --split test --final \
     --manifest outputs/NEW_DATA/manifest.json --seed 0 --adapter NAME --output outputs/NEW_TEST_S0
   ```

   `--split hardtest` works the same way. Sampling differs between server sessions, so a rerun
   matches the reported pass@1 closely, not exactly.
5. **The tables.** `report` takes one run per seed for each arm:

   ```sh
   python -m agent_from_scratch.evals report --arm "A=RUN_S0,RUN_S1,RUN_S2" --arm "B=..." \
     --pair A:B [--subset NAME=SKELETON,...] --output TABLE.md
   ```

The whole session is `scripts/stage11_session.sh` (freeze, test, hard test, external), after
`scripts/stage11_calibrate.sh` has chosen the hard test's levels. The pod cannot delete itself:
keep the Mac awake, open and on power until the cost guard has copied the results back and
deleted the pod. A closed lid on battery stops the guard and the pod bills idle.

Start here. Everything runs from the **repository root** in the project environment through one
entry point:

```sh
python -m agent_from_scratch.evals COMMAND --output outputs/NEW_NAME [options]
```

| I want to know… | Command | Model | Time (16 GB Mac, 32k window) |
|---|---|---|---|
| Did I break anything the agent could already do? | `bench --split val`: 64 generated tasks (inspection, updates, recovery, stopping) | real | ~12 min with the Qwen3-4B Q8_0 GGUF (`--model-path`, measured) |
| What is the control checkpoint's baseline? | `bench --backend vllm` per seed in `BENCH_SEEDS`, then `aggregate`; `--split test --final --manifest FILE` refuses on manifest drift or a protocol mismatch. Runbook: [docs/benchmark.md](../docs/benchmark.md) | real (GPU) | see runbook |
| How reliable is a split across samples? | `aggregate outputs/s0 outputs/s1 outputs/s2`: per-task pass counts, mean pass@1, pass^k, pass@k, mixed-outcome tasks | none | instant |
| Does context management hold when evidence outgrows the window? | `pressure`: 4 tasks over 9 files of ~15k characters | real | ~15–25 min |
| How do two runs differ? | `compare outputs/A outputs/B`: paired outcomes, exact McNemar p, trajectory shape | none | instant |
| How do I collect teacher demonstrations? | `bench --backend teacher --split train --student-gguf FILE`: an API model in the same harness | API | ~7 min for the 80 train tasks (measured) |
| How do verified train runs become SFT data? | `export --track {teacher,rft} RUN_DIR ...`: one example per model request, with provenance and a funnel | none | seconds |
| How do I train an adapter on that data? | `train --examples DATASET_DIR`: LoRA with loss on the completion only, one adapter per epoch | real (GPU) | 3 epochs: teacher-v3 17 min on an A40; teacher-wide 28 min and mix 5.3 h on an A100 (measured) |
| Does a served adapter reproduce its targets? | `replay --examples DATASET_DIR --adapter NAME`: greedy continuations against the targets | real (vLLM) | not yet measured |
| Do the units still work? | `python -m unittest discover -s agent_from_scratch/tests -q` | scripted | seconds |
| Does it feel right by hand? | `python -m agent_from_scratch.agent` (REPL) | real | — |

`--output` must be a **new** directory outside the package; evidence is never overwritten.
`--tasks a,b` runs a subset and `--seed N` fixes the per-request seed. `bench` always uses the
frozen protocol in `config.py` (`BENCH_DECODING` sampling, `BENCH_N_CTX`), so `--seed` selects
which sample a run is; `--backend llama_cpp` (default) runs the local GGUF for pipeline checks and
`--backend vllm --base-url URL [--adapter NAME]` the control checkpoint; `--backend teacher` runs
an API model on the train split only (Stage 9 training data, above). `pressure` keeps greedy
decoding and takes `--n-ctx N` to override the window (recorded in `metadata.json`; runs at
different windows are not comparable).

## The loop when you change the agent

1. Unit tests. They are the only check that runs without a model.
2. `bench --split val` before and after the change with the same `--seed`. Never tune against
   `--split test`: it is reserved for weight comparisons (Stages 9–11).
3. `pressure` if you touched context, compaction, planning, or a tool that returns bulky output.
   `bench` cannot see these: its tasks stay far below the window's elision and summary triggers.
4. `compare` the two runs. Read direction and trajectory, not the p-value; with a few dozen tasks
   the exact McNemar test rarely reaches significance.

To ablate one mechanism, keep everything else fixed and pass any `AgentLimits` field as JSON.
`metadata.json` records what was on:

```sh
python -m agent_from_scratch.evals pressure --output outputs/t3 --limits '{"elide_ratio": null}'
python -m agent_from_scratch.evals pressure --output outputs/t4
python -m agent_from_scratch.evals compare outputs/t3 outputs/t4
```

Run them one after another: each process holds about 6 GB, and two together slow both.

## What is in this folder

| File | Role |
|---|---|
| `__main__.py` | the `bench` / `freeze` / `aggregate` / `dataset` / `export` / `public` / `mix` / `train` / `replay` / `grpo` / `pressure` / `compare` dispatcher |
| `sft/` | Stage 9 Items 2–4: `export.py` turns verified train rollouts into SFT examples (eligibility, selection, the `export` command); `public.py` converts public tool-use rows into the same format; `mix.py` combines our examples with public ones; `teacher.py` is the DeepSeek teacher backend behind `bench --backend teacher`; `tensorize.py` turns an example into a token row with labels on the completion only; `train.py` is the LoRA training loop (`fit`, and `policy_step` for GRPO) and the `train` command; `replay.py` is the `replay` command; `grpo.py` is Stage 10's GRPO pilot (strict reward, advantages, task rotation, rollout rows, the `grpo` command) |
| `bench/` | generated benchmark: 38 skeletons (16 train = 320 tasks, 8 val = 64, 14 test = 140), one content-first verifier (`verify.py`), the trusted fixture check command (`check.py`), freeze manifest and protocol check (`manifest.py`), `aggregate.py` for k-sample runs. Shapes and rationale: [docs/STAGE8_DESIGN.md](../docs/STAGE8_DESIGN.md) |
| `run.py` | `open_run`, `parse_limits` and `save`, which every suite shares |
| `verify.py` | shared post-run measurement: `metrics()`, `snapshot()`, `exchanges()`; `pressure`'s answer verifier and `summarize()`. Never fed back to the agent |
| `trajectory.py` | `profile()` and `compare()`; the `compare` command |
| `pressure.py` | window-pressure suite; files are generated from `--seed`, nothing large is stored |
| `core_lines.py` | core-size audit for the review trigger in [context-memory.md](../docs/context-memory.md); not an eval |

The Stage 2B suite (17 hand-written tasks on frozen byte-offset file tools, 1 KiB reads and
recorded web replay) was retired after Stage 8: `bench --split val` covers the same four families
on the current tools. Its tasks, fixtures and runner are in git history at `ce2111a`, and its
results are summarized in [docs/STAGE.md](../docs/STAGE.md) (Stage 2B).
The live-network tool demos are in `examples/tools_smoke.py` and `examples/tools_demo.py`.

`freeze` writes `evals/bench/manifest.json`, the Stage 8 freeze of the val tasks, the 60 Stage 8
test tasks and the protocol (source, prompts, decoding, limits); a test asserts those tasks have
not changed. A test run is checked against the versioned manifest that `dataset --check` writes:
`bench --split test --final --manifest FILE` refuses to run if the tree has since drifted.

## Reading a run directory

```
outputs/NAME/
  metadata.json        settings (incl. n_ctx), seed, git revision, source/task hashes, limit overrides
  results.json         one record per task: passed, checks, stop_reason, metrics()
  summary.json         pass counts by family, requests, usage
  trajectory.json      profile(): stop reasons, survival by request, action mix, mechanism use
  TASK_ID/             task.json, result.json, state/runs/*.jsonl (full trace)
```

Fields to read first: `stop_reason` (`context_limit` is overflow, and the target is zero),
`peak_context_ratio`, `elided_messages`, `compact_requests`, `plan_updates`, `stuck_reminders`,
and `actions` (one label per model request: explore, modify, execute, plan, answer or error).
`pressure` also records `answer_found`: `passed` needs the exact format, and `answer_found`
says whether the right value appeared at all, which separates a format slip from lost evidence.

## Adding a task

- **Bench**: write a builder and a scripted `solution()` in `bench/skeletons.py`, register it with
  a split, and list it in `bench/splits.json`. The gate tests require the solution to pass and a
  fake "done" to fail. Any change to a frozen skeleton, tool, prompt, limit or verifier needs a new
  `freeze`; never add or retune a test skeleton after looking at test-split results.
- **Pressure**: write a builder in `pressure.py` that fills a temporary directory and returns
  `(prompt, expected_answer)`, then add it to `TASKS`. Keep the answer exact-match, and bump the
  suite name in `main()` when you change an existing task's prompt.

## Limits to remember

- A greedy run is one sample, not several trials. Metal decoding is not bit-reproducible: two
  identical configurations can differ on a task (one task took 4 requests in one run and 5 in
  another with elision never firing).
- The 7B model batches calls when asked to "read every file". Nine parallel reads exceed the
  eight-call limit and end in invented answers, so pressure tasks ask for one call per response.
- `bench` flags false completion automatically: a reply that claims done or a value while the
  task's state checks fail.
