# Evaluating the agent

## Stage 9 task data

The general-assistant training pool has 80 tasks from 4 shapes; a separate `holdout` split
reserves 80 tasks from 4 different shapes. Stage 8's 64 dev / 60 test tasks remain unchanged.
Generate initial workspaces and private contracts, optionally running real tool/verifier gates
with a scripted model (no model weights or API calls):

```sh
python -m agent_from_scratch.evals dataset --output outputs/NEW_STAGE9_DATA --check
python -m unittest agent_from_scratch.tests.test_stage9_tasks agent_from_scratch.tests.test_stage9_dataset -v
```

Each task's `workspace/` contains only its inputs; `task.json` beside it contains private
expectations. Do not send the contract or scripted solution to the actor/teacher. A successful
scripted gate is not a training demonstration or a real-model capability measurement.

For Item 2, the ordinary model runner accepts `bench --split train`. Reserved `holdout` model
runs require `--final --manifest outputs/NEW_STAGE9_DATA/manifest.json` and the frozen vLLM
protocol. They reject drift and limit overrides; use dev for selection and reserve holdout
results for the formal base/adapter comparison. The generated Stage 9 manifest is separate
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
python -m agent_from_scratch.evals bench --backend vllm --split dev --seed 0 --adapter NAME --output outputs/NEW_DEV_RUN
```

The loss falls on the assistant's completion only; the prompt, with its tool results and
runtime feedback, is context. `train` refuses a pilot dataset without `--pilot`, a dataset
exported under another chat template, and a row longer than the window, which is never
truncated. `--max-per-skeleton N` caps the examples per skeleton in whole trajectories, and
`--limit N` with `--save-last` is the overfit smoke test. `replay` asks for a greedy
continuation of each prompt and reports the exact matches; omit `--adapter` for the base
model. Its rate is a fit measure on training prompts, never a capability result. Design,
decisions and the pod runbook: [docs/STAGE9_ITEM3_PLAN.md](../docs/STAGE9_ITEM3_PLAN.md).

Start here. Everything runs from the **repository root** in the project environment through one
entry point:

```sh
python -m agent_from_scratch.evals COMMAND --output outputs/NEW_NAME [options]
```

| I want to know… | Command | Model | Time (16 GB Mac, 32k window) |
|---|---|---|---|
| Did I break anything the agent could already do? | `bench --split dev`: 64 generated tasks (inspection, updates, recovery, stopping) | real | ~12 min with the Qwen3-4B Q8_0 GGUF (`--model-path`, measured) |
| What is the control checkpoint's baseline? | `bench --backend vllm` per seed in `BENCH_SEEDS`, then `aggregate`; `--split test --final` refuses on manifest drift or a protocol mismatch. Runbook: [docs/benchmark.md](../docs/benchmark.md) | real (GPU) | see runbook |
| How reliable is a split across samples? | `aggregate outputs/s0 outputs/s1 outputs/s2`: per-task pass counts, mean pass@1, pass^k, pass@k, mixed-outcome tasks | none | instant |
| Does context management hold when evidence outgrows the window? | `pressure`: 4 tasks over 9 files of ~15k characters | real | ~15–25 min |
| How do two runs differ? | `compare outputs/A outputs/B`: paired outcomes, exact McNemar p, trajectory shape | none | instant |
| How do I collect teacher demonstrations? | `bench --backend teacher --split train --student-gguf FILE`: an API model in the same harness | API | ~7 min for the 80 train tasks (measured) |
| How do verified train runs become SFT data? | `export --track {teacher,rft} RUN_DIR ...`: one example per model request, with provenance and a funnel | none | seconds |
| How do I train an adapter on that data? | `train --examples DATASET_DIR`: LoRA with loss on the completion only, one adapter per epoch | real (GPU) | not yet measured |
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
2. `bench --split dev` before and after the change with the same `--seed`. Never tune against
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
| `__main__.py` | the `bench` / `freeze` / `aggregate` / `dataset` / `export` / `train` / `replay` / `pressure` / `compare` dispatcher |
| `sft/` | Stage 9 Items 2 and 3: `export.py` turns verified train rollouts into SFT examples (eligibility, selection, the `export` command); `teacher.py` is the DeepSeek teacher backend behind `bench --backend teacher`; `tensorize.py` turns an example into a token row with labels on the completion only; `train.py` is the LoRA training loop and the `train` command; `replay.py` is the `replay` command |
| `bench/` | Stage 8 generated benchmark: 18 skeletons (8 dev = 64 tasks, 10 test = 60 tasks), one content-first verifier (`verify.py`), the trusted fixture check command (`check.py`), freeze manifest and protocol check (`manifest.py`), `aggregate.py` for k-sample runs. Shapes and rationale: [docs/STAGE8_DESIGN.md](../docs/STAGE8_DESIGN.md) |
| `run.py` | `open_run`, `parse_limits` and `save`, which every suite shares |
| `verify.py` | shared post-run measurement: `metrics()`, `snapshot()`, `exchanges()`; `pressure`'s answer verifier and `summarize()`. Never fed back to the agent |
| `trajectory.py` | `profile()` and `compare()`; the `compare` command |
| `pressure.py` | window-pressure suite; files are generated from `--seed`, nothing large is stored |
| `core_lines.py` | core-size audit for the review trigger in [context-memory.md](../docs/context-memory.md); not an eval |

The Stage 2B `dev` suite (17 hand-written tasks on frozen byte-offset file tools, 1 KiB reads and
recorded web replay) was retired after Stage 8: `bench --split dev` covers the same four families
on the current tools. Its tasks, fixtures and runner are in git history at `ce2111a`, and its
results are summarized in [docs/STAGE.md](../docs/STAGE.md) (Stage 2B).
The live-network tool demos are in `examples/tools_smoke.py` and `examples/tools_demo.py`.

`freeze` writes `evals/bench/manifest.json`, hashing everything the test split must not drift
from (source, prompts, decoding, limits, the task set itself). Run it once, after the last
skeleton lands; `bench --split test --final` refuses to run if the tree has since drifted.

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
