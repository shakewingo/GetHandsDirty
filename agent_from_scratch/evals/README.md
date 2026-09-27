# Evaluating the agent

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
| Do the units still work? | `python -m unittest discover -s agent_from_scratch/tests -q` | scripted | seconds |
| Does it feel right by hand? | `python -m agent_from_scratch.agent` (REPL) | real | — |

`--output` must be a **new** directory outside the package; evidence is never overwritten.
`--tasks a,b` runs a subset and `--seed N` fixes the per-request seed. `bench` always uses the
frozen protocol in `config.py` (`BENCH_DECODING` sampling, `BENCH_N_CTX`), so `--seed` selects
which sample a run is; `--backend llama_cpp` (default) runs the local GGUF for pipeline checks and
`--backend vllm --base-url URL [--adapter NAME]` the control checkpoint. `pressure` keeps greedy
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
| `__main__.py` | the `bench` / `freeze` / `aggregate` / `pressure` / `compare` dispatcher |
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
