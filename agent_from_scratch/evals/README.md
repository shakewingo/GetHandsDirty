# Evaluating the agent

Start here. Everything runs from the **repository root** in the project environment through one
entry point:

```sh
python -m agent_from_scratch.evals COMMAND --output outputs/NEW_NAME [options]
```

| I want to know… | Command | Model | Time (16 GB Mac, 32k window) |
|---|---|---|---|
| Did I break anything the agent could already do? | `dev`: 17 small tasks (reading, editing, recovery, stopping) | real | ~2 min, ~4.5 min with planning |
| Does context management hold when evidence outgrows the window? | `pressure`: 4 tasks over 9 files of ~15k characters | real | ~15–25 min |
| How do two runs differ? | `compare outputs/A outputs/B`: paired outcomes, exact McNemar p, trajectory shape | none | instant |
| Does a change hold on the frozen Stage 8 benchmark? | `bench --split dev` (15 tasks, current tools) or `bench --split test --final` (60 tasks, needs a frozen manifest) | real | dev: ~1 min; test: ~5 min |
| Do the units still work? | `python -m unittest discover -s agent_from_scratch/tests -q` | scripted | seconds |
| Does it feel right by hand? | `python -m agent_from_scratch.agent` (REPL) | real | — |

`--output` must be a **new** directory outside the package; evidence is never overwritten.
`--tasks a,b` runs a subset, `--seed N` fixes the per-request seed, and `--n-ctx N` overrides the
window (recorded in `metadata.json`; runs at different windows are not comparable).

## The loop when you change the agent

1. Unit tests. They are the only check that runs without a model.
2. `dev` before and after the change with the same `--seed`. A drop here is a real regression.
3. `pressure` if you touched context, compaction, planning, or a tool that returns bulky output.
   `dev` cannot see these: at 32k it peaks near 8% of the usable window, and elision starts at 60%.
4. `compare` the two runs. Read direction and trajectory, not the p-value; with 17 or 4 tasks the
   exact McNemar test almost never reaches significance.

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
| `__main__.py` | the `dev` / `pressure` / `compare` dispatcher |
| `run.py` | dev suite runner; also `open_run` and `parse_limits`, which every suite shares |
| `pressure.py` | window-pressure suite; files are generated from `--seed`, nothing large is stored |
| `verify.py` | fixed post-run verifiers and per-run `metrics()`; never fed back to the agent |
| `trajectory.py` | `profile()` and `compare()`; the `compare` command |
| `tasks.jsonl`, `splits.json`, `fixtures/` | the frozen dev suite: prompts, permitted tools, verifiers, workspaces |
| `legacy_files.py`, `check.py` | the dev suite's frozen byte-offset file tools and its trusted check command |
| `bench/` | Stage 8 generated benchmark: 14 skeletons (4 dev, 10 test × 6 variants), one content-first verifier, freeze manifest. Shapes and rationale: [docs/STAGE8_DESIGN.md](../docs/STAGE8_DESIGN.md) |
| `core_lines.py` | core-size audit for the review trigger in [context-memory.md](../docs/context-memory.md); not an eval |

The Stage 1, 2A and 2B checkpoints and the old per-stage README are in
[docs/checkpoints/](../docs/checkpoints/). The live-network tool demos are in
`examples/tools_smoke.py` and `examples/tools_demo.py`.

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
  TASK_ID/             task.json, result.json, state/runs/*.jsonl (full trace), final workspace (dev)
```

Fields to read first: `stop_reason` (`context_limit` is overflow, and the target is zero),
`peak_context_ratio`, `elided_messages`, `compact_requests`, `plan_updates`, `stuck_reminders`,
and `actions` (one label per model request: explore, modify, execute, plan, answer or error).
`pressure` also records `answer_found`: `passed` needs the exact format, and `answer_found`
says whether the right value appeared at all, which separates a format slip from lost evidence.

## Adding a task

- **Dev**: add a line to `tasks.jsonl`, a `fixtures/NAME/workspace/` directory and its skeleton to
  `splits.json`; `load_tasks()` validates the contract. The suite is frozen for comparability, so
  a change means a new suite version, not an edit.
- **Pressure**: write a builder in `pressure.py` that fills a temporary directory and returns
  `(prompt, expected_answer)`, then add it to `TASKS`. Keep the answer exact-match, and bump the
  suite name in `main()` when you change an existing task's prompt.

## Limits to remember

- The dev suite uses the frozen byte-offset tools, a 1 KiB read cap, fixed shell commands and
  recorded web replay. It does not exercise `edit_file`, `glob_files`, `grep_text` or live
  networking. Its Stage 2B score of 12/17 predates the Stage 3A prompt and the 32k window; the
  current-code baseline is 9/17.
- A greedy run is one sample, not several trials. Metal decoding is not bit-reproducible: two
  identical configurations can differ on a task (one dev task took 4 requests in one run and 5 in
  another with elision never firing).
- The 7B model batches calls when asked to "read every file". Nine parallel reads exceed the
  eight-call limit and end in invented answers, so pressure tasks ask for one call per response.
- False completion is a claim review, not an automatic check. Write a JSON map of task IDs to
  `{"false_completion": bool, "review_note": str}` and run `dev --output OLD_RUN --review FILE`;
  no model is loaded.
