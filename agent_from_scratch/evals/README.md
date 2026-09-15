# Tiny agent development checks

Current behavior and measured limits: [FOUNDATION_CHECKPOINT.md](FOUNDATION_CHECKPOINT.md).

## Stage 2B behavioral baseline and prompt comparison

[BEHAVIOR_CHECKPOINT.md](BEHAVIOR_CHECKPOINT.md) records the original **8/17** baseline and
three prompt ablations. The current full prompt scores **12/17**, retaining all original
passes. It was selected over the shorter 12/17 candidates for fewer unintended writes and
false claims, with a measured cost: **270.07 s / 5,060 completion tokens**, versus baseline's
78.01 s / 1,020. The extra JSON/tool-call rules do not establish reliable web completion.
Run from the repository root with the project's Python environment:

```sh
python -m agent_from_scratch.evals.run --output outputs/behavior-new --seed 11
python -m agent_from_scratch.evals.run --output outputs/behavior-subset --tasks check_fix_clean,check_fix_fault
```

Runs use the current `prompts/system.md`. Prompt-comparison evidence is under
`outputs/stage2b-prompt-20260915/`: exact prompt snapshots, all condition results/traces,
`experiment.json`, `comparison.json` and the 118-test validation log. To reproduce an older
condition, restore its saved `*-system.md` before starting the command and keep it fixed
until the run ends. Use a fresh output directory; metadata records the actual prompt.

`tasks.jsonl` defines prompts, fixtures, permitted tools, request budgets, fixed verifiers,
source evidence, allowed changes and split allocation. `splits.json` assigns task skeletons
before their paired variants; train/test names are reservations, not finished task sets.
Each task gets a new temporary workspace and empty session state. No memory is implemented yet.
Read results are capped at 1024 bytes in this suite; writes retain the runtime's 8192-byte cap.
`RecordedWeb` replays extracted results, including a first-request 503. It does not exercise
DNS/TLS/HTML parsing; the Stage 2A tool tests and live smoke cover those separately.

The evaluator runs after `Agent.run_turn` returns. It checks JSON types/fields, source byte
ranges, required execution, no-op behavior and unintended writes, including writes later
restored. Final text follows each task's explicit format (numeric calculator answers allow
equivalent number spelling). Per-check results distinguish formatting from missing evidence
or incorrect artifacts. The trusted `check_fixture` command is part of the task environment;
the independent outcome verifier never feeds back, retries, or reopens a completed turn.

Evidence contains settings/source/task/split hashes, per-task schemas/fixture hashes,
before/after snapshots, final workspaces, raw traces/session records, `results.json` and
`summary.json`. Output must be fresh and outside the source package. Temporary paths and
process timing vary; a greedy run is a baseline sample, not multiple independent trials.
Missing usage stays null; invalid-call counts follow runtime error codes, so prose containing
an unexecuted tool block can end a turn without incrementing that counter.

False completion is a separate claim review: a failed task may honestly report failure.
Create a JSON object mapping task IDs to `{"false_completion": true/false, "review_note": "evidence"}`,
then apply it without loading the model or changing automatic outcomes:

```sh
python -m agent_from_scratch.evals.run --output outputs/behavior-new --review claim-review.json
```

This saves `review.json` and `summary-reviewed.json`; omitted tasks remain unreviewed.
The checkpoint's claim annotations were reviewed by the assistant against saved evidence.
Scripted reference trajectories in `tests/test_behavior_eval.py` validate all verifiers and
failure injection. Their pass count is kept separate from real-model capability results.

## Stage 2A tools

Implementation, boundaries, and measured model failures: [TOOLS_CHECKPOINT.md](TOOLS_CHECKPOINT.md).
The explicit tool demo adds `web_fetch` and `shell` to calculator/file tools:

```sh
mkdir -p outputs/tools-demo/workspace
printf '{"output":"old.txt","retries":3}\n' > outputs/tools-demo/workspace/config.json
python -m agent_from_scratch.examples.tools_demo \
  --workspace outputs/tools-demo/workspace --state outputs/tools-demo/state \
  --allow-host docs.python.org
```

Try changing `config.json`'s output to `report.txt`, then requesting `check_fixture`;
or fetch a documentation page and ask to save a short note with its source URL.
The shell command map lives in `examples/tools_demo.py`: `inspect_fixture` and
`check_fixture` run the trusted `examples/fixture_command.py` outside the writable
workspace. Web hosts are exact matches; repeat `--allow-host` to configure more.
Both tools are synchronous. The demo requires a POSIX main thread (macOS/Linux).

Reproduce the real-model tool smoke, using a new output directory each time:

```sh
python -m agent_from_scratch.evals.tools_smoke --output outputs/tools-smoke-new
python -m agent_from_scratch.evals.tools_smoke --output outputs/tools-conversation-new --conversation
```

The first command runs three autonomous tasks; the second uses explicit user turns
and reloads the session between them. Neither is the Stage 2B benchmark. Each saves
actual traces and artifacts, including failed tasks and false completion claims.

## Earlier foundation checks

Run from the repository root in the `transformer-practice` environment:

```sh
python -m unittest discover -s agent_from_scratch/tests -v
python -m agent_from_scratch.agent
```

Use ordinary text, for example `Read tools/files.py` or `Read llm.py in full and briefly summarize it`.
There is no `/read` command or runtime completion callback. Tool results automatically
feed the next model request; a normal answer ends the turn. The model can still
answer too early. `final_response` is a stop reason, not task success.

For a real-model check, copy a fixed source file into a disposable workspace.
Keep the same fixture across comparisons and use a fresh output directory:

```sh
mkdir -p outputs/foundation-demo/workspace
cp agent_from_scratch/llm.py outputs/foundation-demo/workspace/llm.py
python -m agent_from_scratch.evals.foundation \
  --workspace outputs/foundation-demo/workspace \
  --output outputs/foundation-demo/results \
  --cases exact,full,partial,edit --seeds 11 --read-bytes 8192
```

Cases:

- `exact`: `Read file <path>` without extra guidance.
- `full`: explicitly asks to read the whole file and briefly summarize, without tool/offset instructions.
- `guided`: explicitly instructs continuation with next_offset until eof; a separate diagnostic condition.
- `partial`: requests only the first 1024 bytes.
- `history`: the exact request with an actual prior conversation exchange.
- `edit`: resets config.json, requests a change to output, then reads it back.

All cases use the same ordinary runtime. Read coverage is scored against the frozen
fixture only after run_turn returns. No scorer feedback reaches the model.
The retired guarded/read_command conditions exist only in historical evidence.

Run writable cases only in disposable workspaces. `--readonly` blocks writes while
retaining schemas and recording attempts; it cannot be combined with edit.
Output must be outside the workspace. File hashes detect side effects, and a changed
target aborts the remaining comparison. `--read-bytes` changes the default while
retaining the 8192-byte schema ceiling; record explicitly requested sizes when comparing.

Each run is now a single `state/runs/<run_id>.jsonl` file (run schema version 2).
Its model_requests include raw_response, status, error_code/error_message,
finish_reason and reported usage. No success-response or parse-error side files
are created. Session records remain version 1 and contain replayable messages,
without raw backend envelopes. Old evidence files are not rewritten.

Handled interruption saves the run; force-killing the process may lose the active
run. state_dir=None disables persistence. Console logs describe progress; the run
trace supplies the raw output needed to diagnose parsing failures.

Results also retain fixture/template/code hashes, settings, coverage, tool errors,
call counts, elapsed time and final artifacts. answer_relevant starts as null;
review the final answer against the fixture separately. Coverage does not prove
summary quality, and edit-case file-read coverage is not its task-success metric.
Do not overwrite old experiments or aggregate trials across different revisions.
