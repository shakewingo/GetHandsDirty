# Foundation dev checks

Current behavior and measured limits: [FOUNDATION_CHECKPOINT.md](FOUNDATION_CHECKPOINT.md).

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
