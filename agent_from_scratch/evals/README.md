# Tiny agent development checks

Current tools and measured limits: [TOOLS_CHECKPOINT.md](TOOLS_CHECKPOINT.md).
Earlier loop/session evidence: [FOUNDATION_CHECKPOINT.md](FOUNDATION_CHECKPOINT.md).

## Stage 2B behavioral baseline and prompt comparison

[BEHAVIOR_CHECKPOINT.md](BEHAVIOR_CHECKPOINT.md) records the original **8/17** baseline and
three prompt ablations. The selected prompt scored **12/17** with the restricted fixture tools, retaining all original
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
condition, use the code at that checkpoint as well as its saved `*-system.md` and keep
both fixed until the run ends. The parser now accepts narrated call batches; restoring
only an old prompt does not restore the old harness. Use a fresh output directory;
metadata records the actual prompt.

`tasks.jsonl` defines prompts, fixtures, permitted tools, request budgets, fixed verifiers,
source evidence, allowed changes and split allocation. `splits.json` assigns task skeletons
before their paired variants; train/test names are reservations, not finished task sets.
Each task gets a new temporary workspace and empty session state. No memory is implemented yet.
Read results are capped at 1024 bytes in this suite; writes retain the frozen legacy tools' 8192-byte cap.
`RecordedWeb` replays extracted results, including a first-request 503. It does not exercise
DNS/TLS/HTML parsing; the Stage 2A tool tests and live smoke cover those separately.
The suite retains fixed shell commands, the original recorded-web schema/description,
and byte-based file tools in `evals/legacy_files.py`. Its 12/17 result does not measure
the expanded filesystem/shell/search/fetch configuration.
Its write checks combine `write_file` call evidence with final filesystem snapshots;
they do not detect every transient change restored by general shell or `edit_file` calls.
An expanded-tool benchmark needs corresponding verifier coverage before claiming those checks.

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
Missing usage stays null; invalid-call counts follow runtime error codes. The historical
parser treated prose containing a tool block as a final answer. The current parser extracts
up to eight unquoted `<tool_call>` blocks with surrounding narration and executes them
in order before continuing the loop. Narration stays in history; raw output stays in the trace.
The REPL prints complete narration messages before dispatching their calls, so intermediate
answers reach the user. Library callers can receive them with `run_turn(on_progress=...)`.
This is message-level progress; model generation remains non-streaming.
Malformed call JSON/shapes or oversized batches trigger parse-error recovery before execution.
Each action gets its own ID/result. On failure the remaining calls receive `skipped` results
and the model can replan. Ctrl-C interrupts the turn; a 40-attempt turn budget stops excess calls.
Completed actions are retained; batches are not transactions. Calls needing newly observed
arguments must wait for another model request. Execution remains synchronous.
Metrics distinguish announced `tool_calls`, invoked `tool_attempts`, and `skipped_calls`.
Use inline/fenced code or Markdown blockquotes for literal examples; an unquoted block is an action.

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
The default registry in `tools/register.py` now includes ten tools (the eight below plus
`glob_files` and `grep_text`; see [harness-empirical-study.md](../docs/harness-empirical-study.md)): calculator,
list/read/write/edit files, general shell, web fetch and web search. Install the tool dependencies
in the project's Python environment, then restart the REPL and use `/new` for a fresh session:

```sh
python -m pip install -r agent_from_scratch/requirements-tools.txt
python -m agent_from_scratch.agent
```

- `read_file(path, offset?, limit?, column?, pages?, encoding?)` returns line-numbered text,
  starting at **line 1**, with a default 2,000-line / 16,000-character window. Continue with
  `next_offset` and `next_column`; PDF page ranges have a separate `next_pages` cursor.
  Reads text/code, PDF, DOCX, XLSX and PPTX. Images return metadata only; no OCR or vision.
- `write_file(path, content, append?)` creates/replaces text or appends it; parent directories
  are created. Writes can exceed the old 8 KiB limit (100 MiB file ceiling), preserve existing
  permissions and leave identical content untouched. `encoding` defaults to UTF-8.
- `edit_file(path, old_text, new_text, ...)` changes exact text without regenerating the file.
  Select `occurrence`, `line_hint` or `replace_all` for repeated matches; optionally check
  `expected_replacements` and `expected_version` from a read. Ambiguous/stale edits fail
  without replacing the file. Full writes also accept `expected_version`.
- `list_files(path, recursive?, max_entries?, offset?, include_ignored?)` returns names, types
  and sizes. It paginates, skips common build/cache directories and does not follow symlink loops.
- `shell(command, working_dir?)` runs local commands/scripts, pipes and redirects using
  `/bin/sh`, the active Python environment and a 30-second deadline. Stdout/stderr retain
  4,096 bytes each. The default cwd is `agent_from_scratch`; cwd is not a sandbox.
- `web_search(query, count?)` discovers URLs through `ddgs`, without an API key; returns
  titles, URLs and bounded snippets. Snippets may be stale.
- `web_fetch(url, extract_mode?)` reads HTTP/HTTPS without a default host list. It supports
  gzip/deflate and local readability extraction; `extract_mode="text"` retains navigation
  in the extracted HTML text. Neither mode evaluates CSS visibility or renders the page.
  Limits: 20 seconds, 5 redirects, 262,144 decoded body bytes and 4,096 output characters.
  It does not execute JavaScript or sign in; HTTP 403 remains an observable failure.

File tools accept absolute, `~`, and workspace-relative paths. Normal REPL file tools are
unrestricted; pass `restrict_to_workspace=True` when constructing an isolated file registry.
The old demo/evaluations retain confined byte-offset tools. Restart and use `/new` after
this update so old byte-offset messages are not reused as line-based calls. PDF/Office
extraction is text-only. Stage 3A now measures and enforces request fit; compaction remains planned.

Try: `Get China's current time using the local computer clock`, or
`Create hello.py that prints hello, run it with Python and report the actual output`.
For general tools with a separate working directory:

```sh
mkdir -p outputs/general-demo/workspace
python -m agent_from_scratch.examples.tools_demo --general-tools \
  --workspace outputs/general-demo/workspace --state outputs/general-demo/state
```

The original fixture demo retains its fixed command/host configuration:

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

## Stage 3A instruction layers

`context.py` loads the static system prompt, an explicitly configured user defaults file,
and `AGENTS.md` directly inside the configured workspace. Instructions are loaded once
per turn in that order, with source hashes in `TurnResult.settings["instructions"]`.
The message builder uses that snapshot before each generation; files are reread next turn.

The default `python -m agent_from_scratch.agent` REPL uses `agent_from_scratch` as its
instruction workspace, matching its tools. Library `Agent(...)` calls default to system-only
instructions; opt in with `instruction_config=InstructionConfig(workspace=..., user_path=...)`
from `agent_from_scratch.context`. Frozen evaluators do not load personal/workspace rules.

The existing demo accepts an explicit user file and uses its `--workspace` for root rules:

```sh
python -m agent_from_scratch.examples.tools_demo --general-tools \
  --workspace outputs/general-demo/workspace --state outputs/general-demo/state \
  --user-instructions /absolute/path/to/user-rules.md
```

The workspace must exist. Omit `--user-instructions` to disable that layer. An absent root
`AGENTS.md` is optional, but a configured user file must exist. Invalid UTF-8, unreadable
or non-regular files, workspace rule symlinks escaping the root, and oversized sources
fail before generation. The REPL reports the error and remains usable. Defaults: 8 KiB
per source and 16 KiB assembled (including labels); these are not token-window guarantees.
No ancestor discovery, include processing or Jinja rendering occurs in instruction files.

Current explicit requests take priority over saved preferences according to the system
prompt; source order alone does not enforce compliance. The updated system prompt is a
new condition and does not inherit the historical restricted-tool 12/17 score.

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

Each run is a single `state/runs/<run_id>.jsonl` file (run schema version 4).
Requests record `purpose` (`agent`/`compact`), actual `input_messages` and `tools`,
plus covered/last-sent raw boundaries. Summary requests include before/after budgets.
For older traces without actual inputs, use the raw `input_message_count` prefix.
The full raw transcript still supplies session deltas and outcome-verifier evidence;
summary calls count in request/usage metrics and never create tool observations.
`ModelRequest.call_ids` links all calls in a batch; older schema-2 traces use singular `call_id`.
Its model_requests include raw_response, status, error_code/error_message,
finish_reason and reported usage. Each model request also records pre-generation `context`:
prompt tokens, effective window, configured response reserve and remaining room. This is
exact for the installed project Qwen formatter; unsupported handlers have unavailable/null
counts. Stage 3A item 4 blocks generation unless the measured remaining room covers the
configured margin (default 256 tokens), stopping the turn with `context_limit`.
Blocked entries stay in the trace but are excluded from model-call and usage metrics.
Behavioral metrics report zero usage for zero actual calls; actual calls with missing usage
remain unknown. Completed tool effects and raw results are retained, while unsuccessful
turns remain excluded from session replay. See the
[fit enforcement notes](../docs/CONTEXT_STATE_DESIGN.md#implemented-request-fit-enforcement).
No success-response or parse-error side files
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
