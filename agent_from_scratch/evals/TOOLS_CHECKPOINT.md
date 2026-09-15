# Stage 2 tools checkpoint — 2026-09-15

## Sequential tool batches

Run `2da9f836826345a1a5b05b1616c4f8f3` contained three syntactically valid calls in each
of three model responses: write → shell → edit. The former one-call restriction rejected
all of them. The model also confused a filename change with a content edit.

- `LLMResponse.tool_calls` holds ordered `ToolCall` records, for native calls or Qwen
  blocks with narration. Parse every call's JSON/shape before execution; accept up to eight.
- Execute sequentially, validating each tool's arguments at dispatch. Stop a batch at the
  first failure; retain completed effects and mark the remainder `skipped` before replanning.
  Ctrl-C interrupts the active call and skips the remainder. A separate 40-attempt turn
  limit prevents batching from bypassing the model-request budget. No parallelism or rollback.
- Each call has a distinct ID/result. Trace schema 3 records `ModelRequest.call_ids`;
  sessions retain the assistant batch and all observations. Existing trace files are unchanged.
- The system prompt/template describe batch behavior. Edit descriptions/errors distinguish
  content replacement from renaming with `shell`/`mv`. The REPL now displays complete
  assistant narration before executing its calls through `run_turn(on_progress=...)`;
  this supports intermediate answers without streaming model tokens.

**Validation:** 167 tests pass; Pyright reports 0 errors / 0 warnings. Coverage includes
ordered write/rename/read, malformed later calls preventing all execution, failure/replanning,
interruption, skipped side effects, budgets, IDs, replay, template rendering and showing
the original path before a rename, followed by the final path after it. Logs:
`outputs/batch-tools-full-tests-20260915.log`, `outputs/batch-tools-pyright-20260915.log`.

**Real-model development checks:** `outputs/batch-tools-smoke-20260915/` contains runnable
`smoke.py` / `progress_smoke.py`, source/prompt/schema snapshots, raw traces, artifacts and checks.
All calls use fresh disposable workspaces; the user's original files were not executed against.
The unchanged original wording and an explicit rename/preserve-content variant were tested:

| Revision | Original wording | Explicit rename wording |
|---|---|---|
| `initial` — batch support + descriptions | Rename failed; false completion claim | Passed all checks |
| `feedback-followup` — clearer edit errors | Renamed and preserved contents; omitted original absolute path from final answer | Passed all checks |
| `final-followup` — final-answer reminder, later removed | Renamed; still omitted original absolute path and added content before moving | Repeated truncated/oversized output; interrupted after 350 s, no tools executed |
| `progress-followup` — visible complete narration, earlier prompt restored | Passed all checks in 3 model responses / 26.95 s; recovered from a mistaken edit via `mv` | Passed all checks in 4 model responses / 48.19 s |

The exact request now passes with the actual REPL reporting behavior: original absolute
path in visible progress before the rename, final path in the terminal answer, original
file gone, contents preserved, sentinel untouched. A dedicated REPL test verifies that
display order against the filesystem. The progress smoke checks displayed progress plus
the final answer; earlier experiments correctly checked only their then-visible final answer.
Artifact checks are unchanged. The model still initially selected the wrong edit operation,
then used the improved error feedback to recover. Failed attempts remain saved.
These are adaptive diagnostics, not independent trials or a general reliability claim.
The historical 12/17 used the earlier runtime, template and prompt.

## Earlier narrated tool-call parser fix

At this revision, the Qwen fallback extracted a single unquoted `<tool_call>` block anywhere in an
assistant response. Narration is retained separately from the normalized tool call;
JSON decoding preserves literal tags inside arguments. Inline/fenced code and Markdown
blockquote lines remain text. Malformed or multiple unquoted calls fail before execution.

Replaying the user's saved response from run `132f4756e301453ebc0443a85e93e921`
now yields `tool_call / edit_file`, instead of `direct`; this check did not execute the
edit or modify the user's file. A scripted backend exercises the real parser and loop
through write → narrated shell rename → narrated read → final answer, checking file
contents, linked observations, session history and raw trace preservation.
These are mechanism checks, not a rerun of the historical 12/17 model benchmark.
The original example also chose the wrong operation: `edit_file` replaces contents;
renaming uses `shell`, and parsing alone does not correct the model's tool choice.

**Validation:** 159 tests pass; Pyright reports 0 errors / 0 warnings. Logs:
`outputs/mixed-tool-call-full-tests-20260915.log` and
`outputs/mixed-tool-call-pyright-20260915.log`. The parser replay is saved in
`outputs/mixed-tool-call-replay-20260915.json`. README and STAGE reflect the new contract;
the system prompt and agent loop are unchanged by this parser fix.

## General filesystem follow-up

The default registry now includes `edit_file` alongside general read/write/list tools.
`tools/files.py` follows Nanobot's filesystem interfaces; document extraction lives in
`tools/file_documents.py`. The old confined, byte-based tools are frozen in
`evals/legacy_files.py` for the original demo and evaluators.

- Read uses 1-based line offsets, numbered content, optional encoding, versions and
  lossless line/character continuation. Defaults: 2,000 lines, 16,000 output characters,
  100 MiB source-file ceiling. PDF ranges select up to 20 pages and return `next_pages`;
  `document_eof` distinguishes a completed page range from a completed PDF.
- PDF, DOCX, XLSX and PPTX return extracted text with page/sheet/slide markers as applicable.
  Spreadsheet formulas remain formulas, not recomputed values. Images return metadata;
  this text-only model cannot inspect pixels or perform OCR. No read deduplication is added.
- Write creates/replaces/appends beyond the former 8 KiB cap. Edit replaces exact text with
  occurrence/line/all-match selection and optional replacement-count/version checks.
  Both use atomic replacement, preserve existing mode bits, and leave no-op content untouched.
  Failed or ambiguous edits preserve the original. Automatic fuzzy replacements are not used.
- Paths may be absolute, home-relative or workspace-relative; confinement is optional.
  Listings include entry types/sizes, recursion, noise-directory filtering and pagination,
  without recursively following symlinks. Local permissions still govern access.

**Validation:** 154 unit/integration tests pass; Pyright reports 0 errors / 0 warnings.
Checks include UTF-8/UTF-16, long-line reconstruction, append/mode/no-op behavior,
stale versions, concurrent modification before replacement, failed-write cleanup,
ambiguous edits, CRLF preservation, symlink cycles and actual PDF/Office/image fixtures.

**Model evidence:** `outputs/filesystem-smoke-20260915-164623/` saves fresh fixtures,
source/settings/schema snapshots, raw traces and artifacts. Targeted line reading and
Word-document extraction passed. The first edit failed because an optional version was
sent as null; nullable optional schemas and clearer selector descriptions fixed the exact
rerun in `nullable-followup/`. A separate directed edit also passed, with artifact checks
confirming only the requested value changed. The initial script-repair case still stopped
with an unexecuted repair in prose. This is an adaptive live smoke, not a new benchmark score.

The system prompt and loop stop policy remain unchanged. Restart the REPL and use `/new`:
normal `read_file.offset` now means lines, while frozen evaluation offsets still mean bytes.
Actual token budgeting/compaction remains Stage 3; character limits do not guarantee context fit.

## General-tool follow-up

The default REPL now uses general `shell(command, working_dir?)`, HTTP/HTTPS
`web_fetch(url, extract_mode?)`, and `web_search(query, count?)`. The old command map
and host-list mode remain available for the original fixture demo and benchmark.
This rescope follows the [nanobot shell/web design](../tools/THIRD_PARTY.md), adapted
to our synchronous loop and structured `ToolResult`/trace contract.

- Shell executes local programs, model-authored scripts, pipes, redirects and environment
  assignments using `/bin/sh` and the active Python environment. Defaults: 30 seconds,
  4,096 bytes per output stream, process-group cleanup. It runs with local user permissions;
  the configured cwd and file-tool path checks do not sandbox general commands.
- Fetch uses `httpx` with gzip/deflate, redirects and local `readability-lxml` extraction.
  `extract_mode="text"` retains all visible text when main-content extraction omits a detail.
  Defaults: 20 seconds, 5 redirects, 262,144 retained decoded body bytes plus lookahead,
  4,096 returned characters. Results include retrieval time, extractor and truncation.
- Search uses `ddgs` without an API key: 1–10 source records, bounded titles/snippets,
  retrieval time and untrusted-content metadata. A search snippet is not a live reading.
  No browser execution, login, remote reader or provider configuration framework is added.

**Mechanism checks at that revision:** 131 tests passed; targeted Pyright reported 0 errors / 0 warnings.
New checks cover general script execution/cwd, pipes/redirection, exit failures, output
caps, descendant cleanup, compressed HTML, readable/full text, 403 observations,
extraction deadline propagation, and search validation/failure/timeout/interruption.

**Live evidence:** `outputs/general-tools-smoke-20260915-162442/` retains source/settings,
schemas, raw traces, artifacts and direct tool results. Direct calls executed the China
clock command, a local Python script and a pipeline; fetched Python docs and `time.is`
with HTTP 200; and returned documentation search results. `timeanddate.com` still returned
403. Successful extraction does not establish completeness or freshness of a clock page.

The first four fresh-session model demos deliberately retain their failures:

| Request | Observed result |
|---|---|
| Search for China's current time | Search succeeded; model incorrectly used yesterday's cached snippet |
| Read China time from the local computer | Shell succeeded; final answer matched stdout |
| Create and run a Python script | Repeated multiple-call responses were rejected; no tools executed |
| Run and repair a broken script | Failed execution and file read succeeded; repair remained unexecuted prose |

The separate `directed/` checks passed **3/3**: execute an existing script, fetch
`example.com`, and find the official Python zoneinfo documentation URL. Each used
one tool call plus a final answer. These establish basic tool usability, not autonomous repair.

The first demo process aborted during native Metal cleanup after all four traces were
saved. The directed follow-up closed the model explicitly and exited cleanly. Both REPL
entry points now release the model in `finally` through `LLM.close()`.

These are live diagnostics, not a replacement benchmark or a general success-rate estimate.
The system prompt, one-call protocol and final-answer stop policy were not changed.
The earlier 12/17 remains the restricted fixture-tool baseline. See [README](README.md)
for installation, fresh-session startup and the general-tools demo.

## Original restricted Stage 2A evidence

Branch: `feat/agent-foundation`; Stage 2A was implemented after `8e27181`.
**Historical 2A checkpoint:** tool implementation and mechanism checks were complete,
while the real model failed the autonomous workflows below. Stage 2B is now complete;
see [the behavioral checkpoint](BEHAVIOR_CHECKPOINT.md) for the baseline and prompt comparison.

## Implementation

- `tools/web.py`: synchronous `web_fetch(url)`, exact configured HTTPS hosts on
  port 443, every redirect checked, at most 3 redirects. Default limits: 15 seconds,
  65,536 body bytes plus one lookahead byte, 4,096 returned characters. Text/HTML/JSON
  results include source/final URL, HTTP status, truncation flags, and `untrusted`.
  HTML extraction removes markup and script/style/head content; it is not a browser
  or full article extractor. Compression is rejected after requesting identity encoding.
- `tools/shell.py`: `shell(command_id)` selects fixed `Command(argv, description,
  timeout)` configuration. It uses workspace cwd, no shell expansion, minimal environment,
  closed stdin, and separate bounded stdout/stderr (4,096 retained raw bytes each).
  A nonzero exit or timeout retains observations in a failed `ToolResult`.
- `ToolExecutionError` carries an error code plus partial output. `ToolInterrupted`
  carries partial observations while stopping the turn. The run boundary records a
  matched interrupted/unknown tool result; incomplete turns never enter replay history.
  Run/session schemas remain 2/1; existing file tools, parsing, prompts, and stop policy are unchanged.
- `examples/tools_demo.py` builds an explicit six-tool registry. Its two fixed commands
  inspect/check `config.json` using a trusted script outside the writable workspace.
  `-I` prevents that Python script from importing code through workspace/Python environment setup.

Web's total request timer uses POSIX `SIGALRM` on the main thread, restores the prior
handler, and refuses to replace an active timer. It bounds the tested slow-header,
slow-body, and delayed resolver cases; this is not an OS process isolation guarantee.
Python signal delivery can be delayed inside native code. See the
[Python signal documentation](https://docs.python.org/3/library/signal.html).
HTTPS uses the standard client's default certificate/hostname verification; see
[HTTPSConnection](https://docs.python.org/3/library/http.client.html#http.client.HTTPSConnection).

Shell cleans up its process group on timeout/Ctrl-C and after completion, then reaps
the direct child. Configure only trusted foreground commands that neither daemonize
nor interpret workspace data as code. This is a POSIX local tool, not a general sandbox;
cwd and fixed argv do not isolate arbitrary programs. The example verifier stays outside
agent-writable state. Launch/configuration commands are in [README.md](README.md).

## Verification

**102 tests pass**, including the previous 82 tests. Coverage includes:

- Actual saved HTML through a local HTTP transport: extraction, text/JSON, denied
  hosts/credentials/ports, redirect escape/loop/failure, byte/text caps, HTTP errors,
  slow headers/body, timer restoration, and interrupted connection cleanup.
- Real subprocesses: both pipes overflowing without deadlock, nonzero exit,
  retained output on timeout, descendant cleanup after leader exit, and real SIGINT.
- Ordinary-loop scripted models: failed check → read → write → passed check,
  failed fetch → fetch → local file, paired trace IDs, and usable REPL after an
  interrupted shell action with no automatic replay. These prove harness behavior.
- Pyright: **0 errors / 0 warnings** across runtime, tools, examples, and eval scripts.

Commands (repository root, project Python environment):

```sh
python -m unittest discover -s agent_from_scratch/tests -v
npx --yes pyright --pythonpath /Users/yingyao/miniconda3/envs/transformer-practice/bin/python agent_from_scratch/*.py agent_from_scratch/tools/*.py agent_from_scratch/examples/*.py agent_from_scratch/evals/*.py
```

Runtime: **1,523 lines / 13 files**; examples: 74 / 2; tests: 1,717 / 9;
eval scripts: 347 / 2. Templates and generated evidence are excluded.

## Real-model evidence and remaining gap

Qwen2.5-7B Q4_K_M, temperature 0, `n_ctx=8000`, `max_tokens=2048`, 12 requests maximum.
Final smoke evidence: `outputs/stage2a-final-20260915/`; each case starts with fresh
workspace/session state. A direct live HTTPS fetch also returned HTTP 200 with explicit
download/text truncation. The page is live, so this is not a frozen benchmark.

| Autonomous request | Executed tools | Actual outcome |
|---|---|---|
| Read/change config, then check | `shell(check_fixture)` | Check failed; config unchanged; proposed follow-up call stayed in ordinary answer text |
| Check, recover, check again | `shell(check_fixture)` | Error was returned correctly; no repair or successful recheck |
| Fetch documentation and save note | `web_fetch` | HTTPS fetch succeeded; answer claimed a saved note, but no write occurred |

All three returned `final_response` after two model requests; **0/3 task outcomes passed**.
The untouched fixture remained intact. Raw traces show the difference between a tool
that executed and a tool-call block embedded in prose. The reviewed parser intentionally
does not execute such prose examples. No runtime verifier, forced continuation, or prompt
patch was added to turn these failures into apparent successes.

A separate five-turn conversation in `outputs/stage2a-conversation-20260915/` reloads
completed history into a fresh Agent each turn, reusing the loaded model. Read, check,
and web calls execute, but proposed writes still remain in ordinary answer text; the
configuration stays unchanged and the note is absent. This does not prove restart of
the native model or autonomous task completion.

Use these failures as Stage 2B cases, keeping tool/protocol errors, premature stopping,
and false completion separate. Small deterministic smokes are not generalization scores.
Logs: `outputs/stage2a-tests-20260915.log`, `outputs/stage2a-pyright-cli-20260915.log`,
`outputs/stage2a-smoke-20260915.log`. Metadata records settings, prompts, source hashes,
schemas, final artifacts, and full run traces. Earlier runs are retained separately.
