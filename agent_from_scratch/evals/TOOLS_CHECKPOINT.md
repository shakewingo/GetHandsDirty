# Stage 2A checkpoint — 2026-09-15

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
