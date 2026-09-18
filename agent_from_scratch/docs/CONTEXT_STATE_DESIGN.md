# Context semantics and state ownership

Updated September 18, 2026. Stage 3A and Stage 3B item 1 are implemented; checkpoints
remain planned. Stage 2's 167-test handoff and
historical model-score boundary are recorded in [STAGE.md](STAGE.md).

## Implemented: one preparation path

`ContextBuilder.build_messages` in `context.py` accepts explicit instruction, history
and current-turn message lists and deep-copies the assembled view. `ContextState.messages()`
selects raw history or summary + retained suffix before using that builder. The builder
performs no storage access or model calls; `compact_context()` generates summaries.

`Agent` appends all events to raw `TurnResult.messages`; the prepared list is used only for
generation. The existing raw layout and public `run_turn` signature are preserved, so
session saving, evaluation and trace-prefix reconstruction continue to work. Session loading
and its error handling remain in the REPL. Token measurement and fit enforcement are described below.

Item 1 checkpoint: **170 deterministic tests passed** in `transformer-practice` using
`python -m unittest discover -s agent_from_scratch/tests`. The three new regressions check
nested copy isolation, independent per-request inputs through feedback/tool execution and
session replay, and isolation from a backend mutating its input. No real-model benchmark
was rerun for this structural change.

## Implemented: instruction layers in the same module

`context.py` also owns `InstructionConfig`, bounded file reading and `load_instructions`.
The loading function runs at turn start; the pure message builder still runs before each
generation. There is no separate instruction module or mutable instruction cache.

System rules, explicitly configured user defaults and root `AGENTS.md` are combined into
one system message. Sources are literal UTF-8, with no ancestor search, imports or template
execution. An absent root file is recorded as missing; unconfigured layers are disabled.
Configured-source errors stop before model/tool calls and session writes, while the REPL
can accept another request. Defaults cap sources at 8 KiB and the assembled text at 16 KiB.
These byte caps do not establish prompt-token fit.

Run settings retain configured paths/caps and each loaded source's resolved path, hash
and byte count. Hashes cover the exact included UTF-8 text before source labels (the core
system text alone has outer whitespace stripped). Subsequent turns reload; changes during
a tool exchange leave the active turn's snapshot intact. Session history excludes these
system messages. Actual mid-turn instruction replacement remains future compact work.

Item 2 checkpoint: **178 deterministic tests passed**, including literal loading, source selection,
UTF-8 and combined caps, symlink/error handling, per-turn snapshots, provenance and REPL
recovery. Implementation details and acceptance checks are in the
[item 2 plan](STAGE3A_ITEM2_PLAN.md). The system prompt was revised; historical behavioral
scores do not measure this revision.

Local Qwen2.5-7B Q4_K_M diagnostic: the model used the workspace's `project_check`
command and returned the current request's two-line report instead of the saved one-word
preference. The clarified case took two model requests / 4.95 seconds with temperature 0,
8,000-token context and 256 output tokens. Evidence is retained under
`outputs/stage3a-instructions-clarified-20260916-8xy9ea9k/` (script, configuration, source
hashes, raw trace and observations). The initial case in
`outputs/stage3a-instructions-20260916-ny9oz72_/` also executed the check and used two lines,
but failed the exact-text scorer on a period included ambiguously in its user prompt.
Only that diagnostic wording was clarified for the second run; both results are retained.
This is an adaptive usability check, not a benchmark or general compliance guarantee.

## Implemented: next-request token measurement

`LLM.measure_context(messages, tools)` reuses the installed Qwen formatter and backend
tokenizer, including the same special-token flags as generation. This counts the fully
rendered request: instructions, schemas, history, tool results, feedback and role markers.
It does not generate text or change the KV cache. Keeping this in `llm.py` puts model-specific
formatting and tokenization together; `context.py` still only loads and assembles messages.

Immediately before every generation, `Agent` measures the prepared messages with the same
schemas it passes to `generate`. `ModelRequest.context` retains the result even if generation
fails. The optional additive field keeps run trace schema 3; older traces lack it, so readers
should use `.get("context")`. Backend-reported `usage` remains separate and unchanged.

| Field | Meaning |
|---|---|
| `count_method` | `exact` for the installed project formatter; `unavailable` for unsupported/replaced handlers |
| `prompt_tokens` | Tokens in this next formatted input; null when unavailable |
| `window_tokens` | Effective backend `n_ctx()`, which can differ from the requested size |
| `response_reserve` | Configured positive `max_tokens`, not actual output usage; null if unbounded |
| `remaining_tokens` | Window minus prompt minus reserve, before any safety margin; null if either count is unknown |

Measurement itself records negative remaining room without truncation or compaction;
item 4's fit check below decides whether to generate. There is no new CLI command, component breakdown or
fallback estimator. Previous input/output totals cannot count a changed request, and cache
reuse does not increase window capacity. The tradeoff is formatting/tokenizing once to
measure and again to generate; avoid replacing the generation API in this small step.

Item 3 checkpoint: **183 deterministic tests passed** in `transformer-practice`. New checks compare
measurement against the real llama.cpp chat handler with a scripted tokenizer, cover
Unicode/schemas/tool results/feedback, effective window size, unavailable counts and negative
room, and verify measurement survives a backend failure in the saved trace.

Local Qwen2.5-7B Q4_K_M diagnostic at temperature 0 and 64 output tokens: a calculator turn
finished with `4`; its two requests measured/reported **535/535** and **652/652** prompt tokens.
Three further requests (Unicode without tools, all default schemas, and tool-result history
with literal markers) matched at **21**, **1,776**, and **318** tokens. The backend's effective
window was **8,192** for requested `n_ctx=8000`. A deliberately oversized input measured
**12,030** tokens and **−3,902** remaining; it was not sent for generation. This checks counting
parity and visibility, not overflow recovery or general agent capability.

Evidence: `outputs/stage3a-token-count-20260917-g4zi2pyf/` contains `smoke.py`, `result.json`,
`run.log` and the saved turn trace, including configuration, source hashes and comparisons.

## Implemented: request fit enforcement

Immediately after measuring, `Agent._measure_context_limit` marks a request blocked when
`remaining_tokens < context_margin_tokens`, or when exact measurement / a bounded output
reserve is unavailable. `_run_turn` then returns before `generate`. Marking a status alone
does not exit the loop: review caught and fixed that missing return at the caller.
`AgentLimits` defaults the margin to 256 tokens; like its other fields, values are caller-configured
without constructor validation.
Equality is allowed. The margin is recorded in run settings; measurement still reports
remaining room **before** subtracting the margin.

The request gets status `blocked`, while the turn stops with `context_limit`. Unknown budget
uses error code `context_unavailable`; known overflow uses `context_limit`. No backend usage,
raw response or finish reason is invented. Evaluators exclude blocked entries from model-call
and usage accounting. Behavioral metrics report zero usage when no calls ran, while missing
usage on an actual request remains unknown. Raw trace entries remain available for inspection.

After tools or parser feedback, the next iteration rebuilds and measures again. No result
is sliced to force fit. A final answer ends normally without reserving another response.
The existing tool caps remain in force; the gate does not predict result size or undo tool
effects. A stopped turn retains all exchanges in its run trace, but the existing session
policy excludes its messages from replay. The REPL explains this without adding a command.

Validation: **191 deterministic tests pass**. New regressions cover exact fit and one-token
overflow, unknown budgets, stops after a file write and parser feedback,
trace/session preservation, final answers, REPL reporting and evaluator accounting. Scripted
backends have explicit fitting-budget fixtures; they do not claim real tokenizer accuracy.

Local Qwen2.5-7B Q4_K_M diagnostic, temperature 0, output reserve 256, margin 256:

| Effective window | Read cap (characters) | Next prompt after tool | Result |
|---|---|---|---|
| 2,048 | 256 | 1,167 tokens | Two generations; final answer `READ` |
| 2,048 | 16,000 | 7,145 tokens | One generation; next request blocked |
| 4,096 | 16,000 | 7,144 tokens | One generation; next request blocked |

Each read was explicitly truncated with a continuation cursor; `READ` acknowledges the
returned preview, not full-file coverage. Serialization/metadata also consume tokens.
An oversized initial prompt measured 5,302 tokens in the 2,048-token window and made zero
model calls. Every generated request matched backend-reported prompt usage; saved raw
results survived all blocked continuations. These are targeted diagnostics, not a capability
benchmark or evidence that 256 is an optimal margin. Default read caps were not changed.

Evidence: `outputs/stage3a-context-fit-20260917-bwg1rwz4/` contains the reproducible `smoke.py`,
fixture, `result.json`, run traces, source hashes, diagnostic log and before/after test logs.
Compaction and its own budget are the next Stage 3B work.

## Keeping useful tool evidence in a small context

The book puts tool-result budgets and context preparation before generation
([Chapter 3, §3.3](../../../harness-books/book1-claude-code/chapter-03-query-loop-heartbeat.md)).
Its compact goal is a context that can continue the work, including corrections and current
constraints ([Chapter 5, §5.6–5.7](../../../harness-books/book1-claude-code/chapter-05-context-memory-compact.md));
[Appendix A.4](../../../harness-books/book1-claude-code/appendix-a-checklists.md) supplies the checks.

Use three mechanisms, in this order:

1. **Select and extract.** List/search before reading; return relevant bounded ranges.
   Extract page body text while retaining headings, table labels, units and source locations.
   A preview alone cannot establish whole-document coverage.
2. **Store and retrieve.** Keep large available outputs as artifacts, and show status,
   source/version, observed ranges, a bounded preview and a retrievable reference in context.
   Read details on demand through the existing workspace policy. Nanobot's
   [maybe_persist_tool_result](../../../nanobot/nanobot/utils/helpers.py) implements this pattern;
   its [context governance](../../../nanobot/nanobot/agent/context_governance.py) exempts
   `read_file` to avoid a persist → read → persist loop.
3. **Summarize when the budget requires it.** A runtime-controlled, bounded summary request
   is sufficient; no third-party skill system is needed. Preserve goal, constraints, confirmed
   observations, errors/corrections, pending work and evidence references. Keep error codes,
   call IDs, coverage and permissions as host-owned data, not facts the summarizer may rewrite.

Summarization is lossy. Source references allow checks and rereads; they do not guarantee that
every relevant fact survives. Test late-file facts, negation, numeric values/units, exceptions,
recent corrections and successful continuation after compact. Summary calls also consume
tokens and need their own input/output limits. Selected text still consumes context tokens;
a filename, embedding or cache does not give the actor unseen semantic content for free.

The normal `read_file` now has line/character cursors, versions and a 16,000-character
window, plus PDF/Office text extraction; no read deduplication is added before context
ownership exists. `web_fetch` extracts HTML and caps content, but has no continuation cursor
or stored full-page artifact. Content discarded or never downloaded cannot be recovered
from a preview. Frozen Stage 2B uses legacy byte reads (1 KiB) and fixed web outputs.
Stage 3 adds actual token budgeting, context assembly and compact;
artifact retrieval should remain a small extension of existing tools, not a new retrieval stack.
Tool content caps exclude metadata, JSON escaping, schemas and role markers. Search returns
multiple bounded records; even one tool response can exceed the remaining prompt budget.

## State: distinct lifetimes, explicit owners

`ToolResult`, `ModelRequest` and `TurnResult` describe different scopes. A completed model
request and a failed tool execution can both be true. Keep those records and scoped error codes.
The book's assembled loop state concerns the information needed to continue the next iteration;
it does not require merging every result into one generic object.

For Stage 3, introduce only the live state needed by context construction:

| State | Owner and purpose |
|---|---|
| Existing result/request records | Tool/model/turn evidence, linked by call ID and run ID |
| Agent local variables | Iteration, tool attempts, repeated failures and call IDs remain local; no `LoopState` class |
| `ContextState` | Owns raw messages, summary, covered and last-sent boundaries, summary-call count and attempted boundary; builds independent inputs |
| Versioned summary checkpoint | Session store persists summary + raw boundary + source digest/configuration for restart |

The separation is in place: **raw transcript versus an independent model-facing view**.
Stage 3B item 1 adds `ContextState`: summary, covered raw boundary and last actor-sent raw
boundary. `compact_context()` tracks bounded attempts in that state and builds a candidate
without editing raw messages, publishing it only after a smaller, fitting prompt is measured.
Run schema 4 records actual
inputs/schemas and purpose (`agent`/`compact`); the old raw-prefix
interpretation remains valid only for older records without `input_messages`.
Session saving still slices the **raw** list using history length, never a compacted view.
`measure` in `evals/foundation.py` and `exchanges` in `evals/verify.py` still consume unchanged
raw evidence; their request/usage accounting includes summary calls. Explicit persisted
deltas/checkpoints remain Stage 3C. See [implementation and evidence](context-memory.md).

The runtime now stores `LLMResponse.tool_calls: list[ToolCall]` and
`ModelRequest.call_ids` (run trace schema 3). A model request can yield several ordered
actions; each has its own `ToolResult`, including explicit `skipped` results when a batch
stops. There is no parallel execution or rollback of completed calls.

Keep each assistant batch with all its call results (including skipped/interrupted results),
retain observations and runtime parser feedback the actor has not seen, and publish
a new checkpoint only after its source raw messages are saved and the rebuilt prompt fits.
Failed compact leaves the old checkpoint intact. Summarized file state is historical evidence;
reread before relying on it as current. Nanobot likewise separates
[context compaction state](../../../nanobot/nanobot/agent/context_governance.py) from its
[session summary checkpoint](../../../nanobot/nanobot/session/summary.py).

Failed/interrupted turns may already have changed files even though their messages are
excluded from session replay. Checkpoint fallback must not imply rollback or tool replay.
One existing budget caveat: `AgentLimits.max_tool_calls_per_response` is recorded in settings,
but the parser enforces `config.MAX_TOOL_CALLS_PER_RESPONSE` (8) directly. The defaults agree;
overriding that field alone does not change parsing. Reconcile this when wiring Stage 3's
budgets so recorded configuration describes the limits actually enforced.

This is enough structure for our synchronous tiny agent. A generic event bus, universal State
class or framework rewrite would add scope without resolving the current coupling.
