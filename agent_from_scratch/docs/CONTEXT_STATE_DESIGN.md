# Context semantics and state ownership

September 15 review notes. These are Stage 3 design decisions, not Stage 2B runtime changes.

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

## State: distinct lifetimes, explicit owners

`ToolResult`, `ModelRequest` and `TurnResult` describe different scopes. A completed model
request and a failed tool execution can both be true. Keep those records and scoped error codes.
The book's assembled loop state concerns the information needed to continue the next iteration;
it does not require merging every result into one generic object.

For Stage 3, introduce only the live state needed by context construction:

| State | Owner and purpose |
|---|---|
| Existing result/request records | Tool/model/turn evidence, linked by call ID and run ID |
| Small `LoopState`, when needed | Agent owns iteration, repeated-failure and summary-attempt counters; compact cannot reset them |
| `ContextState` | Context builder owns summary, covered raw boundary, last-sent boundary and prepared view/budget |
| Versioned summary checkpoint | Session store persists summary + raw boundary + source digest/configuration for restart |

The first refactor is **raw transcript versus model-facing input**. Today both use
`TurnResult.messages`; traces reconstruct inputs from `input_message_count`, and session saving
slices the same list using history length. In-place compaction would break both assumptions.
Instead, preserve an explicit raw turn delta and record the actual inputs/schemas for each
model request, including purpose (`agent` or `compact`). Then build a separate context view.

The runtime now stores `LLMResponse.tool_calls: list[ToolCall]` and
`ModelRequest.call_ids` (run trace schema 3). A model request can yield several ordered
actions; each has its own `ToolResult`, including explicit `skipped` results when a batch
stops. There is no parallel execution or rollback of completed calls.

Keep each assistant batch with all its call results, retain observations the actor has not seen, and publish
a new checkpoint only after its source raw messages are saved and the rebuilt prompt fits.
Failed compact leaves the old checkpoint intact. Summarized file state is historical evidence;
reread before relying on it as current. Nanobot likewise separates
[context compaction state](../../../nanobot/nanobot/agent/context_governance.py) from its
[session summary checkpoint](../../../nanobot/nanobot/session/summary.py).

This is enough structure for our synchronous tiny agent. A generic event bus, universal State
class or framework rewrite would add scope without resolving the current coupling.
