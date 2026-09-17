# Stage 3A, item 2: instruction loading and composition

Status: implemented, September 16, 2026; **178 deterministic tests pass**.
Adjusted to keep loading and assembly in `context.py`, as requested. The steps below
remain a guide to the implementation. Scope comes from [STAGE.md](STAGE.md).
The real-model diagnostic and its retained initial punctuation mismatch are recorded in
[the context design notes](CONTEXT_STATE_DESIGN.md).

**Outcome:** each turn uses a bounded, identifiable snapshot of system rules,
explicit user defaults and workspace rules. Every generation in that turn receives
the same instruction snapshot through the existing context builder.

Item 1 already separates raw messages from prepared inputs. Build on that boundary.
Token counting, compaction, memory storage and automatic rule editing were outside this item.
Stage 3A measurement and fit enforcement are now complete; see STAGE.md for current status.
New CLI commands remain deferred. The existing `memory.py` placeholder has no role in this item.

## Design decisions to implement

| Decision | Implementation | Purpose and benefit |
|---|---|---|
| Known sources | Package `prompts/system.md`, one configured user file, and root `AGENTS.md` | Makes instruction discovery predictable |
| Explicit workspace | Pass a resolved workspace root; never infer it from process cwd or state storage | Keeps project rules aligned with the tool workspace |
| Fixed composition | System → user defaults → workspace rules, with labeled blocks | Makes the assembled instructions inspectable |
| One system message | Combine the layers into the existing first message | Preserves current history boundaries and session/eval assumptions |
| Stable turn snapshot | Load once at turn start; rebuild context from that snapshot each iteration | Tool execution cannot change the active instructions halfway through a turn |
| Independent source records | Retain source kind, path, included text, hash and byte count | Explains which instruction versions affected a run |
| Bounded input | Enforce per-source and assembled UTF-8 byte limits | Prevents unexpectedly large instruction files from being silently ingested |
| Explicit failures | Stop before generation when a required/configured source cannot be loaded | Avoids running with silently missing or incomplete instructions |

The composition order is not an automatic conflict-resolution algorithm. The core
prompt must say that additional rules operate within the runtime contract, and that
current explicit user instructions override saved preferences. Workspace conventions
guide project work where the current request leaves a choice. Runtime code continues
to enforce tool schemas, execution policies and budgets.

For example, a saved preference for short answers yields to a current request for a
detailed explanation. A workspace instruction requesting unlimited calls cannot
change the parser or loop limits. Prompt wording guides the model; it does not prove
compliance or create an isolation boundary.

## Small data and function contracts

Keep instruction loading and composition in `context.py`, alongside the existing builder.
Use separate functions within that module so `ContextBuilder.build_messages()` remains
free of storage access. No separate instruction module is needed at this scale.

Use ordinary functions, one frozen configuration dataclass, and plain source metadata records:

```python
InstructionConfig:
    workspace: Path | None = None
    user_path: Path | None = None
    max_source_bytes: int = 8192
    max_total_bytes: int = 16384

Source metadata (dict):
    kind: str             # system / user / workspace
    path: str | None      # resolved source path when configured
    status: str           # loaded / missing / disabled
    sha256: str           # loaded sources: hash of exact included UTF-8 text
    byte_count: int       # loaded sources: length of exact included UTF-8 text
```

These are proposed initial byte caps, not measured token-window guarantees.
Validate that caps are positive. The assembled cap includes labels and separators.

Suggested function responsibilities:

- `_read_instruction(...)`: read one bounded UTF-8 file.
- `load_instructions(config)`: load in fixed order, assemble one string, and return
  that string with JSON-serializable source metadata. Keep these operations together
  while the implementation is small; no extra source or snapshot classes.
- `InstructionLoadError`: distinguish configuration/loading failures from model failures.

Keep the system source fixed to the package's `prompts/system.md`. All three files are
plain Markdown. The current system prompt contains no template expressions; preserve
its existing outer-whitespace stripping when replacing its `render_prompt` call.
Load external files literally, without Jinja execution, imports or include processing.
Hash the exact included text after any documented normalization; use the same text for
assembly. A hash identifies content, not its authority or correctness.

## Implementation sequence

### 1. Organize the core prompt

Edit `prompts/system.md` into Purpose, Behavior, Tools and Reporting sections.
Keep the existing tool protocol, recovery, no-op, JSON and answer-format rules.
Add the source/conflict policy above without introducing unrelated behavioral changes.

**Self-check:** account for every original rule. Explain which section owns it and why.
Record this as a prompt revision; even reorganization may change model behavior.

### 2. Implement bounded reading and source selection

Resolve configured paths once when configuring the agent. Expand an explicitly supplied
home-relative user path; resolve other relative paths at configuration time. A configured
workspace must exist and be a directory. Its only discovered source is `AGENTS.md` directly
inside that root. Reject a workspace rule symlink resolving outside the configured root.
The explicitly configured user file may live outside the workspace.

Read at most `max_source_bytes + 1` bytes before UTF-8 decoding; reject oversized input.
Require a regular file. Do not use a file-size check followed by an unlimited read.

| Source condition | Behavior |
|---|---|
| System file missing | Error |
| User path not configured | Omit user layer |
| Explicit user path missing | Error |
| Workspace root not configured | Omit workspace layer |
| Root `AGENTS.md` absent | Omit workspace layer |
| Present source unreadable, non-regular, invalid UTF-8 or oversized | Error |
| Empty optional file | Accept an empty source and record its identity |
| Empty system file | Error |

Only the genuinely absent optional workspace file is skipped. A broken symlink or
permission error must not be mistaken for an absent source. Do not search ancestors,
nested directories, user home defaults or paths mentioned inside a rule file.

**Self-check:** construct source records from temporary files before integrating the agent.

### 3. Assemble a labeled instruction snapshot

Render the system text first, then labeled user and workspace blocks when present.
Keep resolved paths and hashes in metadata rather than spending prompt space on them.
Reject an oversized final string; never silently truncate a rule or omit a loaded layer.

Return one string for one `system` message. This preserves the current
`1 + history_length` boundary. The context builder still deep-copies instructions,
completed history and current-turn messages before every generation.

**Self-check:** compare the assembled text against a small explicit expected example,
including ordering, separators and total encoded size.

### 4. Wire configuration and turn-start loading

Add an optional keyword-only `instruction_config` argument to `Agent.__init__`.
Keep `run_turn(user_input, history=None, *, session_id=None, ...)` compatible.
Library construction defaults to system-only instructions; external sources are opt-in.

The default REPL entry point passes the workspace already used by its default registry.
`examples/tools_demo.py` passes its `--workspace` and adds an optional
`--user-instructions` path argument. Use this existing demo for exercising the new feature.
Custom registries must pass their matching root explicitly; do not guess it from tools.
Existing frozen evaluators continue with system-only configuration.

At turn start, load sources and assemble the instruction string before constructing
`TurnResult`. Replace only the current direct `render_prompt("system.md")` initialization.
Reuse this snapshot for all model iterations. Do not call the loader from inside the loop.

Library loading failures raise `InstructionLoadError`. The REPL catches that error,
prints the source and reason, and continues accepting input. No model/tool call or session
append occurs. Treat this as setup failure, not `model_error`; a failed setup does not
create a normal run trace in this increment.

Compaction will reuse the loader later. Do not add its trigger now; changing instructions
mid-run will also require the later actual-input trace design.

### 5. Attach source metadata to the run

Put JSON-serializable configuration and source records into `TurnResult.settings`:
configured root/user path, caps, composition order/version, and each loaded source's
kind, resolved path, hash and byte count. Paths become strings; do not duplicate source
text in settings because the assembled text is already in `messages[0]`.

Record an absent workspace file explicitly as `missing` in metadata; an unconfigured
layer is `disabled`. This distinguishes intentional omission from failed loading.
No trace/session schema migration is needed for additional settings metadata.

On a new turn, read again and recompute hashes. Existing session history continues to
exclude system messages, so a previous instruction snapshot does not become replay history.

## Files to change

| File | Change |
|---|---|
| `prompts/system.md` | Four sections and explicit layer/conflict policy |
| `context.py` | Configuration, bounded loading, hashing, composition and errors; retain pure builder |
| `agent.py` | Accept configuration, load once, attach metadata, handle REPL setup errors |
| `examples/tools_demo.py` | Pass matching workspace and optional user instruction path |
| `tests/test_context.py` | Extend existing tests with source selection, limits, hashing and assembly |
| `tests/test_turn.py` | Snapshot lifetime, recovery, trace metadata and session compatibility |
| `evals/README.md` | Configuration/demo instructions and historical-score caveat |
| `docs/STAGE.md`, `docs/CONTEXT_STATE_DESIGN.md` | Record verified implementation and remaining work |

Keep `ContextBuilder.build_messages()`, tool execution, session schema and the Qwen
template behavior unchanged. Prefer small additions to existing files throughout.
Update existing prompt mocks where loading moved; retain their behavioral assertions.

## Acceptance checks

1. System-only and three-source cases produce exactly one instruction message in order.
2. Parent/nested `AGENTS.md` and include-like text cause no additional reads.
3. Missing/invalid/oversized cases follow the table, make zero model/tool calls, preserve
   existing session contents, and leave the REPL usable after the source is corrected.
4. Enforce exact byte boundaries with multibyte UTF-8, and reject a combined block that
   exceeds its cap even when every source individually fits.
5. A source edit changes its next-turn hash; hashes match the actual included text.
6. A scripted model changes a source file during a tool exchange: later requests in that
   turn keep the original snapshot, while the following turn loads the edit.
7. Trace metadata and the system message describe the same snapshot. Reloaded session
   history contains conversation messages, not duplicated instruction layers.
8. The existing full suite passes, preserving batches, parser feedback, interruption,
   session commands and frozen evaluator behavior.

Run from the repository root:

```sh
conda run --no-capture-output -n transformer-practice python -m unittest discover -s agent_from_scratch/tests
```

After deterministic checks, run a small real-model diagnostic in disposable files:
user defaults request short answers, workspace rules name a check command, and the
current request asks for detailed reporting. Inspect the actual prompt, observed calls
and final answer. Retain any failure; loading correctness does not guarantee compliance.
This prompt revision does not inherit the historical 12/17 behavioral score.

## Completion and learning checkpoint

Mark only item 2 complete after the loading/assembly contracts, turn integration and
deterministic checks pass. Record the real-model diagnostic separately, including whether
it was run and which requested behaviors it demonstrated. Do not require a positive model
result to report a correct loader, or describe unmeasured compliance as verified.

Implement the bounded reader, failure policy, ordering, hash generation and turn-lifetime
wiring yourself. Be able to explain why a byte cap is not a token budget, why hashes do
not establish trust, why instructions are reloaded between turns, and why the pure context
builder still runs before every generation.
