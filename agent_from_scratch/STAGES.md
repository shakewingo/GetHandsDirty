# Project: Two-week agent learning sprint

Curriculum: [REPORT_ANALYSIS.md](REPORT_ANALYSIS.md). This file tracks how curriculum are actually being implemented and growing over time.

**Goal: experience and explain as much of the curriculum as possible in two weeks,
not build a overcomplex production harness.** Keep one project; move on after a small working
example, a deliberately broken case, and a short explanation. Use AI to help with some harnessing and test;
Use myself to think the design and implement for core components/mechanism.

Stage 0.5 records the current baseline; it is not an extra curriculum milestone.
Suggested remaining timeboxes: Stage 1 one day, Stage 2 one day, Stage 3 one day,
Stage 4 one day, Stage 5 half a day, then reserve the remaining time for Stage 6
and small Stage 7–8 exercises. Count time already spent toward the two weeks.
If time gets tight, shrink later exercises to a runnable example and an honest
limitation note; don't claim complete training or RL experiments from a preview.

# Coding Rules for AI
- Keep codebase structured and styling following current project conventions. Keep the code light and readable, avoid over-engineering.
- Use the MOST MINIMAL but SIGNIFICANT implementation. DO NOT over-engineer or over-abstract things.

# Stage 0.5: Current minimal agent

By Sep 11, 2025

## Completion criteria

Explain and demonstrate one model → calculator → model round trip, including
where messages are appended and why the loop stops. Tool/parser tests pass.
This is a working prototype, not yet verified bounded error recovery.

Status: code and earlier user-run examples demonstrate the round trip. All 15
current tool/parser tests pass on review; no real-model run was repeated for this
documentation update. Sequential-call recovery remains a Stage 1 check.

## Included features and code

| Feature | Current implementation | Limitation / next step |
|---|---|---|
| Local inference | `llm.py`: `LLM` loads local Qwen through llama-cpp-python; `generate()` supplies messages and tool schemas. | Model loaded once per `LLM` instance, not every inference. `max_tokens=512` bounds each generation, not the whole turn. |
| Response parsing | `llm.py`: `parse_response()`, `LLMResponse`, `ResponseErrorCode`; `utils.py`: Qwen tag/JSON fallback. | Accepts one native or Qwen-tag call; rejects multiple calls, malformed/empty/truncated output. Raised errors currently escape `run_turn()`. Not a general provider adapter. |
| Tool contract | `tools/base.py`: `Tool.invoke()` validates arguments, executes, returns `ToolResult` with centralized `ToolErrorCode` messages. | ~~Supports a local subset of JSON Schema, not the full standard. No need to extend it now.~~|
| Tool dispatch | `tools/register.py`: `ToolRegistry` exposes schemas, checks names, rejects duplicate registration, dispatches calls. | ~~~~Only calculator registered; sufficient for Stage 1.~~ |
| Calculator | `tools/calculator.py`: add/subtract/multiply/divide with two numeric operands. | ~~Does not evaluate expression strings. Description currently suggests otherwise; clarify it, don't build an expression engine for Stage 1.~~ |
| Within-turn continuation | `agent.py`: `run_turn()` appends assistant content and tool success/error messages before the next model request. | Inner loop retries the same failed arguments up to twice, without model correction between attempts. Native `tool_calls` are not preserved in assistant messages (for the purpose of observability); Qwen IDs default to empty. |
| Bounded iteration | `agent.py`: `max_iterations=10`; direct response breaks the loop. | ~~`max_tool_calls=2` is a per-response retry count, not a turn-wide budget. Neither setting interrupts a blocked call.~~ |
| CLI and observation | `run_repl()` accepts requests; `TurnResult` holds messages; Loguru prints execution and final messages. | Every request starts fresh. No explicit completion status; printing the last message may mislabel a tool observation as an answer. Trace is in memory/logs only. |
| Tests | `tests/test_tools.py`, `tests/test_response.py`: 15 unit tests, no model weights required. | No loop-level tests after rollback. Token fields exist on `LLMResponse`, but only `usage` is populated by the parser; no turn totals. |

## Out of scope → where it belongs

| Deferred topic | Best-fit stage | Small learning exercise, not a prerequisite now |
|---|---|---|
| Async tool execution | Stage 2 for timeout/cancellation concepts; Stage 4 for parallel independent tools | First understand stopping a run versus stopping an operation. An async rewrite is optional; wrapping synchronous inference in `async` does not make it cancellable. |
| Skills | Stage 4: compare orchestration approaches | Load one reusable instruction file for a task and compare with the baseline prompt. No discovery framework or skill marketplace. |
| Memory | Stage 2: session history; Stage 4: retrieval/long-term memory | Optionally keep messages across two REPL requests. Later compare retrieving one saved fact. Saving a trace is not automatically model memory. |
| Context compaction | Stage 2 if retained history hits context limits; otherwise Stage 4 experiment | Summarize old context once and check whether an important fact survives. Preserve the system instruction, current task, and complete tool-call/result pairs. No compaction engine. |

None of these four topics blocks Stage 1. Durable restart/resume and production
cancellation can wait until Stage 5 or beyond this sprint if no task needs them.

# Stage 1: Close the basic loop — one day only

## Completion criteria

A scripted model demonstrates **two dependent tool calls → final answer**,
**bad arguments → correction**, **malformed output → correction**, and
**repeated failures → iteration-limit stop**. A direct answer exits immediately.
The REPL never presents the last tool observation as a final answer.
Try one local-model multi-step prompt and keep its outcome, successful or not.
Model reliability across many prompts belongs to Stage 3, not this gate.

## Must-have changes only

1. **One execution per model request** — remove the inner retry loop in
   `agent.py`. Append exactly one success/failure observation, then ask the model
   again. Keep `max_iterations` as the only turn budget; remove the misleading
   `max_tool_calls` setting. At most one tool executes per iteration.
2. **Let the model correct parsing errors** — catch the existing `ResponseError`
   around generation, append concise runtime feedback, and continue within the
   same iteration budget. Don't invent a tool result for an unparsed call.
   No extra retry counter, JSON repair engine, or new error taxonomy.
3. **Preserve the tool request** — a small message-building helper/branch retains
   native assistant `tool_calls` and matching IDs. Give missing IDs a simple
   per-turn ID. Represent Qwen calls consistently with their observations;
   no provider framework and no multi-call support needed.
4. **Distinguish answer from exhaustion** — add only `final_answer` and a simple
   `stop_reason` to `TurnResult` (`final_response` / `max_iterations`). Set the
   answer only for a direct response; let `run_repl()` print an explicit stop
   notice otherwise. Backend exceptions are a Stage 2 exercise.
5. **Prove the loop cheaply** — add a short `tests/test_turn.py` with the scripted
   cases above; check that the next model call sees each result once and that
   calls stop at the configured limit. Clarify calculator's description as
   “one operation on two numbers”; keep its existing schema.

## One-day work order and handoff

- Morning: items 1–2; explain why repeating identical arguments isn't recovery.
- Afternoon: items 3–5 and one real-model smoke run. Save a short transcript/note.
- Stop after these checks. No metrics subsystem, total-token accounting, expression
  parser, new tools, new dependencies, or broad refactor.

Use the Python companion's lesson 06 for the loop concept, not as code to copy
wholesale. Stage 1 hands a small callable `run_turn()` and known failure cases to
Stage 2; these same cases become evaluation fixtures in Stage 3.

# Stage 2: Observe and contain failures — one day target

## Completion criteria

Save and inspect one successful run and one injected backend-failure run.
The backend failure produces a clear error outcome without an automatic retry,
and the REPL can accept another request. A saved record contains the task,
messages, outcome, elapsed time, and inference settings. Explain which operations
the iteration limit cannot stop. No database, async rewrite, or resume engine required.

## Small core, building on Stage 1

1. **Backend failure boundary** — around `LLM.generate()`, handle ordinary backend
   exceptions separately from recoverable `ResponseError`; expose `model_error`
   plus a short error message. Add one injected-error test. Do not swallow
   `KeyboardInterrupt` with a broad `BaseException` handler.
2. **One saved run record** — serialize the existing `TurnResult` plus input,
   elapsed time, model identifier, temperature, `max_tokens`, and iteration limit
   to JSON/JSONL outside the core loop. Record error feedback too. Use harmless
   prompts; no raw-response archive, event bus, telemetry service, or token accumulator.
3. **Small failure lab** — rerun Stage 1's malformed-output, bad-arguments, and
   limit cases; add the backend exception. Inspect what was executed versus what
   the model merely requested. Note that calculator retries are harmless, but
   future file writes may not be safe to repeat.

## Optional taste — choose at most one if the core is done

- **Timeout/cancellation:** one standalone subprocess timeout demonstration;
  distinguish terminating that process from merely stopping the wait for a thread.
  Don't redesign local inference to accommodate it.
- **Session memory:** keep history for two REPL requests and provide an explicit
  reset. Compare with today's fresh-turn behavior; long-term retrieval waits.

If these exceed the day, record the boundary and move on. Before introducing
external or side-effecting tools, scope their workspace/permissions and timeout
behavior to that actual tool; the calculator prototype is not a sandbox.

## Connection to later stages

- **Stage 3 — measure:** turn the saved tasks into a tiny development/held-out suite;
  check answers or artifacts, not just “model returned text.” Report success and
  elapsed time; add token accounting only if needed for a comparison.
- **Stage 4 — compare:** try one planning, verification, memory, retrieval, or skill
  change on the same tasks. Don't implement the whole architecture catalogue.
- **Stage 5 — deliver:** repeatable setup and a CLI demo; add isolation or persistence
  only where the chosen task requires it.
- **Stage 6–8 — learn training and experiments:** retain verified successful and
  failed runs as candidate examples, separate training from held-out evaluation,
  then attempt small SFT/reward/experiment-loop exercises. Saved transcripts need
  validation before becoming training data; a smoke test is not evidence of gains.

Reference selectively: companion lessons 07 (memory), 09 (action boundaries),
11 (evals), and 12 (telemetry). Curriculum order here follows the project's needs,
not a requirement to finish every tutorial feature.

# My Scratch Notes
1. [DONE] Your parser also deserves hardening:
- ~~Prefer native message.tool_calls when the backend supplies them.~~
- ~~Keep decode_qwen_tool_call() as a provider-specific compatibility fallback.~~
- ~~Do not classify calls merely because the content contains the substring "tool_call".~~
- ~~Convert malformed output into a protocol error instead of allowing .index() or JSON parsing to terminate the process.~~

[OPTIONAL]: Design the internal representation as tool_calls: list[ToolCall], even if you execute only one sequentially for now. That gives you future power without introducing concurrency yet.
2. Tool vs Skill:
- Tool is a simple execution with certain schema, while skill contains multiple tools/more complex logic and require a mechanism for model to know the summary of existing skills as well as how to execute it.

# My Takeaways
1. Download only required quantized model files for llama-cpp to load:
`hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q4_k_m-*.gguf"` (~4.68GB)
2. SOLVE PROBLEM EACH PER TIME, FIND THE REAL ONE! You just need to find that one/few gap to solve the problem!
3. llama.cpp: `LLM.__init__()` loads weights once per instance; `generate()` reuses it.
   Prefix-cache reuse is separate from model loading.
- ~~how to silence setup logging for each inference?~~
- ~~seems the module has prefix-matching, wondering what content it is and how it is calculated?~~
```
</tool_call>
User: what did we just talked about?
Llama.generate: 272 prefix-match hit, remaining 11 prompt tokens to eval
```

# TODO
1. [Optional Stage 4 tool-design comparison] Enhance calculator to accept an expression
   instead of `left` / `right`. Compare argument failures with the current binary tool;
   don't assume it is automatically more robust. Use a restricted arithmetic parser,
   never unrestricted `eval`. Keep the binary tool for Stage 1's multi-step exercise.
