# Telegram bot deployment — design

Decided September 22, 2026. Branch `feat/telegram-bot`, cut from `main`. Optional side
experiment tracked in [STAGE.md](../../STAGE.md#stage-8-extra--telegram-bot-deployment--optional-outside-the-sprint-gate):
not a Stage 8 gate item, not part of the Stage 9–11 training line.

## Goal

Run the existing agent as a long-lived process reachable from a phone over Telegram, so the
loop/tools/memory mechanisms can be exercised outside a terminal session, any time. No new
agent capability; this only adds a way to reach the agent that already exists.

## Constraints that shaped this

- The primary dev laptop cannot stay powered on 24/7 (it travels, gets shut down), so the
  process must live somewhere else.
- A Mac mini M4 (16 GB RAM, ~50 GB free disk) is available and can stay on continuously. That
  is enough headroom for the local Qwen2.5-7B q4 GGUF model plus `llama-cpp-python`'s Metal
  offload, so the backend does not need to change.
- `Agent.run_turn` (`agent.py:74`) is already decoupled from the REPL: it takes `user_input` +
  optional `history`/`checkpoint` and returns a `TurnResult`, with session persistence handled
  by `SessionStore`. `run_repl` (`agent.py:390`) is the reference for exactly how to drive it
  (load history/checkpoint, call `run_turn`, handle `/new` `/reset` `/session` via the existing
  `_session_command`, print the result by `RunStopReason`).
- `LLM.parse_response` already expects an OpenAI-shaped chat-completion dict, and `tools/web.py`
  already depends on `httpx` — no new dependency is needed for this feature.
- The existing file/shell tools are unrestricted (STAGE.md: "General shell uses local user
  permissions; cwd/file-tool bounds do not sandbox it"). Anyone who can message the bot can run
  arbitrary commands on the Mac mini. This is the dominant design constraint, not a footnote.

## Rejected approaches

- **Cloud API backend** instead of the local model: needs a new `LLM`-compatible backend
  (token counting, `measure_context`, response shape) — real work, and it exists as a separate,
  more careful goal in Stage 9 for the trainable checkpoint. Unnecessary now that an always-on
  local host is available.
- **Telegram webhook + HTTP server**: needs a public HTTPS endpoint (port forwarding, TLS cert,
  or a tunnel process), which is another always-on component to keep alive and another failure
  surface. Long polling needs only outbound HTTPS.
- **A bot framework (python-telegram-bot, aiogram)**: pulls in an async event loop for a
  single-user, single-conversation-at-a-time bot. STAGE.md already excludes asynchronous/
  concurrent calls from this project's scope, and the agent loop itself is synchronous.

## Architecture

One new module, `bots/telegram_bot.py` (~150–250 lines). No changes to `agent.py`, `llm.py`,
`session.py`, `context.py`, `compact.py`, or any `tools/*` module.

```
Telegram (long poll, httpx) → allowlist check → command dispatch (/new /reset /compact /session)
    → SessionStore.load_history/load_checkpoint → Agent.run_turn → format reply by RunStopReason
    → sendMessage
```

- One `LLM` + `Agent` instance is constructed once at process start and reused for every
  message; the model loads once and stays resident (unlike a fresh CLI invocation).
- `chat_id` (as a string) is the `session_id` passed to `run_turn`/`SessionStore`, giving each
  Telegram chat its own persisted conversation for free.
- `/new`, `/reset`, `/compact`, `/session <id>` are intercepted before `run_turn` and handled
  with the same `Agent._session_command` logic `run_repl` uses; no duplicated command logic.
- Telegram's `getUpdates` `offset` parameter drives the poll loop; each processed update's ID
  becomes the next request's offset floor, so a restart does not replay already-handled
  messages.
- Message processing is strictly sequential (one update at a time, one `run_turn` at a time),
  matching the existing synchronous loop and the project's stated exclusion of concurrency.
- `run_turn` is called without an `on_progress` callback: a long tool-using turn sends nothing
  until the turn ends, then one reply. Streaming intermediate progress messages to Telegram is
  explicitly deferred — STAGE.md already excludes streaming from this project's scope, and one
  reply per turn is the simplest thing that works for a single operator.

## Security

- A single allowlisted Telegram numeric user ID (the operator's own) is checked on every
  incoming update before anything else runs. Any other sender is dropped (logged, not
  executed, no reply). This is the only access control in front of unrestricted shell/file
  tools, so it is mandatory, not configurable-off.
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USER_ID` are read from the environment, never
  hardcoded or committed. The launchd plist injects them (or points at an untracked env file).
- No change to the existing tool permission model. Tightening file/shell tool scope is out of
  scope for this change — noted as a follow-up if the bot ever needs to be reachable by anyone
  other than the operator.

## Error handling

- An exception raised while handling one update (including inside `run_turn`) is caught,
  logged, and answered with a short "internal error" reply; the poll loop continues to the
  next update. One bad turn must not end 24/7 availability.
- A network error calling the Telegram API is retried with backoff inside the loop, rather than
  exiting the process. launchd's `KeepAlive` is the last resort, not the primary recovery path.
- Every non-`final_response` `RunStopReason` (`max_iterations`, `model_error`, `tool_limit`,
  `no_progress`, `context_limit`, `interrupted`) gets a distinct, short human-readable reply
  instead of a raw dump, so the operator can tell from their phone what happened.
- `run_turn` raising `InstructionLoadError` (bad/missing rule files) is caught the same way
  `run_repl` catches it: reply that instructions are unavailable, keep the session/turn as not
  started, continue the loop.

## Deployment / process supervision

- Runs on the Mac mini via `launchd`: a plist with `RunAtLoad` and `KeepAlive` under
  `~/Library/LaunchAgents/`, stdout/stderr redirected to a log file for later inspection.
- No inbound network exposure: long polling is outbound-only HTTPS to Telegram's servers, so no
  port forwarding, dynamic DNS, or certificate is needed.
- The GGUF weights and `llama-cpp-python` need to be present on the Mac mini (weights copied
  over once, ~4.7 GB); this is a one-time setup step, not part of the code change.

## Testing

- Unit tests (mocking the Telegram HTTP calls, no model/network) for: allowlist filtering,
  command parsing/dispatch, `chat_id` → `session_id` mapping, and reply formatting per
  `RunStopReason`.
- Manual smoke test against a real Telegram bot token in a throwaway chat: round-trip a
  message, confirm `/new` starts a fresh session, confirm a non-allowlisted sender is ignored.
- No changes to the existing deterministic test suite; this is purely additive.

## Out of scope

- Multi-user support, concurrent conversations, streaming replies.
- Any change to tool permissions/sandboxing.
- Cloud/API backend swap (that's a separate, Stage-9-scoped concern for the trainable
  checkpoint, not this bot).
- Rich Telegram features (inline keyboards, file uploads/downloads, voice) — plain text only.
