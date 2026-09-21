from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from loguru import logger
from .config import AgentLimits
from .llm import LLM, ResponseError, ResponseErrorCode, ResponseType
from .session import SessionStore
from .compact import CompactOutcome, Compactor, compact_prompt_digest
from .context import (ContextState, InstructionConfig, InstructionLoadError,
                      context_blocker, context_fits, load_instructions, window_share)
from .tools.base import ToolErrorCode, ToolRegistry, ToolResult
from .tools.plan import PlanTool
from .tools.register import default_registry, workspace as default_workspace
from .trace import (ModelRequest, ModelRequestStatus, RunStopReason, TraceStore, TurnResult,
                    used_model_calls)
from .utils import color_label

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


def recovery_feedback(error: ResponseError) -> str:
    hints = {
        ResponseErrorCode.INVALID_TOOL_CALL:
            "Use complete tool-call objects with registered names and object arguments.",
        ResponseErrorCode.TOO_MANY_TOOL_CALLS:
            "Issue only the next necessary calls within the batch limit; later calls can follow their results.",
        ResponseErrorCode.TRUNCATED_RESPONSE:
            "The output was cut off. Summarize observed results more concisely; "
            "do not replay long tool outputs. If the requested answer cannot fit, "
            "state what remains undelivered.",
        ResponseErrorCode.EMPTY_RESPONSE:
            "Return the next useful action or a concise answer to the original request.",
        ResponseErrorCode.INVALID_RESPONSE:
            "The response envelope was invalid. Retry the response to the original request.",
        ResponseErrorCode.UNSUPPORTED_FINISH_REASON:
            "Retry using the supported response format.",
    }
    detail = f"{error} " if error.code == ResponseErrorCode.INVALID_TOOL_CALL else ""
    return ("[Runtime feedback] Your response could not be processed. "
            f"No tool executed for this response. {detail}{hints[error.code]}")


REPEATED_CALL = ("[Runtime feedback] You called {tool} with the same arguments {count} times "
                 "and already have this result. Move on to the next concrete step.")
REPEATED_FAILURE = ("[Runtime feedback] You called {tool} with the same arguments {count} times "
                    "and it keeps failing the same way. Repeating it will not work: read the "
                    "error, then try a different call or reconsider the approach.")


class Agent:
    def __init__(self, llm: LLM, state_dir: str | None = None,
                 *, registry: ToolRegistry | None = None, limits: AgentLimits = AgentLimits(),
                 instruction_config: InstructionConfig = InstructionConfig()):
        self.llm = llm
        self.limits = limits
        self.state_dir = state_dir
        self.registry = default_registry if registry is None else registry
        self.instruction_config = instruction_config

    def execute_tool(self, tool_name: Any, tool_params: Any, call_id: str = "") -> ToolResult:
        """Single dispatch point; evaluations override it to enforce read-only runs."""
        return self.registry.invoke(tool_name, tool_params, call_id=call_id)

    def run_turn(self, user_input: str, history: list[ChatCompletionRequestMessage] | None = None,
                 *, session_id: str | None = None,
                 compact: bool = False,
                 checkpoint: dict | None = None,
                 on_progress: Callable[[str], None] | None = None) -> TurnResult:
        """Execute supplied history; compact=True requests the same path used under pressure.

        Args:
            user_input: this turn's request.
            history: the full raw replayable history; a checkpoint never replaces it.
            session_id: associates and saves this run; it does not load history.
            compact: summarize before the first generation, as pressure would.
            checkpoint: a validated `SessionStore` checkpoint whose summary replaces the
                covered prefix in the model-facing view only.
            on_progress: receives assistant narration that accompanies a tool batch.

        Returns:
            TurnResult: raw messages, stop reason, and one record per model request.
        """
        started = monotonic()
        instructions, instruction_metadata = load_instructions(self.instruction_config)
        result = TurnResult(
            messages=[{"role": "system", "content": instructions},
                      *deepcopy(history or []), {"role": "user", "content": user_input}],
            run_id=uuid4().hex, session_id=session_id, input=user_input,
            started_at=datetime.now(timezone.utc).isoformat(),
            settings={**self.llm.settings(), **asdict(self.limits), "instructions": instruction_metadata},
        )
        trace = TraceStore(Path(self.state_dir, "runs") if self.state_dir is not None else None)

        turn_start = 1 + len(history or [])
        state = ContextState(raw=result.messages, turn_start=turn_start, last_sent=turn_start,
                             plan="" if self.limits.planning else None)
        if checkpoint is not None:
            # load_checkpoint already bound this to the supplied history, so covered stays
            # within the turn-start invariant; raw history is replayed unchanged.
            state.covered, state.summary = checkpoint["covered"] + 1, checkpoint["summary"]
            result.settings["checkpoint"] = {key: checkpoint[key] for key in
                                             ("run_id", "covered", "source_sha256")}
        try:
            self._run_turn(result, state, compact=compact, on_progress=on_progress)
        except KeyboardInterrupt as error:
            # A completed model request can still be followed by an interrupted tool.
            if result.model_requests and result.model_requests[-1].status == ModelRequestStatus.STARTED:
                result.model_requests[-1].status = ModelRequestStatus.INTERRUPTED
            self._finish_pending_tools(result,
                "Interrupted. Inspect effects before retrying; remaining batch calls were not executed.",
                interrupted=True, output=getattr(error, "output", None))
            result.stop_reason = RunStopReason.INTERRUPTED
        result.elapsed_seconds = round(monotonic() - started, 2)
        trace.save_run(result)
        self._save_session(result, turn_start - 1, state)
        return result

    @staticmethod
    def _finish_pending_tools(result: TurnResult, detail: str, *, interrupted: bool = False,
                              output: Any = None) -> None:
        """Give every announced call a result, including a stopped batch's remainder."""
        calls = next((calls for m in reversed(result.messages) if (calls := m.get("tool_calls"))), [])
        done = {m["tool_call_id"] for m in result.messages if m["role"] == "tool"}
        for call in calls:
            if call["id"] in done:
                continue
            observation = ToolResult.failure(
                ToolErrorCode.INTERRUPTED if interrupted else ToolErrorCode.SKIPPED,
                tool_name=call["function"]["name"], call_id=call["id"],
                detail=detail, output=output if interrupted else None,
            )
            result.messages.append(observation.to_message())
            interrupted = False

    def _block_on_budget(self, result: TurnResult, request: ModelRequest) -> bool:
        """Record a hard context-budget block on this request for later compact; True when blocked."""
        blocked = context_blocker(request.budget, self.limits.context_margin_tokens)
        if blocked is None:
            return False
        request.status = ModelRequestStatus.BLOCKED
        request.error_code, request.error_message = blocked
        result.stop_reason = RunStopReason.CONTEXT_LIMIT
        result.error_message = blocked[1]
        return True

    def _reload_instructions(self) -> tuple[str | None, dict[str, Any]]:
        """Reread rule sources at a compact boundary; keep the turn snapshot on failure.

        Returns:
            tuple: the reloaded system text, or None to keep the current snapshot, and the
                provenance recorded on the compact request either way.
        """
        try:
            return load_instructions(self.instruction_config)
        except InstructionLoadError as error:
            # Mid-turn this must not end a turn already recovering from context pressure;
            # turn start still refuses to run at all on the same error.
            logger.error("Keeping the turn's instruction snapshot: {}", error)
            return None, {"status": "error", "detail": str(error)}

    def _save_session(self, result: TurnResult, history_length: int, state: ContextState) -> None:
        """Append this turn's raw delta, then a checkpoint that references it.

        The two writes are ordered and separately atomic: a checkpoint never names messages
        that are not already on disk. An incomplete turn contributes neither.
        """
        if self.state_dir is None or result.session_id is None:
            return
        completed = result.stop_reason == RunStopReason.FINAL_RESPONSE
        store = SessionStore(Path(self.state_dir, "sessions"))
        try:
            store.append(result.session_id, result.run_id,
                         result.messages[1 + history_length:] if completed else [],
                         started_at=result.started_at, stop_reason=result.stop_reason)
        except (OSError, ValueError) as error:
            logger.error("Could not save session for run {}; this turn will not be remembered: {}",
                         result.run_id, error)
            return
        if not completed or not state.summary or state.covered <= 1:
            return
        try:
            # covered indexes raw, whose system message is at 0; session history is raw[1:].
            store.append_checkpoint(
                result.session_id, result.run_id, covered=state.covered - 1,
                summary=state.summary, history=result.messages[1:],
                config={"compact_prompt_sha256": compact_prompt_digest(),
                        "summary_max_tokens": self.limits.summary_max_tokens,
                        "model": self.llm.settings()})
        except (OSError, ValueError) as error:
            logger.error("Could not save the summary checkpoint for run {}; "
                         "the next turn replays raw history: {}", result.run_id, error)

    def _run_turn(self, result: TurnResult, state: ContextState, *, compact: bool = False,
                  on_progress: Callable[[str], None] | None = None) -> None:
        # Keep the original layout for tracing, session saving and evaluators.
        # Only raw_messages receives new events; prepared views are disposable.
        raw_messages = result.messages
        compactor = Compactor(self.llm, self.limits,
                              reload_instructions=self._reload_instructions)
        last_failure, failure_count = None, 0
        streak_key, streak = None, 0  # consecutive identical calls, across batches
        tool_attempts = 0
        used_ids = {call["id"] for m in raw_messages for call in m.get("tool_calls", [])}
        planner = PlanTool() if state.plan is not None else None

        def repeated_failure(key: tuple) -> bool:
            nonlocal last_failure, failure_count
            failure_count = failure_count + 1 if key == last_failure else 1
            last_failure = key
            if failure_count < self.limits.max_same_failures:
                return False
            result.stop_reason = RunStopReason.NO_PROGRESS
            result.error_message = f"The same {key[0]} failure ({key[-1]}) occurred {failure_count} times."
            return True

        for iteration in range(1, self.limits.max_iterations + 1):
            if used_model_calls(result.model_requests) >= self.limits.max_iterations:
                return
            outcome = CompactOutcome.SKIPPED
            try:
                prepared_messages = state.messages()
                schemas = self.registry.schemas()
                if planner is not None:
                    schemas = {**schemas, planner.name: planner.to_schema()}
                budget = self.llm.measure_context(prepared_messages, schemas)
                # Cheap first: stubs cost no model call, so they run before the summary trigger.
                share = window_share(budget)
                if (self.limits.elide_ratio is not None and share is not None
                        and share >= self.limits.elide_ratio
                        and state.elide(self.limits.elide_min_chars)):
                    prepared_messages = state.messages()
                    budget = self.llm.measure_context(prepared_messages, schemas)
                    share = window_share(budget)
                # A share, like elision, so the trigger keeps its meaning at any window size.
                # The hard fit rule still triggers when the share is unknown or margin is short.
                if (compact or not context_fits(budget, self.limits.context_margin_tokens)
                        or (share is not None and share >= self.limits.compact_ratio)):
                    compact = False
                    # The actor's request is deliberately not appended yet: it precedes no
                    # summary in the trace, and the compactor reserves the turn's last slot
                    # for it, so counting it here would spend that reserve on itself.
                    outcome = compactor.attempt(state, schemas, result.model_requests,
                                                iteration, before=budget)
                    if outcome is CompactOutcome.APPLIED:
                        prepared_messages = state.messages()
                        budget = self.llm.measure_context(prepared_messages, schemas)
            except Exception as error:
                request = ModelRequest(iteration, len(raw_messages), covered_boundary=state.covered,
                                       last_sent_boundary=state.last_sent,
                                       status=ModelRequestStatus.MODEL_ERROR)
                result.model_requests.append(request)
                result.stop_reason = RunStopReason.MODEL_ERROR
                result.error_message = f"{type(error).__name__}: {error}"
                request.error_message = result.error_message
                logger.error("Could not prepare the actor input: {}", error)
                return

            request = ModelRequest(iteration, len(raw_messages), budget=budget,
                                   covered_boundary=state.covered,
                                   last_sent_boundary=state.last_sent,
                                   elided_messages=len(state.elided),
                                   input_messages=deepcopy(prepared_messages),
                                   tools=deepcopy(schemas))
            result.model_requests.append(request)
            # A failed attempt only ends the turn when the request it was meant to relieve
            # cannot be sent anyway. A manual compact still has a working turn behind it.
            blocked = self._block_on_budget(result, request)
            if blocked and outcome is CompactOutcome.FAILED and request.error_code == "context_limit":
                # Name the real cause; an unavailable measurement keeps its own code.
                result.error_message = "Compaction failed; raw evidence and completed tool effects were preserved."
                request.error_message = result.error_message
            if blocked:
                return

            # Parsing may fail, but the actor still received this input. Fresh feedback
            # and subsequent tool results remain beyond this boundary until its next call.
            state.last_sent = len(raw_messages)
            try:
                response = self.llm.generate(
                    prepared_messages, schemas,
                    max_tool_calls=self.limits.max_tool_calls_per_response)
            except ResponseError as error:
                request.status = ModelRequestStatus.PARSE_ERROR
                request.raw_response = error.raw_response
                request.error_code = error.code
                request.error_message = str(error)
                if isinstance(error.raw_response, dict):
                    request.usage = LLM.read_usage(error.raw_response.get("usage"))
                    choices = error.raw_response.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        request.finish_reason = choices[0].get("finish_reason")
                logger.error("LLM response error: {}", error)
                raw_messages.append({"role": "user", "content": recovery_feedback(error)})
                if repeated_failure(("response", error.code)):
                    return
                continue
            except Exception as error:
                request.status = ModelRequestStatus.MODEL_ERROR
                result.stop_reason = RunStopReason.MODEL_ERROR
                result.error_message = f"{type(error).__name__}: {error}"
                request.error_message = result.error_message
                logger.error("Model backend error: {}", error)
                return
            for index, call in enumerate(response.tool_calls, 1):
                if not call.call_id or call.call_id in used_ids:
                    suffix = f"_{index}" if len(response.tool_calls) > 1 else ""
                    call.call_id = f"{result.run_id}_call_{iteration}{suffix}"
                    while call.call_id in used_ids:
                        call.call_id += "_"
                used_ids.add(call.call_id)
            request.status = ModelRequestStatus.COMPLETED  # previous object has been appended to result, and updates will reflect there too.
            request.call_ids = [call.call_id for call in response.tool_calls]
            request.usage = LLM.read_usage(response.usage)
            request.finish_reason = response.finish_reason
            request.raw_response = response.raw_response
            raw_messages.append(response.to_message())
            if response.type == ResponseType.tool_call:
                if response.content and on_progress is not None:
                    on_progress(response.content) # print out assistant's any extra content other than just calling tools
                reminder = None
                for call in response.tool_calls:
                    if tool_attempts >= self.limits.max_tool_calls:
                        self._finish_pending_tools(result, "Turn tool-call budget exhausted.")
                        result.stop_reason = RunStopReason.TOOL_LIMIT
                        result.error_message = f"Reached {self.limits.max_tool_calls} tool attempts."
                        return
                    tool_attempts += 1
                    logger.debug("Tool call detected: {} {}", call.name, call.arguments)
                    if planner is not None and call.name == planner.name:
                        # A harness component, not a workspace action: never the registry's.
                        tool_result = planner.invoke(call.arguments, call.call_id)
                        if tool_result.ok:
                            state.plan = planner.render()
                    else:
                        tool_result = self.execute_tool(call.name, call.arguments, call.call_id)
                    if tool_result.ok:
                        logger.debug("Tool executed successfully: {} ({})", tool_result.tool_name, tool_result.call_id)
                    else:
                        logger.error("Tool execution failed due to: {}", tool_result.error_message)
                    raw_messages.append(tool_result.to_message())

                    key = (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False))
                    streak = streak + 1 if key == streak_key else 1
                    streak_key = key
                    if tool_result.ok:
                        last_failure, failure_count = None, 0
                        if streak == self.limits.stuck_reminder_calls:
                            reminder = REPEATED_CALL.format(tool=call.name, count=streak)
                    else:
                        self._finish_pending_tools(result,
                            "An earlier call failed. Reconsider these calls using its result before retrying.")
                        if repeated_failure(key + (tool_result.error_code,)):
                            return
                        # Warn once, one failure before the stop, so the model gets a last chance.
                        if failure_count == self.limits.max_same_failures - 1:
                            reminder = REPEATED_FAILURE.format(tool=call.name, count=failure_count)
                        break
                # After the whole batch, so no call is separated from its result.
                if reminder is not None:
                    raw_messages.append({"role": "user", "content": reminder})
                    result.stuck_reminders += 1
            elif response.type == ResponseType.direct:
                result.final_answer = response.content
                result.stop_reason = RunStopReason.FINAL_RESPONSE
                return

    def _session_command(self, command: str, session_id: str, store: SessionStore | None) -> str:
        if command == "/reset":
            if store:
                store.reset(session_id)
            return session_id
        if command == "/new":
            return uuid4().hex
        if store is None:
            raise ValueError("Session switching requires a state directory.")
        next_id = command.split(maxsplit=1)[1]
        store.load_records(next_id)  # Validate before switching away from the current session.
        return next_id

    def run_repl(self, session_id: str = "default_session", *,
                 user_color: int = 33, agent_color: int = 32):
        """ANSI label colors: 31 red, 32 green, 33 yellow, 34 blue, 35 magenta, 36 cyan."""
        user_label = color_label("User:", user_color)
        agent_label = color_label("Agent:", agent_color)
        store = SessionStore(Path(self.state_dir, "sessions")) if self.state_dir is not None else None
        compact_next = False
        while True:
            try:
                user_input = input(f"{user_label} ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            command = user_input.strip()
            if command.lower() in {"exit", "quit", "/quit", "/exit"}:
                break
            if command == "/compact":
                # Compaction runs inside a turn, against that turn's own view, so the
                # request that follows is the earliest point this can take effect.
                compact_next = True
                print("The next request will summarize earlier history before acting.")
                continue
            if command in {"/new", "/reset"} or command.startswith("/session "):
                try:
                    session_id = self._session_command(command, session_id, store)
                    print(f"Session: {session_id}")
                except (OSError, ValueError) as error:
                    print(f"Could not update session: {error}")
                continue
            try:
                history = store.load_history(session_id) if store else []
                checkpoint = store.load_checkpoint(session_id, history) if store else None
            except (OSError, ValueError) as error:
                print(f"Session unavailable: {error}. Use /new, /reset, or /session <id> to recover.")
                continue
            try:
                result = self.run_turn(user_input, history, session_id=session_id,
                                       compact=compact_next, checkpoint=checkpoint,
                                       on_progress=lambda text: print(f"{agent_label} {text}"))
            except InstructionLoadError as error:
                print(f"Instructions unavailable: {error}")
                continue  # The turn never started, so a pending /compact still applies.
            compact_next = False
            if result.stop_reason == RunStopReason.FINAL_RESPONSE:
                print(f"{agent_label} {result.final_answer}")
            elif result.stop_reason == RunStopReason.MODEL_ERROR:
                print(f"Stopped: model error occurred: {result.error_message}")
            elif result.stop_reason == RunStopReason.MAX_ITERATIONS:
                print("Stopped: reached maximum iterations.")
            elif result.stop_reason == RunStopReason.INTERRUPTED:
                print("Turn interrupted.")
            elif result.stop_reason == RunStopReason.CONTEXT_LIMIT:
                print(f"Stopped: {result.error_message}")
                print(
                    "Completed tool actions remain in effect. "
                    "This turn is not included in replayable session history."
                )
            else:
                print(f"Stopped: {result.stop_reason}. {result.error_message or ''}")


if __name__ == "__main__":
    llm = LLM()
    # Planning lengthens weak-model runs, so the REPL raises the turn budgets with it.
    agent = Agent(llm, state_dir="./outputs/sessions",
                  limits=AgentLimits(planning=True, max_iterations=30, max_tool_calls=60),
                  instruction_config=InstructionConfig(workspace=default_workspace))
    try:
        agent.run_repl()
    finally:
        llm.close()
