from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from loguru import logger
from .llm import LLM, ResponseError, ResponseType, _MODEL_PATH, _QWEN_TEMPLATE
from .session import SessionStore
from .tools.base import ToolResult
from .tools.register import ToolRegistry, default_registry
from .trace import ModelRequest, ModelRequestStatus, RunStopReason, TraceStore, TurnResult
from .utils import color_label, render_prompt

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


class Agent:
    def __init__(self, llm: LLM, state_dir: str | None = None,
                 *, registry: ToolRegistry | None = None):
        self.llm = llm
        self.max_iterations = 20
        self.state_dir = state_dir
        self.registry = default_registry if registry is None else registry

    def execute_tool(self, tool_name: Any, tool_params: Any, call_id: str = "") -> ToolResult:
        return self.registry.invoke(tool_name, tool_params, call_id=call_id)

    def run_turn(self, user_input: str, history: list[ChatCompletionRequestMessage] | None = None,
                 *, session_id: str | None = None) -> TurnResult:
        """Execute supplied history; session_id associates and saves this run, not loads history."""
        started = monotonic()
        result = TurnResult(
            messages=[{"role": "system", "content": render_prompt("system.md")},
                      *deepcopy(history or []), {"role": "user", "content": user_input}],
            run_id=uuid4().hex, session_id=session_id, input=user_input,
            started_at=datetime.now(timezone.utc).isoformat(),
            settings={**self.llm.settings(), "max_iterations": self.max_iterations},
        )
        trace = TraceStore(Path(self.state_dir, "runs") if self.state_dir is not None else None)
        try:
            self._run_turn(result, trace)
        except KeyboardInterrupt:
            # A completed model request can still be followed by an interrupted tool.
            if result.model_requests and result.model_requests[-1].status == ModelRequestStatus.STARTED:
                result.model_requests[-1].status = ModelRequestStatus.INTERRUPTED
            result.stop_reason = RunStopReason.INTERRUPTED
        result.elapsed_seconds = round(monotonic() - started, 2)
        trace.save_run(result)
        self._save_session(result, len(history or []))
        return result

    def _save_session(self, result: TurnResult, history_length: int) -> None:
        if self.state_dir is None or result.session_id is None:
            return
        messages = result.messages[1 + history_length:] if result.stop_reason == RunStopReason.FINAL_RESPONSE else []
        try:
            SessionStore(Path(self.state_dir, "sessions")).append(
                result.session_id, result.run_id, messages,
                started_at=result.started_at, stop_reason=result.stop_reason,
            )
        except (OSError, ValueError) as error:
            logger.error("Could not save session for run {}; this turn will not be remembered: {}",
                         result.run_id, error)

    def _run_turn(self, result: TurnResult, trace: TraceStore) -> None:
        messages = result.messages
        for iteration in range(1, self.max_iterations + 1):
            request = ModelRequest(iteration, len(messages))
            result.model_requests.append(request)
            try:
                response = self.llm.generate(messages, self.registry.schemas())
            except ResponseError as error:
                request.status = ModelRequestStatus.PARSE_ERROR
                if isinstance(error.raw_response, dict):
                    request.usage = LLM.read_usage(error.raw_response.get("usage"))
                logger.error("LLM response error: {}", error)
                trace.save_parse_error(result, iteration, error)
                messages.append({"role": "user", "content": (
                    f"Your previous response could not be processed: {error}. "
                    "No tool executed for that response. Continue the user's task: emit only one "
                    "tool call with a registered name and arguments matching its schema. "
                    "For <tool_call>, use one JSON object with name and arguments fields, "
                    "without extra braces. Correct the call yourself; a formatting error "
                    "does not require clarification from the user."
                )})
                continue
            except Exception as error:
                request.status = ModelRequestStatus.MODEL_ERROR
                result.stop_reason = RunStopReason.MODEL_ERROR
                result.error_message = f"{type(error).__name__}: {error}"
                logger.error("Model backend error: {}", error)
                return
            if response.type == ResponseType.tool_call and not response.call_id:
                response.call_id = f"{result.run_id}_call_{iteration}"
            request.status = ModelRequestStatus.COMPLETED  # previous object has been appended to result, and updates will reflect there too.
            request.call_id = response.call_id or None
            request.usage = LLM.read_usage(response.usage)
            request.finish_reason = response.finish_reason
            request.response_file = trace.save_model_response(result, iteration, response)
            messages.append(response.to_message())
            if response.type == ResponseType.tool_call:
                logger.debug("Tool call detected: {} {}", response.tool_name, response.tool_params)
                tool_result = self.execute_tool(response.tool_name, response.tool_params, response.call_id)
                if tool_result.ok:
                    logger.debug("Tool executed successfully: {}", tool_result)
                else:
                    logger.error("Tool execution failed due to: {}", tool_result.error_message)
                messages.append({"role": "tool", "content": json.dumps(asdict(tool_result)),
                                 "tool_call_id": tool_result.call_id})
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
        while True:
            try:
                user_input = input(f"{user_label} ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            command = user_input.strip()
            if command.lower() in {"exit", "quit", "/quit", "/exit"}:
                break
            if command in {"/new", "/reset"} or command.startswith("/session "):
                try:
                    session_id = self._session_command(command, session_id, store)
                    print(f"Session: {session_id}")
                except (OSError, ValueError) as error:
                    print(f"Could not update session: {error}")
                continue
            try:
                history = store.load_history(session_id) if store else []
            except (OSError, ValueError) as error:
                print(f"Session unavailable: {error}. Use /new, /reset, or /session <id> to recover.")
                continue
            result = self.run_turn(user_input, history, session_id=session_id)
            if result.stop_reason == RunStopReason.FINAL_RESPONSE:
                print(f"{agent_label} {result.final_answer}")
            elif result.stop_reason == RunStopReason.MODEL_ERROR:
                print(f"Stopped: model error occurred: {result.error_message}")
            elif result.stop_reason == RunStopReason.MAX_ITERATIONS:
                print("Stopped: reached maximum iterations.")
            elif result.stop_reason == RunStopReason.INTERRUPTED:
                print("Turn interrupted.")


if __name__ == "__main__":
    llm = LLM(model_path=str(_MODEL_PATH), temperature=0.0, max_tokens=2048,
              n_gpu_layers=-1, n_ctx=8000, chat_template_path=_QWEN_TEMPLATE)
    agent = Agent(llm, state_dir="./outputs/sessions")
    agent.run_repl()
