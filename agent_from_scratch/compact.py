"""Bounded summarization of a turn's own transcript, run before the waiting actor."""

from __future__ import annotations

from copy import deepcopy
import json

from .config import PROMPTS_DIR
from .context import ContextState, context_fits
from .llm import LLM, ResponseError, ResponseType
from .trace import ModelRequest, ModelRequestStatus, used_model_calls


def compact_context(state: ContextState, llm, schemas: dict, limits,
                    requests: list, iteration: int) -> bool:
    """One bounded attempt shared by manual calls and automatic pressure recovery.

    Publish only a smaller, fitting view. No raw edits, tool execution or checkpoint IO.
    The same model's configured output reserve bounds both actor and summary generation.
    """
    boundary = state.compact_boundary()
    if (boundary <= state.covered or boundary == state.attempted_boundary
            or state.summary_calls >= limits.max_compact_calls
            # Leave the actor the last slot: a summary nobody can act on is wasted.
            or used_model_calls(requests) >= limits.max_iterations - 1):
        return False
    state.attempted_boundary = boundary
    request = ModelRequest(iteration, len(state.raw), purpose="compact",
                           covered_boundary=boundary, last_sent_boundary=state.last_sent)
    requests.append(request)
    generated = False
    try:
        prompt = (PROMPTS_DIR / "compact.md").read_text(encoding="utf-8")
        request.input_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({
                "previous_summary": state.summary,
                "messages": state.raw[state.covered:boundary],
                "current_request": state.raw[state.turn_start].get("content"),
            }, ensure_ascii=False)},
        ]
        request.tools = {}
        request.context = llm.measure_context(request.input_messages, {})
        if not context_fits(request.context, limits.context_margin_tokens):
            request.status = ModelRequestStatus.BLOCKED
            request.error_message = "Summary input does not fit; raw evidence was preserved."
            return False
        state.summary_calls += 1
        generated = True
        response = llm.generate(deepcopy(request.input_messages), {})
        request.raw_response = response.raw_response
        request.usage = LLM.read_usage(response.usage)
        request.finish_reason = response.finish_reason
        request.status = ModelRequestStatus.COMPLETED
        if response.type != ResponseType.direct or not response.content.strip():
            request.error_message = "Compaction requires a nonempty text summary without tool calls."
            return False
        candidate = ContextState(state.raw, state.turn_start, state.last_sent,
                                 covered=boundary, summary=response.content)
        before = llm.measure_context(state.messages(), schemas)
        after = llm.measure_context(candidate.messages(), schemas)
        request.compact_before = before
        request.compact_after = after
        if (not context_fits(after, limits.context_margin_tokens)
                or not before or before.get("prompt_tokens") is None
                or after["prompt_tokens"] >= before["prompt_tokens"]):
            request.error_message = "Summary did not produce a smaller fitting actor input."
            return False
        state.covered, state.summary = boundary, response.content
        return True
    except Exception as error:
        request.status = (ModelRequestStatus.PARSE_ERROR if isinstance(error, ResponseError)
                          else ModelRequestStatus.MODEL_ERROR)
        if not generated:
            request.status = ModelRequestStatus.BLOCKED
        request.error_message = f"{type(error).__name__}: {error}"
        if isinstance(error, ResponseError):
            request.error_code = error.code
            request.raw_response = error.raw_response
            if isinstance(error.raw_response, dict):
                request.usage = LLM.read_usage(error.raw_response.get("usage"))
                choices = error.raw_response.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    request.finish_reason = choices[0].get("finish_reason")
        return False
