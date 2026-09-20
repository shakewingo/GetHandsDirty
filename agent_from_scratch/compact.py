"""Bounded summarization of a turn's own transcript, run before the waiting actor."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
import json

from .config import AgentLimits, PROMPTS_DIR
from .context import ContextState, context_fits
from .llm import LLM, ResponseError, ResponseType
from .trace import ModelRequest, ModelRequestStatus, used_model_calls


class CompactOutcome(StrEnum):
    """What one attempt spent, and therefore what the caller must do next."""

    APPLIED = "applied"  # A smaller, fitting view is published; re-measure before sending.
    SKIPPED = "skipped"  # A gate refused before anything was spent; the hard gate decides.
    FAILED = "failed"    # An attempt reached the model and produced no usable view.


class Compactor:
    """One bounded attempt per call, shared by manual requests and pressure recovery.

    Publishes only a smaller, fitting view. No raw edits, tool execution or checkpoint IO.
    The same model's configured output reserve bounds both actor and summary generation.
    """

    def __init__(self, llm: LLM, limits: AgentLimits):
        self.llm = llm
        self.limits = limits
        self._prompt: str | None = None

    def attempt(self, state: ContextState, schemas: dict, requests: list[ModelRequest],
                iteration: int, *, before: dict | None = None) -> CompactOutcome:
        """Summarize one legal cut. The outcome, not a mutated field, tells the caller what happened.

        `before` is the caller's own measurement of this same view and these same schemas;
        supplying it avoids tokenizing the turn's largest input a second time.
        """
        boundary = self._plan(state, requests)
        if boundary is None:
            return CompactOutcome.SKIPPED
        # Record the attempt before spending anything: this exact cut is never retried.
        state.attempted_boundary = boundary
        request = ModelRequest(iteration, len(state.raw), purpose="compact",
                               covered_boundary=boundary, last_sent_boundary=state.last_sent)
        requests.append(request)
        summary = self._summarize(state, boundary, request)
        if summary is None:
            return CompactOutcome.FAILED
        if self._publish(state, boundary, summary, schemas, request, before):
            return CompactOutcome.APPLIED
        return CompactOutcome.FAILED

    def _plan(self, state: ContextState, requests: list[ModelRequest]) -> int | None:
        """Return the cut worth a model call, or None to refuse without spending one."""
        boundary = state.compact_boundary()
        if (boundary <= state.covered or boundary == state.attempted_boundary
                or state.summary_calls >= self.limits.max_compact_calls
                # Leave the actor the last slot: a summary nobody can act on is wasted.
                or used_model_calls(requests) >= self.limits.max_iterations - 1):
            return None
        return boundary

    def _summarize(self, state: ContextState, boundary: int, request: ModelRequest) -> str | None:
        """Generate the handoff for raw[covered:boundary]; None when nothing usable came back."""
        generated = False
        try:
            if self._prompt is None:
                self._prompt = (PROMPTS_DIR / "compact.md").read_text(encoding="utf-8")
            request.input_messages = [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": json.dumps({
                    "previous_summary": state.summary,
                    "messages": state.raw[state.covered:boundary],
                    "current_request": state.raw[state.turn_start].get("content"),
                }, ensure_ascii=False)},
            ]
            request.tools = {}
            request.budget = self.llm.measure_context(request.input_messages, {})
            if not context_fits(request.budget, self.limits.context_margin_tokens):
                request.status = ModelRequestStatus.BLOCKED
                request.error_message = "Summary input does not fit; raw evidence was preserved."
                return None
            state.summary_calls += 1
            generated = True
            response = self.llm.generate(deepcopy(request.input_messages), {})
            request.raw_response = response.raw_response
            request.usage = LLM.read_usage(response.usage)
            request.finish_reason = response.finish_reason
            request.status = ModelRequestStatus.COMPLETED
            if response.type != ResponseType.direct or not response.content.strip():
                request.error_message = "Compaction requires a nonempty text summary without tool calls."
                return None
            return response.content
        except Exception as error:
            self._record_error(request, error, generated=generated)
            return None

    def _publish(self, state: ContextState, boundary: int, summary: str, schemas: dict,
                 request: ModelRequest, before: dict | None) -> bool:
        """Swap in the candidate only when it both fits and is strictly smaller."""
        try:
            candidate = ContextState(raw=state.raw, turn_start=state.turn_start,
                                     last_sent=state.last_sent, covered=boundary,
                                     summary=summary, elided=state.elided, plan_text=state.plan_text)
            # Nothing touched by planning or summarizing feeds messages(), so a measurement
            # the caller took of this same view still describes it exactly.
            if before is None:
                before = self.llm.measure_context(state.messages(), schemas)
            after = self.llm.measure_context(candidate.messages(), schemas)
            request.compact_before = before
            request.compact_after = after
            if (not context_fits(after, self.limits.context_margin_tokens)
                    or not before or before.get("prompt_tokens") is None
                    or after["prompt_tokens"] >= before["prompt_tokens"]):
                request.error_message = "Summary did not produce a smaller fitting actor input."
                return False
            state.covered, state.summary = boundary, summary
            return True
        except Exception as error:
            self._record_error(request, error, generated=True)
            return False

    @staticmethod
    def _record_error(request: ModelRequest, error: Exception, *, generated: bool) -> None:
        """A failure before generation was never charged to the model; record it as blocked."""
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
