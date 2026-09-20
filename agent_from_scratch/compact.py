"""Bounded summarization of a turn's own transcript, run before the waiting actor."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from enum import StrEnum
from hashlib import sha256
import json

from .config import AgentLimits, PROMPTS_DIR
from .context import ContextState, context_fits
from .llm import LLM, ResponseError, ResponseType
from .trace import ModelRequest, ModelRequestStatus, used_model_calls


def compact_prompt_digest() -> str:
    """Identify the summarizer instructions a checkpoint was produced under."""
    return sha256((PROMPTS_DIR / "compact.md").read_bytes()).hexdigest()


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

    def __init__(self, llm: LLM, limits: AgentLimits, *,
                 reload_instructions: Callable[[], tuple[str | None, dict]] | None = None):
        self.llm = llm
        self.limits = limits
        self.reload_instructions = reload_instructions
        self._prompt: str | None = None

    def attempt(self, state: ContextState, schemas: dict, requests: list[ModelRequest],
                iteration: int, *, before: dict | None = None) -> CompactOutcome:
        """Summarize one legal cut, with at most one corrective retry at that same cut.

        Args:
            state: the turn's context state; only `covered`, `summary`, `attempted_boundary`
                and `summary_calls` may change, and only on success or on a spent call.
            schemas: the tool schemas the actor's request will carry.
            requests: the turn's request list; one entry is appended per summary call.
            iteration: the agent loop iteration these calls belong to.
            before: the caller's own measurement of this same view and these same schemas,
                which avoids tokenizing the turn's largest input a second time.

        Returns:
            CompactOutcome: APPLIED when a smaller fitting view was published, SKIPPED when
                a gate refused before anything was spent, FAILED when a call reached the
                model and produced no usable view.
        """
        boundary = self._plan(state, requests)
        if boundary is None:
            return CompactOutcome.SKIPPED
        # Record the attempt before spending anything: this exact cut is never retried by a
        # *later* attempt, whatever the retry below does inside this one.
        state.attempted_boundary = boundary
        outcome = CompactOutcome.SKIPPED
        retry: str | None = None
        for _ in range(self.limits.max_summary_calls_per_attempt):
            if outcome is not CompactOutcome.SKIPPED and self._exhausted(state, requests):
                break
            request = ModelRequest(iteration, len(state.raw), purpose="compact",
                                   covered_boundary=boundary, last_sent_boundary=state.last_sent)
            requests.append(request)
            outcome = CompactOutcome.FAILED
            summary = self._summarize(state, boundary, request, retry)
            if summary is not None and self._publish(state, boundary, summary, schemas,
                                                     request, before):
                return CompactOutcome.APPLIED
            if request.status is ModelRequestStatus.BLOCKED:
                break  # The summarizer's own input does not fit; a retry cannot change that.
            retry = request.error_message
        return outcome

    def _plan(self, state: ContextState, requests: list[ModelRequest]) -> int | None:
        """Return the cut worth a model call, or None to refuse without spending one."""
        boundary = state.compact_boundary()
        if (boundary <= state.covered or boundary == state.attempted_boundary
                or self._exhausted(state, requests)):
            return None
        return boundary

    def _exhausted(self, state: ContextState, requests: list[ModelRequest]) -> bool:
        """True when another summary call would exceed its own or the turn's ceiling."""
        return (state.summary_calls >= self.limits.max_compact_calls
                # Leave the actor the last slot: a summary nobody can act on is wasted.
                or used_model_calls(requests) >= self.limits.max_iterations - 1)

    def _summarize(self, state: ContextState, boundary: int, request: ModelRequest,
                   retry: str | None = None) -> str | None:
        """Generate the handoff for raw[covered:boundary]; None when nothing usable came back.

        Args:
            state: the turn's context state, read for the previous summary and raw messages.
            boundary: the exclusive end of the range to summarize.
            request: the record this call writes its input, budget, usage and errors into.
            retry: why the previous call in this attempt was unusable, so the final call can
                correct it instead of repeating it; None on the first call.

        Returns:
            str | None: the handoff text, or None when the call was blocked, failed or
                returned something the actor cannot use.
        """
        generated = False
        try:
            if self._prompt is None:
                self._prompt = (PROMPTS_DIR / "compact.md").read_text(encoding="utf-8")
            instructions = self._prompt
            if retry:
                instructions += (f"\n\nThe previous attempt was rejected: {retry}\n"
                                 "Return only the plain-text handoff, and make it shorter.")
            request.input_messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps({
                    "previous_summary": state.summary,
                    "messages": state.raw[state.covered:boundary],
                    "current_request": state.raw[state.turn_start].get("content"),
                }, ensure_ascii=False)},
            ]
            request.tools = {}
            request.budget = self.llm.measure_context(
                request.input_messages, {}, max_tokens=self.limits.summary_max_tokens)
            if not context_fits(request.budget, self.limits.context_margin_tokens):
                request.status = ModelRequestStatus.BLOCKED
                request.error_message = "Summary input does not fit; raw evidence was preserved."
                return None
            state.summary_calls += 1
            generated = True
            response = self.llm.generate(deepcopy(request.input_messages), {},
                                         max_tokens=self.limits.summary_max_tokens)
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
            # The boundary is the one point in a turn that already rebuilds the prompt, so
            # stable rules are reread here and measured as part of the candidate.
            instructions = state.instructions
            if self.reload_instructions is not None:
                text, request.instructions = self.reload_instructions()
                if text is not None:
                    instructions = {"role": "system", "content": text}
            candidate = ContextState(raw=state.raw, turn_start=state.turn_start,
                                     last_sent=state.last_sent, covered=boundary,
                                     summary=summary, instructions=instructions)
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
            state.instructions = instructions
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
