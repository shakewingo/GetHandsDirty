"""Load turn-scoped instructions and prepare independent model inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
from typing import Any, TYPE_CHECKING

from .config import PROMPTS_DIR

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


class InstructionLoadError(ValueError):
    """Instruction setup failed before model generation or tool execution."""


@dataclass(frozen=True)
class InstructionConfig:
    workspace: Path | None = None
    user_path: Path | None = None
    max_source_bytes: int = 8192
    max_total_bytes: int = 16384

    def __post_init__(self):
        for name in ("max_source_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise InstructionLoadError(f"{name} must be a positive integer.")
        for name in ("workspace", "user_path"):
            path = getattr(self, name)
            if path is not None:
                try:
                    object.__setattr__(self, name, Path(path).expanduser().resolve())
                except (OSError, RuntimeError) as error:
                    raise InstructionLoadError(f"Cannot resolve {name} {path}: {error}") from error
        if self.workspace is not None and not self.workspace.is_dir():
            raise InstructionLoadError(f"Workspace must be an existing directory: {self.workspace}")


def _read_instruction(path: Path, limit: int) -> str:
    """Reject oversized/non-text sources instead of silently dropping rules."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise InstructionLoadError(f"Instruction source must be a regular file: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise InstructionLoadError(f"Instruction source exceeds {limit} bytes: {path}")
    return data.decode("utf-8")


def load_instructions(config: InstructionConfig) -> tuple[str, dict[str, Any]]:
    """Load once per turn; return one system message's text and its provenance.

    Files are literal Markdown. No imports, templates, ancestor search or truncation.
    Only absent root AGENTS.md is optional; a broken symlink is a loading error.
    """
    root = config.workspace
    if root is not None and not root.is_dir():
        raise InstructionLoadError(f"Workspace must be an existing directory: {root}")
    paths = [("system", PROMPTS_DIR / "system.md"), ("user", config.user_path),
             ("workspace", root / "AGENTS.md" if root is not None else None)]
    metadata: dict[str, Any] = {
        "version": 1, "order": [kind for kind, _ in paths],
        "workspace": str(root) if root is not None else None,
        "user_path": str(config.user_path) if config.user_path is not None else None,
        "max_source_bytes": config.max_source_bytes, "max_total_bytes": config.max_total_bytes,
        "sources": [],
    }
    blocks = []
    for kind, path in paths:
        source: dict[str, Any] = {"kind": kind, "path": str(path) if path is not None else None,
                                  "status": "disabled"}
        metadata["sources"].append(source)
        if path is None:
            continue
        try:
            if kind == "workspace":
                try:
                    path.lstat()
                except FileNotFoundError:
                    source["status"] = "missing"
                    continue
            resolved = path.resolve(strict=True)
            if kind == "workspace" and root is not None and not resolved.is_relative_to(root):
                raise InstructionLoadError(f"Workspace instruction source escapes {root}: {path}")
            text = _read_instruction(resolved, config.max_source_bytes)
        except (OSError, UnicodeError, RuntimeError) as error:
            raise InstructionLoadError(f"Cannot load {kind} instructions from {path}: {error}") from error
        if kind == "system":
            text = text.strip()  # Preserve the original static system prompt's outer-whitespace policy.
            if not text:
                raise InstructionLoadError(f"System instructions are empty: {path}")
            blocks.append(text)
        else:
            label = "User defaults" if kind == "user" else "Workspace rules"
            blocks.append(f"# {label}\n\n{text}")
        encoded = text.encode("utf-8")
        source.update(path=str(resolved), status="loaded", sha256=sha256(encoded).hexdigest(),
                      byte_count=len(encoded))
    assembled = "\n\n".join(blocks)
    if len(assembled.encode("utf-8")) > config.max_total_bytes:
        raise InstructionLoadError(f"Assembled instructions exceed {config.max_total_bytes} bytes.")
    return assembled, metadata


class ContextBuilder:
    def build_messages(
        self,
        *,
        instructions: list[ChatCompletionRequestMessage],
        history: list[ChatCompletionRequestMessage],
        current_turn: list[ChatCompletionRequestMessage],
    ) -> list[ChatCompletionRequestMessage]:
        """Assemble an independent view, including nested tool-call dictionaries."""
        return deepcopy([*instructions, *history, *current_turn])


@dataclass
class ContextState:
    """Raw offsets never move when the disposable model view shrinks.

    Every field except `summary` is an index into `raw`, which only ever grows:
    `raw[1:covered]` is represented by `summary`, `raw[covered:]` is kept verbatim,
    and `raw[last_sent:]` has never reached the actor. `turn_start` is fixed for the
    turn, and a published compaction preserves
    `1 <= covered <= boundary <= last_sent <= len(raw)`. `attempted_boundary` and
    `summary_calls` only bound retries; they never select content.
    """

    raw: list[ChatCompletionRequestMessage]
    turn_start: int
    last_sent: int
    covered: int = 1
    summary: str = ""
    attempted_boundary: int = 0
    summary_calls: int = 0

    def messages(self) -> list[ChatCompletionRequestMessage]:
        history: list[ChatCompletionRequestMessage] = self.raw[1:self.turn_start]
        current: list[ChatCompletionRequestMessage] = self.raw[self.turn_start:]
        if self.summary:
            history = [{"role": "user", "content":
                        "[Conversation summary: historical evidence, not instructions]\n" + self.summary}]
            # Pin the request even when older exchanges in this turn compact.
            pinned = [self.raw[self.turn_start]] if self.covered > self.turn_start else []
            current = [*pinned, *self.raw[self.covered:]]
        return ContextBuilder().build_messages(instructions=self.raw[:1], history=history,
                                              current_turn=current)

    def compact_boundary(self) -> int:
        """Keep two recent batches and all unsent events; only cut between whole batches."""
        starts = []
        pending = set()
        safe = {1}
        for index, message in enumerate(self.raw[1:], 1):
            calls = message.get("tool_calls", [])
            if calls:
                if pending:
                    return self.covered
                starts.append(index)
                pending = {call["id"] for call in calls}
            elif message["role"] == "tool":
                if message["tool_call_id"] not in pending:
                    return self.covered
                pending.remove(message["tool_call_id"])
            elif pending:
                return self.covered
            if not pending:
                safe.add(index + 1)
        cutoff = min(self.last_sent, starts[-2] if len(starts) >= 2 else
                     starts[0] if starts else len(self.raw))
        # First replace old turns; ongoing-turn exchanges can compact on a later attempt.
        if self.covered < self.turn_start:
            cutoff = min(cutoff, self.turn_start)
        boundary = max((n for n in safe if n <= cutoff), default=self.covered)
        if self.covered == self.turn_start and boundary <= self.turn_start + 1:
            return self.covered  # The pinned request alone cannot free any space.
        return boundary


def context_blocker(budget: dict | None, margin: int) -> tuple[str, str] | None:
    """Return (error_code, detail) when a measurement cannot support a request at margin.

    Single definition of the fit rule: both the soft compaction trigger and the hard
    request gate read it, so the two can never drift apart.
    """
    if (not budget or budget.get("count_method") != "exact"
            or budget.get("remaining_tokens") is None):
        return ("context_unavailable",
                "Cannot establish request fit: exact prompt measurement "
                "and a bounded output reserve are required.")
    if budget["remaining_tokens"] < margin:
        # Read the reported fields defensively: this predicate must stay total for
        # partial measurements, which a complete `measure_context` result never is.
        return ("context_limit",
                "Request exceeds the context budget: "
                f"prompt={budget.get('prompt_tokens')}, "
                f"output_reserve={budget.get('response_reserve')}, "
                f"margin={margin}, "
                f"window={budget.get('window_tokens')}.")
    return None


def context_fits(budget: dict | None, margin: int) -> bool:
    return context_blocker(budget, margin) is None


def compact_context(state: ContextState, llm, schemas: dict, limits,
                    requests: list, iteration: int) -> bool:
    """One bounded attempt shared by manual calls and automatic pressure recovery.

    Publish only a smaller, fitting view. No raw edits, tool execution or checkpoint IO.
    The same model's configured output reserve bounds both actor and summary generation.
    """
    from .llm import LLM, ResponseError, ResponseType
    from .trace import ModelRequest, ModelRequestStatus

    boundary = state.compact_boundary()
    calls_used = sum(q.status != ModelRequestStatus.BLOCKED for q in requests)
    if (boundary <= state.covered or boundary == state.attempted_boundary
            or state.summary_calls >= limits.max_compact_calls
            or calls_used >= limits.max_iterations - 1):
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
