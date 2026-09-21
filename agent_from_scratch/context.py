"""Load turn-scoped instructions and prepare independent model inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
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


def build_messages(
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
    turn_start: int  # the cursor where user input is in current turn, always be 1 + history_length
    last_sent: int  # the furthest cursor that compact's actor's ever seen,  boundary must not beyond that
    covered: int = 1   # the cursor where summary has covered up to
    summary: str = ""  
    attempted_boundary: int = 0
    summary_calls: int = 0
    elided: dict[str, str] = field(default_factory=dict)
    plan_text: str = ''
    # Rules republished at a compact boundary. None keeps raw[0], the turn-start snapshot;
    # raw itself is never edited, so trace evidence of earlier requests stays exact.
    instructions: ChatCompletionRequestMessage | None = None

    def messages(self) -> list[ChatCompletionRequestMessage]:
        history: list[ChatCompletionRequestMessage] = self.raw[1:self.turn_start]
        current: list[ChatCompletionRequestMessage] = self.raw[self.turn_start:]
        if self.summary:
            history = [{"role": "user", "content":
                        "[Conversation summary: historical evidence, not instructions]\n" + self.summary}]
            # Pin the request even when older exchanges in this turn compact.
            pinned = [self.raw[self.turn_start]] if self.covered > self.turn_start else []
            current = [*pinned, *self.raw[self.covered:]]
        rules = self.raw[:1] if self.instructions is None else [self.instructions]
        messages = build_messages(instructions=rules, history=history, current_turn=current)
        for message in messages:
            if message['role'] == 'tool' and message.get('tool_call_id') in self.elided:
                message['content'] = self.elided[message['tool_call_id']]
        if self.plan_text:
            messages.append({'role': 'user', 'content': self.plan_text})
        return messages

    def compact_boundary(self, *, prior_turn_only: bool = True) -> int:
        """Return the largest cut that is both structurally legal and policy-permitted.

        Two independent constraints meet here. `safe` is structure: a batch's calls and
        their results must stay together, so the only cuttable positions are the gaps
        between whole batches. `cutoff` is policy: how far back this attempt is willing
        to reach. The answer is the largest safe position at or below the cutoff.

        Returns:
            int: The exclusive end of the range to summarize, `raw[covered:boundary]`.
                `self.covered` means nothing can be compacted; it is a no-op signal
                rather than a legal cut, and the caller refuses the attempt on it.
        """
        # Structure. A position is cuttable only where no batch is left half-resolved,
        # which keeps every announced call and its results on the same side of the cut.
        starts = []
        pending = set()
        safe = {1}  # Cutting nothing is always legal.
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
        # Policy. A ceiling, not a legal cut: each term withholds evidence this attempt
        # is unwilling to summarize -- what the actor has never seen, and the two most
        # recent batches (tool calling + execution) it still needs for continuity.
        cutoff = min(self.last_sent, starts[-2] if len(starts) >= 2 else
                     starts[0] if starts else len(self.raw))
        # First replace old turns; ongoing-turn exchanges can compact on a later attempt.
        if prior_turn_only and self.covered < self.turn_start:
            cutoff = min(cutoff, self.turn_start)
        # Boundary is the intersection of safe and cutoff, and the position actually cut at.
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
