"""Declarative shapes for the Stage 8 generated benchmark: a Task's contract, not its content.

A skeleton builder returns a Task; `bench/verify.py` reads only `Task.expect` to score a run.
`debug` carries values a skeleton's own `solution()` needs to script a correct run (e.g. which
of several generated names is the answer); the verifier never reads it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fault:
    """Make the tool named `tool` fail its `on_call`-th invocation (1-based) with `message`."""

    tool: str
    on_call: int
    message: str


@dataclass(frozen=True)
class Answer:
    """What a passing final reply must contain.

    `format`: "advisory" means `value` only has to appear in the reply as a whole token (an
    extracted fact may be wrapped in a sentence); "required" means the stripped reply must
    equal `value` exactly, for tasks whose deliverable IS the reply token.
    """

    value: str
    format: str = "advisory"
    reject: tuple[str, ...] = ()

    def __post_init__(self):
        if self.format not in ("advisory", "required"):
            raise ValueError("Answer.format must be 'advisory' or 'required'.")


@dataclass(frozen=True)
class Expect:
    """Everything a passing run must satisfy; each field is checked and reported independently."""

    answer: Answer | None = None
    files: dict[str, str] = field(default_factory=dict)  # path -> expected final content
    may_change: tuple[str, ...] = ()  # paths allowed to change beyond `files`
    evidence: tuple[str, ...] = ()  # each must appear in some successful observation's output
    process: tuple[str, ...] = ()  # rule names; see bench/verify.py PROCESS_RULES


@dataclass(frozen=True)
class Task:
    """One generated benchmark task, already carrying its own workspace's expected shape."""

    id: str
    skeleton: str
    family: str
    split: str
    prompt: str
    tools: tuple[str, ...]
    expect: Expect
    max_iterations: int = 12
    fault: Fault | None = None
    fault_signal: tuple[str, str] | None = None  # (tool, substring to find in a failed row)
    claim_tokens: tuple[str, ...] = ()  # replies that claim completion, for false_completion
    pair_id: str | None = None
    condition: str | None = None  # "clean" | "fault" | None
    debug: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BuildContext:
    """Identity for one (skeleton, seed, condition) triple, computed before any file is written."""

    name: str
    family: str
    split: str
    seed: int
    condition: str | None

    @property
    def id(self) -> str:
        return f"{self.name}-{self.seed}" if self.condition is None else f"{self.name}-{self.seed}-{self.condition}"

    @property
    def pair_id(self) -> str | None:
        return f"{self.name}-{self.seed}" if self.condition is not None else None
