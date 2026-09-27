"""Run-directory helpers shared by every suite (`bench`, `pressure`).

The Stage 2B `dev` suite that used to live here was removed after Stage 8; its tasks, fixtures and
byte-offset file tools are in git history (see docs/checkpoints/EVAL_HISTORY.md).
"""

from dataclasses import replace
import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys

from loguru import logger
from ..config import AgentLimits
from ..llm import LLM
from .verify import digest


HERE = Path(__file__).resolve().parent


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_limits(parser: argparse.ArgumentParser, text: str) -> dict:
    """Validate `--limits` JSON against AgentLimits; exits with a usage error when invalid."""
    try:
        overrides = json.loads(text)
        replace(AgentLimits(), **overrides)
    except (ValueError, TypeError) as error:
        parser.error(f"--limits must be a JSON object of AgentLimits fields: {error}")
    return overrides


def seed_every_request(model: LLM, seed: int) -> LLM:
    """Pass `seed` to every llama_cpp request; the vLLM backend takes its seed directly."""
    backend = model.llm.create_chat_completion
    model.llm.create_chat_completion = lambda *a, **kw: backend(*a, **kw, seed=seed)
    return model


def _installed(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def open_run(out: Path, parser: argparse.ArgumentParser, *, suite: str, seed: int, overrides: dict,
             n_ctx: int | None = None, model=None, **extra):
    """Create the evidence directory, load the model and write metadata.json; shared by all suites.

    Args:
        out: new directory; must be outside the package so a run cannot edit its own source.
        suite: name recorded in the metadata.
        seed: passed to every request. Under greedy decoding it is bookkeeping; under the bench's
            sampled protocol it selects which of the k samples this run is.
        overrides: validated AgentLimits overrides, recorded for ablation.
        n_ctx: model window override; None keeps `config.N_CTX`. Ignored when `model` is given.
        model: an already constructed backend (`LLM` or `VLLMClient`); None loads the local GGUF.
        **extra: suite-specific metadata fields.
    """
    root = HERE.parent
    if out.is_relative_to(root):
        parser.error("Evidence must be outside the source package and frozen fixtures.")
    out.mkdir(parents=True, exist_ok=False)
    logger.remove()
    if model is None:
        model = seed_every_request(LLM(n_ctx=n_ctx) if n_ctx else LLM(), seed)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    save(out / "metadata.json", {
        "suite": suite, "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": model.settings(), "seed": seed, "python": sys.version,
        "platform": platform.platform(), "llama_cpp_version": _installed("llama-cpp-python"),
        "httpx_version": _installed("httpx"),
        "git_revision": revision,
        "source_sha256": {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in root.rglob("*")
            if p.is_file() and p.suffix in {".py", ".jinja"} and "__pycache__" not in p.parts},
        "system_prompt": (root / "prompts/system.md").read_text(),
        "limit_overrides": overrides, **extra,
    })
    return model
