"""Freeze the Stage 8 benchmark: everything that must not move once the test split runs.

Usage: python -m agent_from_scratch.evals freeze
Writes evals/bench/manifest.json. Run it once, after the last skeleton lands, and again only
when a deliberate change (a new skeleton, a limits default, a prompt file) needs a new freeze.
"""

import dataclasses
import json
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory

from .run import specs
from ..verify import digest
from ...config import AgentLimits, CHAT_TEMPLATE_PATH, MAX_TOKENS, N_CTX, TEMPERATURE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"


def _source_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in ROOT.rglob("*")
           if p.is_file() and p.suffix in {".py", ".jinja"} and "__pycache__" not in p.parts}


def build_manifest() -> dict:
    """Everything a Stage 8 test-split run must match. Loads no model; cheap enough to call
    before every `--final` run."""
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    tasks = []
    for skeleton, ctx in specs():
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
            task = skeleton.build(rng, workspace, ctx)
            tasks.append({"id": task.id, "skeleton": task.skeleton, "split": task.split,
                         "prompt_sha256": digest(task.prompt.encode()),
                         "expect_sha256": digest(json.dumps(dataclasses.asdict(task.expect),
                                                            sort_keys=True).encode())})
    return {"source_sha256": _source_hashes(),
           "system_prompt_sha256": digest((ROOT / "prompts/system.md").read_bytes()),
           "chat_template_sha256": digest(CHAT_TEMPLATE_PATH.read_bytes()),
           "n_ctx": N_CTX, "decoding": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
           "limits": dataclasses.asdict(AgentLimits()), "memory": "off",
           "splits_sha256": digest((HERE / "splits.json").read_bytes()),
           "tasks": tasks, "verifier_sha256": digest((HERE / "verify.py").read_bytes()),
           "git_revision": revision}


def save_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n")


def manifest_drift() -> dict[str, tuple]:
    """Fields that differ from the committed manifest; {} if unfrozen or unchanged.

    `git_revision` is recorded for provenance but never compared: committing the frozen
    manifest itself always changes HEAD, so gating on it would make every freeze self-drift
    against its own commit the moment it lands.
    """
    if not MANIFEST_PATH.exists():
        return {}
    frozen = json.loads(MANIFEST_PATH.read_text())
    current = build_manifest()
    drift = {key: (frozen.get(key), current.get(key)) for key in frozen
            if key not in ("tasks", "git_revision") and frozen.get(key) != current.get(key)}
    if {t["id"]: t for t in frozen.get("tasks", [])} != {t["id"]: t for t in current.get("tasks", [])}:
        drift["tasks"] = "task set or content changed"
    return drift


def main():
    save_manifest()
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
