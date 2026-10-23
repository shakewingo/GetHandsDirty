"""Stage 10's GRPO pilot: strict reward, group advantages, task rotation and training rows.

Usage: python -m agent_from_scratch.evals grpo tasks (--pool | --pass P --iteration I)
           [--skeletons a,b] [--per-iteration N]
       python -m agent_from_scratch.evals grpo step --runs RUN_DIR ... --adapter-in DIR
           --output DIR [--lr 1e-5]

A group is one train task sampled at several seeds by the same adapter. Each rollout earns the
strict reward (the verifier passed it and the answer is exactly the requested value) and an
advantage A_i = (r_i - mean r) / std r within its group; a group whose rewards are all equal
teaches nothing and is skipped. Every agent request of a kept rollout becomes one row: the
prompt the model saw and the text it sampled, weighted by its rollout's advantage.

`step` takes one iteration's runs and the adapter that sampled them, takes one optimizer step
(`train.policy_step`), and writes the next adapter with the optimizer state beside it
(`optimizer.pt`), so the next step continues the same AdamW. Torch, transformers and peft are
imported only when a step trains. Design: docs/STAGE10_PLAN.md (G1-G6) and docs/STAGE.md
("Conditional: a GRPO pilot").
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

from ..run import save
from ...config import BENCH_MODEL, BENCH_N_CTX, CHAT_TEMPLATE_PATH
from ...llm import QWEN_END, compile_chat_template, render_chat

ADAPTER_FILE = "adapter_model.safetensors"
OPTIMIZER_FILE = "optimizer.pt"

# Requests whose reply the model sampled: a parse error is still the model's own text.
SAMPLED_STATUSES = ("completed", "parse_error")


def train_pool(skeletons: list[str] | None = None) -> dict[str, list[str]]:
    """Skeleton name -> the IDs of its train tasks, in registration order; only `skeletons` when given.

    Raises ValueError for a named skeleton that is not on the train split (an unleveled Stage 10
    shape is not registered).
    """
    from ..bench.run import specs  # the skeleton registry is heavy; `aggregate` imports `reward` only

    pool: dict[str, list[str]] = {}
    for skeleton, ctx in specs("train"):
        if skeletons is None or skeleton.name in skeletons:
            pool.setdefault(skeleton.name, []).append(ctx.id)
    missing = sorted(set(skeletons or ()) - set(pool))
    if missing:
        raise ValueError(f"Not train skeletons: {missing}")
    return pool


def iteration_tasks(pool: dict[str, list[str]], pass_index: int, iteration: int,
                    per_iteration: int | None = None) -> list[str]:
    """The task IDs that iteration `iteration` (0-based) of pass `pass_index` samples.

    Within a pass, each skeleton's tasks are shuffled with a seed of their own, then taken in
    turn across the skeletons in name order, so each chunk of `per_iteration` IDs draws evenly
    from every skeleton and one pass covers every task once. The default, two per skeleton,
    makes ten iterations a pass for 20 tasks per skeleton.
    """
    per_iteration = per_iteration or 2 * len(pool)
    queues = []
    for name in sorted(pool):
        ids = list(pool[name])
        random.Random(f"grpo:{pass_index}:{name}").shuffle(ids)
        queues.append(ids)
    order = [ids[i] for i in range(max(map(len, queues))) for ids in queues if i < len(ids)]
    iterations = math.ceil(len(order) / per_iteration)
    if not 0 <= iteration < iterations:
        raise ValueError(f"A pass of {len(order)} tasks has iterations 0-{iterations - 1}, not {iteration}.")
    return order[iteration * per_iteration:(iteration + 1) * per_iteration]


def reward(record: dict) -> int:
    """1 when the verifier passed the rollout and its answer is exactly the requested value.

    A task without an expected answer has no `format_exact` check, and only `passed` counts.
    """
    return int(bool(record["passed"]) and record["checks"].get("format_exact") is not False)


def advantages(rewards: list[int]) -> list[float] | None:
    """(r_i - mean) / std over one group, with the population std; None when nothing differs.

    With k passes in G rollouts this is sqrt((G-k)/k) for a pass and -sqrt(k/(G-k)) for a failure.
    """
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    spread = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if spread == 0:
        return None
    return [(r - mean) / spread for r in rewards]


def sampled_text(request: dict) -> str | None:
    """The text the model sampled for `request`, ending in <|im_end|> when it stopped by itself.

    vLLM's completion shape holds it in `choices[0].text`; a reply that failed to parse is kept
    in the chat shape the client built for the parser, `choices[0].message.content`. A truncated
    reply (`finish_reason` "length") gets no end token: the model did not stop there.
    """
    try:
        choice = request["raw_response"]["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    text = choice.get("text")
    if text is None:
        text = (choice.get("message") or {}).get("content")
    if not isinstance(text, str):
        return None
    finish = request.get("finish_reason") or choice.get("finish_reason")
    return text + QWEN_END if finish == "stop" else text


def sampled_rows(trace: dict, template) -> list[dict]:
    """One {"request_index", "prompt", "completion"} per agent request the model answered.

    The prompt is rendered from the request's own input, as the client rendered it; the
    completion is the sampled text, not the canonical rendering of the parsed turn, because the
    gradient must be taken on the tokens the model produced.
    """
    rows = []
    for index, request in enumerate(trace["model_requests"]):
        if request["purpose"] != "agent" or request["status"] not in SAMPLED_STATUSES:
            continue
        completion = sampled_text(request)
        if completion is None:
            raise ValueError(f"Request {index} was answered but holds no sampled text.")
        rows.append({"request_index": index,
                     "prompt": render_chat(template, request["input_messages"], request["tools"]),
                     "completion": completion})
    return rows


def collect_groups(run_dirs: list[Path]) -> tuple[list[dict], dict]:
    """Read one iteration's bench runs (one per seed) and return (rows, metadata).

    Rows are {"id", "prompt", "completion", "weight", "provenance"} for every rollout of every
    mixed group. The runs must be train-split vLLM samples under the chat template on disk, all
    from one adapter (the update is on-policy), with each (task, seed) once.
    """
    text = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    template, template_sha256 = compile_chat_template(text), sha256(text.encode("utf-8")).hexdigest()
    groups: dict[str, list[tuple[int, Path, dict]]] = {}
    adapters, seen = set(), set()
    for run in run_dirs:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        records = json.loads((run / "results.json").read_text(encoding="utf-8"))
        settings, seed = metadata["settings"], metadata["seed"]
        if metadata["split"] != "train" or any(record["split"] != "train" for record in records):
            raise ValueError(f"{run.name}: only the train split may become training data.")
        if settings["backend"] != "vllm":
            raise ValueError(f"{run.name}: backend {settings['backend']!r}; only vLLM runs keep the sampled text.")
        if settings.get("chat_template_sha256") != template_sha256:
            raise ValueError(f"{run.name}: sampled under a different chat template than the one on disk.")
        adapters.add(settings.get("adapter"))
        for record in records:
            if (record["id"], seed) in seen:
                raise ValueError(f"Duplicate rollout: {record['id']} at seed {seed}.")
            seen.add((record["id"], seed))
            groups.setdefault(record["id"], []).append((seed, run, record))
    if len(adapters) > 1:
        raise ValueError(f"Runs come from different adapters {sorted(map(str, adapters))}; "
                         "one update uses one policy's samples.")

    rows, everything, by_skeleton = [], [], {}
    counts = {"mixed": 0, "all_pass": 0, "all_fail": 0}
    for task_id in sorted(groups):
        members = sorted(groups[task_id], key=lambda member: member[0])
        rewards = [reward(record) for _, _, record in members]
        everything += rewards
        skeleton = members[0][2]["skeleton"]
        stats = by_skeleton.setdefault(skeleton, {"tasks": 0, "mixed": 0, "rollouts": 0, "rewarded": 0})
        stats["tasks"] += 1
        stats["rollouts"] += len(rewards)
        stats["rewarded"] += sum(rewards)
        weights = advantages(rewards)
        if weights is None:
            counts["all_pass" if rewards[0] else "all_fail"] += 1
            continue
        counts["mixed"] += 1
        stats["mixed"] += 1
        for (seed, run, record), weight, earned in zip(members, weights, rewards):
            # A multi-turn task has one trace per turn, in order (`run_ids`).
            for turn, run_id in enumerate(record.get("run_ids", [record["run_id"]])):
                path = run / record["id"] / "state" / "runs" / f"{run_id}.jsonl"
                trace = json.loads(path.read_text(encoding="utf-8"))
                if trace["schema_version"] != 5:
                    raise ValueError(f"{run.name}/{record['id']}: a trace is not schema 5.")
                for row in sampled_rows(trace, template):
                    pair = row["prompt"] + "\0" + row["completion"]
                    rows.append({"id": sha256(pair.encode("utf-8")).hexdigest()[:16],
                                 "prompt": row["prompt"], "completion": row["completion"], "weight": weight,
                                 "provenance": {"task_id": task_id, "skeleton": skeleton, "seed": seed,
                                                "run_dir": run.name, "run_id": run_id, "turn": turn,
                                                "request_index": row["request_index"], "reward": earned}})
    return rows, {"rollouts": len(everything), "tasks": len(groups), **counts,
                  "mean_reward": round(sum(everything) / len(everything), 4) if everything else None,
                  "rows": len(rows), "adapter": next(iter(adapters), None),
                  "chat_template_sha256": template_sha256,
                  "by_skeleton": dict(sorted(by_skeleton.items()))}


def step(runs: list[Path], adapter_in: Path, out: Path, *, lr: float, base: str, revision: str) -> dict:
    """One GRPO iteration's update; writes `out` and returns what `out/step.json` holds.

    With no mixed group there is nothing to learn from: the input adapter and optimizer state
    are copied unchanged and the step is marked skipped, so the next iteration can go on.
    """
    started = time.time()
    rows, groups = collect_groups(runs)
    record = {"runs": [run.name for run in runs], "adapter_in": str(adapter_in),
              "adapter_in_sha256": sha256((adapter_in / ADAPTER_FILE).read_bytes()).hexdigest(),
              "resumed_optimizer": (adapter_in / OPTIMIZER_FILE).exists(), "lr": lr, "groups": groups}
    if not rows:
        shutil.copytree(adapter_in, out)
        return {**record, "skipped": True, "elapsed_seconds": round(time.time() - started, 1)}

    import peft
    from peft import PeftModel
    import torch
    import transformers

    from .tensorize import encode_example
    from .train import hf_tokenize, policy_step

    tokenizer = transformers.AutoTokenizer.from_pretrained(base, revision=revision)
    # `sampled_text` already decided whether a row ends with the stop token (only if the model stopped).
    encoded = [{**encode_example(row, hf_tokenize(tokenizer), max_tokens=BENCH_N_CTX, require_end=False),
                "weight": row["weight"]} for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    # As in `train`: recompute activations in the backward pass, or the longest rows do not fit.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    if record["resumed_optimizer"]:
        # A fresh AdamW taking one step moves every weight by about the rate (a sign update);
        # carrying its moments over makes consecutive iterations one optimizer run.
        optimizer.load_state_dict(torch.load(adapter_in / OPTIMIZER_FILE, map_location=device, weights_only=True))
    result = policy_step(model, optimizer, encoded, lr=lr)
    out.mkdir(parents=True)
    model.save_pretrained(out)
    torch.save(optimizer.state_dict(), out / OPTIMIZER_FILE)
    package = Path(__file__).resolve().parents[2]
    revision_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, text=True, capture_output=True,
                                   check=True).stdout.strip()
    return {**record, "skipped": False, "step": result,
            "adapter_sha256": sha256((out / ADAPTER_FILE).read_bytes()).hexdigest(),
            "base": base, "revision": revision, "device": device, "git_revision": revision_hash,
            "elapsed_seconds": round(time.time() - started, 1),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks", help="Print task IDs, comma-separated: the pool or one iteration")
    which = tasks.add_mutually_exclusive_group(required=True)
    which.add_argument("--pool", action="store_true", help="Every train task")
    which.add_argument("--pass", dest="pass_index", type=int, help="The pass (1-based), with --iteration")
    tasks.add_argument("--iteration", type=int, help="The iteration within the pass (0-based)")
    tasks.add_argument("--per-iteration", type=int, help="Default: 2 per skeleton, so 10 iterations make a pass")
    tasks.add_argument("--skeletons", help="Comma-separated train skeletons: the pool (default: the whole split)")
    update = commands.add_parser("step", help="One optimizer step on one iteration's runs")
    update.add_argument("--runs", type=Path, nargs="+", required=True, help="bench --split train runs, one per seed")
    update.add_argument("--adapter-in", type=Path, required=True, help="The adapter that sampled the runs")
    update.add_argument("--output", type=Path, required=True, help="New directory for the next adapter")
    update.add_argument("--lr", type=float, default=1e-5)
    update.add_argument("--base", default=BENCH_MODEL["id"], help="Hugging Face model ID or local path")
    update.add_argument("--revision", default=BENCH_MODEL["revision"])
    args = parser.parse_args()
    if args.command == "step":
        out = args.output.resolve()
        if out.exists():
            parser.error(f"{out} already exists; evidence is never overwritten.")
        if out.is_relative_to(Path(__file__).resolve().parents[2]):
            parser.error("Evidence must be outside the source package.")
        if not (args.adapter_in / ADAPTER_FILE).is_file():
            parser.error(f"{args.adapter_in} holds no {ADAPTER_FILE}.")
        try:
            record = step(args.runs, args.adapter_in, out, lr=args.lr, base=args.base, revision=args.revision)
        except (ValueError, OSError, KeyError) as error:
            parser.error(f"{type(error).__name__}: {error}")
        save(out / "step.json", record)
        summary = {"skipped": record["skipped"],
                   **{key: record["groups"][key] for key in ("mixed", "all_pass", "all_fail", "mean_reward", "rows")}}
        if not record["skipped"]:
            summary |= {key: round(record["step"][key], 4) for key in ("loss", "grad_norm")}
        print(json.dumps(summary))
        return
    try:
        pool = train_pool(args.skeletons.split(",") if args.skeletons else None)
    except ValueError as error:
        parser.error(str(error))
    if args.pool:
        ids = [task_id for task_ids in pool.values() for task_id in task_ids]
    else:
        if args.iteration is None:
            parser.error("--pass needs --iteration.")
        try:
            ids = iteration_tasks(pool, args.pass_index, args.iteration, args.per_iteration)
        except ValueError as error:
            parser.error(str(error))
    print(",".join(ids))


if __name__ == "__main__":
    main()
