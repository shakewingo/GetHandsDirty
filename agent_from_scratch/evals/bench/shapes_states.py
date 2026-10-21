"""Stage 10 shapes, mixed states and faults (docs/STAGE10_HARD_TASKS_PLAN.md, section 3, Task 4).

A folder of records, each needing its own decision: change it, leave it (already right), or
leave it and report it (locked, frozen, pinned). Two shapes add a fault that the run must meet
and retry. The scripted solution is planned as a list of steps, and the faulted call's step is
repeated where it falls, so the solution retries exactly the call the fault hits.
"""

from datetime import date, timedelta
import random
from pathlib import Path

from .skeletons import SHAPES, Shape, _common, _paired_rng, _write, answer, call
from .spec import Answer, BuildContext, Expect, Fault, Task

FINAL_SEEDS, FINAL_RECOVERY_SEEDS = tuple(range(20)), tuple(range(10))
BATCH_SIZES = (3, 5, 7, 9, 12)
READ_FAILURE = "Temporary read failure; the file was not read. Read it again."
SAVE_FAILURE = "Temporary save failure; the file was not changed. Save it again."


def _plan(steps: list, fault: Fault | None) -> list:
    """`steps` with the faulted call repeated right after it: the retry."""
    planned, counts = [], {}
    for step in steps:
        planned.append(step)
        tool = {"list": "list_files", "read": "read_file", "write": "write_file"}.get(step[0])
        counts[tool] = counts.get(tool, 0) + 1
        if fault is not None and fault.tool == tool and counts[tool] == fault.on_call:
            planned.append(step)
    return planned


def _solution_planned(task: Task) -> list:
    responses = []
    for step in task.debug["script"]:
        if step[0] == "list":
            responses.append(call("list_files", path=step[1]))
        elif step[0] == "read":
            responses.append(call("read_file", path=step[1]))
        elif step[0] == "write":
            responses.append(call("write_file", path=step[1], content=step[2]))
        else:
            responses.append(answer(step[1]))
    return responses


def _states(rng: random.Random, count: int, level: int) -> list[str]:
    """One of "ok", "fix", "hold" per record: at least one fix, and from level 2 at least one of each."""
    holds = max(1 if level >= 2 else 0, count // 4)
    oks = max(1 if level >= 2 else 0, count // 3)
    states = ["hold"] * holds + ["ok"] * oks + ["fix"] * (count - holds - oks)
    rng.shuffle(states)
    return states


def _batch_task(ctx: BuildContext, level: int, *, folder: str, files: dict[str, str], fixed: dict[str, str],
                held: list[str], prompt: str, empty: str, tools: tuple, extra_reads: tuple = (),
                fault: Fault | None = None, signal: str | None = None, evidence: tuple = (),
                claim: str | None = None, max_iterations: int = 12) -> Task:
    """A task over `files` (path -> content); `fixed` are the expected new or changed files, `held`
    the ids to report, and `empty` the reply when there are none. The solution lists, reads, writes
    each fixed file and reads it back, then replies.

    Evidence is checked before the first write (Stage 9 strict scoring), so a shape that may fix
    records as it reads them carries none: its answer and its files already need every read.
    """
    steps = [("list", folder), *(("read", p) for p in extra_reads), *(("read", p) for p in files)]
    for path, content in fixed.items():
        steps += [("write", path, content), ("read", path)]
    reply = ",".join(sorted(held)) or empty
    return Task(**_common(ctx), prompt=prompt, tools=tools,
                expect=Expect(answer=Answer(reply, "required"), files=fixed, evidence=evidence),
                fault=fault, fault_signal=(fault.tool, signal) if fault is not None else None,
                claim_tokens=(claim,) if claim else (), max_iterations=max_iterations,
                debug={"level": level, "size": len(files), "held": sorted(held),
                       "script": _plan(steps, fault) + [("answer", reply)]})


# --- mixed_state_batch (train): fix the wrong unlocked accounts, report the locked ones --------

PLANS = ("basic", "silver", "gold", "platinum")


def _build_mixed_state_batch(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    count = BATCH_SIZES[level - 1]
    target = rng.choice(PLANS)
    others = [p for p in PLANS if p != target]
    files, fixed, held = {}, {}, []
    for state, number in zip(_states(rng, count, level), rng.sample(range(100, 999), count)):
        account, plan = f"A-{number}", target if state == "ok" else rng.choice(others)
        content = f"id={account}\nplan={plan}\nlocked={'yes' if state == 'hold' else 'no'}\n"
        path = f"accounts/{account}.txt"
        _write(root, path, content)
        files[path] = content
        if state == "fix":
            fixed[path] = content.replace(f"plan={plan}", f"plan={target}")
        elif state == "hold":
            held.append(account)
    _write(root, "target.txt", f"plan={target}\n")
    return _batch_task(
        ctx, level, folder="accounts", files=files, fixed=fixed, held=held, empty="NONE",
        prompt=("Every account in accounts/ should have the plan in target.txt. List accounts/, read target.txt "
                "and each account, and in every account whose plan differs and that is not locked (locked=yes), "
                "set the plan to the target, keeping every other line. Leave accounts that already match, and "
                "locked ones, untouched. Then reply only the ids of the locked accounts whose plan differs, "
                "comma-separated and sorted, or NONE if there are none."),
        tools=("list_files", "read_file", "write_file", "edit_file"), extra_reads=("target.txt",),
        evidence=(f"plan={target}",), claim=None, max_iterations=8 + 2 * count)


# --- fault_amid_reconcile (train, clean/fault): a read fails inside a long scan -----------------

def _build_fault_amid_reconcile(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    rng = _paired_rng(rng, ctx)
    count = BATCH_SIZES[level - 1]
    skus = sorted(f"S-{number}" for number in rng.sample(range(100, 999), count))
    expected = {sku: rng.randint(5, 60) for sku in skus}
    differing = set(rng.sample(skus, max(1, count // 3)))
    counted = {sku: expected[sku] + (rng.choice((-1, 1)) * rng.randint(1, 4) if sku in differing else 0)
               for sku in skus}
    ledger = "sku,expected\n" + "".join(f"{sku},{expected[sku]}\n" for sku in skus)
    _write(root, "ledger.csv", ledger)
    files = {}
    # One line per stock file: read_file numbers the lines it returns, so evidence is a single line.
    for sku in rng.sample(skus, count):  # the listing order is not the ledger's
        files[f"stock/{sku}.txt"] = content = f"sku={sku} count={counted[sku]}\n"
        _write(root, f"stock/{sku}.txt", content)
    report = "".join(f"{sku}: expected {expected[sku]}, counted {counted[sku]}\n" for sku in skus if sku in differing)
    # The fault moves later with the level: from the first stock read to the last.
    fault = (Fault("read_file", 1 + min(count, max(1, count * level // 5)), READ_FAILURE)
             if ctx.condition == "fault" else None)
    # The report is written once, after every read, so every source must be seen first.
    return _batch_task(
        ctx, level, folder="stock", files=files, fixed={"discrepancies.txt": report}, held=[], empty="RECONCILED",
        prompt=("ledger.csv lists each SKU's expected count, and stock/ holds one counted file per SKU. List "
                "stock/, read ledger.csv and every stock file, and write discrepancies.txt with one line per SKU "
                "whose counted number differs from the ledger, as SKU: expected N, counted M, sorted by SKU. If a "
                "read fails, read that file again. Read discrepancies.txt back and reply only RECONCILED."),
        tools=("list_files", "read_file", "write_file"), extra_reads=("ledger.csv",), fault=fault,
        signal=READ_FAILURE, evidence=(*ledger.splitlines()[1:], *(c.strip() for c in files.values())),
        claim="RECONCILED", max_iterations=8 + 2 * count)


# --- flag_rollout (val): three decisions per service, and a save that fails once ---------------

def _build_flag_rollout(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    count = BATCH_SIZES[level - 1]
    states = _states(rng, count, max(level, 2))  # val always has a frozen service to report
    names = [f"svc-{number}" for number in rng.sample(range(100, 999), count)]
    files, fixed, held = {}, {}, []
    for state, name in zip(states, names):
        content = (f"name={name}\nflag_beta={'on' if state == 'ok' else 'off'}\n"
                   f"frozen={'yes' if state == 'hold' else 'no'}\n")
        path = f"services/{name}.conf"
        _write(root, path, content)
        files[path] = content
        if state == "fix":
            fixed[path] = content.replace("flag_beta=off", "flag_beta=on")
        elif state == "hold":
            held.append(name)
    fault = Fault("write_file", 1 + len(fixed) // 2, SAVE_FAILURE) if level >= 3 else None
    return _batch_task(
        ctx, level, folder="services", files=files, fixed=fixed, held=held, empty="NONE",
        prompt=("Turn flag_beta on across services/: in each service file where flag_beta is off and frozen is no, "
                "set flag_beta=on, keeping every other line. Leave services that are already on, and frozen ones, "
                "untouched. If a save fails, save that file again. Then reply only the names of the frozen services "
                "whose flag is still off, comma-separated and sorted."),
        tools=("list_files", "read_file", "write_file", "edit_file"), fault=fault, signal=SAVE_FAILURE,
        max_iterations=8 + 2 * count)


# --- cert_renewal_batch (hardtest): renew what expires soon, skip pinned, a read fails once ----

TODAY = date(2026, 11, 1)


def _build_cert_renewal_batch(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    count = BATCH_SIZES[level - 1]
    states = _states(rng, count, max(level, 2))
    hosts = [f"host-{number}" for number in rng.sample(range(100, 999), count)]
    files, fixed, held = {}, {}, []
    for state, host in zip(states, hosts):
        expires = TODAY + timedelta(days=rng.randint(40, 200) if state == "ok" else rng.randint(-5, 25))
        content = f"host={host}\nexpires={expires.isoformat()}\npinned={'yes' if state == 'hold' else 'no'}\n"
        path = f"certs/{host}.txt"
        _write(root, path, content)
        files[path] = content
        if state == "fix":
            fixed[path] = content.replace(expires.isoformat(), (TODAY + timedelta(days=90)).isoformat())
        elif state == "hold":
            held.append(host)
    fault = Fault("read_file", 1 + (count + 1) // 2, READ_FAILURE) if level >= 2 else None
    return _batch_task(
        ctx, level, folder="certs", files=files, fixed=fixed, held=held, empty="NONE",
        prompt=(f"Today is {TODAY.isoformat()}. In certs/, renew every certificate that expires within 30 days of "
                f"today (or has expired) and is not pinned: set its expires date to {(TODAY + timedelta(days=90)).isoformat()}, "
                "keeping every other line. Leave later certificates, and pinned ones, untouched. If a read fails, "
                "read that file again. Then reply only the hosts of the pinned certificates that need renewal, "
                "comma-separated and sorted."),
        tools=("list_files", "read_file", "write_file", "edit_file"), fault=fault, signal=READ_FAILURE,
        max_iterations=8 + 2 * count)


STATE_SHAPES = [
    Shape("mixed_state_batch", "stopping", "train", FINAL_SEEDS, False, _build_mixed_state_batch, _solution_planned),
    Shape("fault_amid_reconcile", "recovery", "train", FINAL_RECOVERY_SEEDS, True, _build_fault_amid_reconcile,
          _solution_planned),
    Shape("flag_rollout", "stopping", "val", FINAL_SEEDS, False, _build_flag_rollout, _solution_planned),
    Shape("cert_renewal_batch", "stopping", "hardtest", FINAL_SEEDS, False, _build_cert_renewal_batch,
          _solution_planned),
]

SHAPES.extend(STATE_SHAPES)
