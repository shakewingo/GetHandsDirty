"""Stage 10 shapes, rules and arithmetic (docs/STAGE10_HARD_TASKS_PLAN.md, section 3, Task 2).

Each builder takes a difficulty level 1-5. The deliverable is a file whose exact content the
stated rules determine, so a run passes only if every rule reached every record. The rules are
plain functions here, so the tests can check them on worked examples.
"""

import json
import random
from pathlib import Path

from .skeletons import SHAPES, Shape, _common, _esc, _observed, _record, _solution_update_records, _write
from .spec import Answer, BuildContext, Expect, Task

UPDATE_TOOLS = ("read_file", "write_file", "edit_file")
FINAL_SEEDS = tuple(range(20))


def _level(values: tuple, level: int):
    return values[level - 1]


def _lines(pairs: dict) -> str:
    """`key=value` lines sorted by key, as the shapes' output files are written."""
    return "".join(f"{key}={value}\n" for key, value in sorted(pairs.items()))


# --- ordered_migration (train): apply an ordered edit plan to one JSON file -------------------

MIGRATION_STEPS = (2, 4, 6, 9, 12)
NEW_KEYS = ("cap", "deadline", "attempts", "pool", "lead", "area", "grade", "labels", "sites", "quota",
            "window", "unit", "backup", "phase", "lane", "batch")


def apply_migration(data: dict, ops: list) -> dict:
    """`data` after each (kind, key, argument) op in order."""
    result = json.loads(json.dumps(data))
    for kind, key, argument in ops:
        if kind == "rename":
            result[argument] = result.pop(key)
        elif kind == "set":
            result[key] = argument
        elif kind == "delete":
            del result[key]
        elif kind == "multiply":
            result[key] = result[key] * argument
        elif kind == "append":
            result[key] = [*result[key], argument]
    return result


def _step_text(kind: str, key: str, argument) -> str:
    return {"rename": f"Rename the key {key} to {argument}.", "set": f'Set {key} to "{argument}".',
            "delete": f"Delete the key {key}.", "multiply": f"Multiply {key} by {argument}.",
            "append": f'Append "{argument}" to the list {key}.'}[kind]


def _migration_ops(rng: random.Random, data: dict, count: int) -> list:
    """`count` valid ops; a step often reuses a key an earlier step created or renamed."""
    fresh = list(NEW_KEYS)
    rng.shuffle(fresh)
    current, ops, touched = data, [], []
    for _ in range(count):
        eligible = {"rename": list(current),
                    "set": [k for k, v in current.items() if isinstance(v, str)],
                    "multiply": [k for k, v in current.items() if isinstance(v, int)],
                    "append": [k for k, v in current.items() if isinstance(v, list)],
                    "delete": list(current) if len(current) > 4 else []}
        kind = rng.choice([k for k, keys in eligible.items() if keys])
        recent = [k for k in touched if k in eligible[kind]]
        key = rng.choice(recent) if recent and rng.random() < 0.6 else rng.choice(eligible[kind])
        argument = {"rename": fresh.pop() if kind == "rename" else None,
                    "set": f"{key[:3]}-{rng.randint(10, 99)}", "multiply": rng.choice((2, 3)),
                    "append": f"v-{rng.randint(10, 99)}", "delete": None}[kind]
        ops.append((kind, key, argument))
        current = apply_migration(current, [ops[-1]])
        touched.append(argument if kind == "rename" else key)
    return ops


def _build_ordered_migration(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    data = {"limit": rng.randint(5, 50), "timeout": rng.randint(10, 90), "retries": rng.randint(1, 5),
            "workers": rng.randint(2, 16), "owner": f"team-{rng.randint(100, 999)}",
            "region": rng.choice(["north", "south", "east", "west"]), "tier": rng.choice(["basic", "plus", "pro"]),
            "tags": [f"t-{rng.randint(10, 99)}" for _ in range(2)], "zones": [f"z-{rng.randint(10, 99)}"]}
    ops = _migration_ops(rng, data, _level(MIGRATION_STEPS, level))
    record = _record(root, "data.json", data)
    plan = "".join(f"{index}. {_step_text(*op)}\n" for index, op in enumerate(ops, 1))
    _write(root, "plan.txt", plan)
    return Task(**_common(ctx),
                prompt=("Apply every step in plan.txt to data.json, in order: a later step may name a key that "
                        "an earlier step created or renamed. Keep every key the plan does not touch. Write the "
                        "result to data.json, read it back, and reply only MIGRATED."),
                tools=UPDATE_TOOLS,
                expect=Expect(answer=Answer("MIGRATED", "required"),
                              files={"data.json": json.dumps(apply_migration(data, ops))},
                              evidence=_observed(record) + (_esc(plan.splitlines()[-1]),)),
                claim_tokens=("MIGRATED",),
                debug={"reads": ["plan.txt", "data.json"], "level": level, "size": len(ops), "ops": ops})


# --- capped_allocation (train): divide a budget by ordered rules ------------------------------

ALLOCATION_TEAMS = (3, 4, 5, 6, 8)
TEAM_NAMES = ("alder", "birch", "cedar", "elm", "fir", "hazel", "larch", "maple", "oak", "pine")


def allocate(budget: int, requests: dict[str, int], *, cap: int | None = None, floor: int | None = None,
             priority: str | None = None, leftover: bool = False) -> dict[str, int]:
    """The allocation `policy.txt` describes, its rules applied in the order written there."""
    shares, remaining, others = {}, budget, dict(requests)
    if priority is not None:
        shares[priority] = min(requests[priority], budget)
        remaining -= shares[priority]
        del others[priority]
    total = sum(others.values())
    for team, request in others.items():
        share = remaining * request // total
        if cap is not None:
            share = min(share, cap)
        if floor is not None:
            share = max(share, min(floor, request))
        shares[team] = min(share, request)
    if leftover:
        left = budget - sum(shares.values())
        while left > 0:
            open_teams = [t for t in others if shares[t] < requests[t] and (cap is None or shares[t] < cap)]
            if not open_teams:
                break
            team = min(open_teams, key=lambda t: (shares[t] - requests[t], t))  # largest unmet, then name
            shares[team] += 1
            left -= 1
    return shares


def _build_capped_allocation(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    teams = sorted(rng.sample(TEAM_NAMES, _level(ALLOCATION_TEAMS, level)))
    while True:  # draw until the floor still fits the budget
        requests = {team: rng.randint(20, 120) for team in teams}
        total = sum(requests.values())
        budget = rng.randint(total // 2, total * 4 // 5)
        plain = [budget * r // total for r in requests.values()]
        rules = {"cap": max(plain) * 4 // 5 if level >= 2 else None,
                 "floor": min(plain) * 3 // 2 + 1 if level >= 3 else None,
                 "priority": rng.choice(teams) if level >= 4 else None, "leftover": level >= 5}
        if rules["cap"] is not None and rules["floor"] is not None and rules["floor"] > rules["cap"]:
            continue
        shares = allocate(budget, requests, **rules)
        if sum(shares.values()) <= budget:
            break
    lines = [f"budget={budget}", "Divide the budget across the teams in requests.json. Apply these rules in order:"]
    if rules["priority"]:
        lines.append(f"- Team {rules['priority']} is served first and receives its full request. The rest of "
                     "the budget is divided among the other teams by the rules below.")
    share = ("the remaining budget times its request divided by the other teams' total request" if rules["priority"]
             else "the budget times its request divided by the total request")
    lines.append(f"- Each team receives {share}, rounded down to a whole number.")
    if rules["cap"] is not None:
        lines.append(f"- No team{' other than the priority team' if rules['priority'] else ''} receives more "
                     f"than {rules['cap']}.")
    if rules["floor"] is not None:
        lines.append(f"- A team receives at least {rules['floor']}, or its full request if that is smaller.")
    lines.append("- No team receives more than it requested.")
    if rules["leftover"]:
        lines.append("- Then hand out what is left of the budget one unit at a time, each to the team with the "
                     "largest unmet request (its request minus what it has; ties go to the alphabetically first "
                     "team), never past a team's request or the cap, until nothing is left or no team can take more.")
    policy = "\n".join(lines) + "\n"
    _write(root, "policy.txt", policy)
    record = _record(root, "requests.json", {"requests": requests})
    return Task(**_common(ctx),
                prompt=("Read policy.txt and requests.json, and allocate the budget by the policy's rules. Write "
                        "allocations.txt with one line per team, team=amount, sorted by team name; read it back "
                        "and reply only ALLOCATED."),
                tools=(*UPDATE_TOOLS, "calculator"),
                expect=Expect(answer=Answer("ALLOCATED", "required"), files={"allocations.txt": _lines(shares)},
                              evidence=_observed(record) + (f"budget={budget}",)),
                claim_tokens=("ALLOCATED",),
                debug={"reads": ["policy.txt", "requests.json"], "level": level, "size": len(teams), "rules": rules})


# --- rule tables: first match wins (rule_table_apply, train; payroll_rule_adjust, hardtest) ---

RULE_LEVELS = ((2, 3), (3, 5), (4, 8), (6, 10), (8, 14))  # (rules including the default, records)


def first_match(record: dict, rules: list) -> int:
    """The outcome of the first rule whose every (field, op, value) holds; the last rule is the default."""
    for conditions, outcome in rules:
        if all({"is": record[f] == v, "over": record[f] > v, "under": record[f] < v,
                "at_least": record[f] >= v}[op] for f, op, v in conditions):
            return outcome
    raise ValueError("The rules have no default.")


def _condition_text(field: str, op: str, value) -> str:
    return {"is": f"{field} is {value}", "over": f"{field} is more than {value}",
            "under": f"{field} is less than {value}", "at_least": f"{field} is at least {value}"}[op]


def _rule_lines(rules: list, outcome_text) -> str:
    """rules.txt: numbered rules, the default last; `outcome_text(outcome)` words an outcome."""
    lines = [f"{index}. If {' and '.join(_condition_text(*c) for c in conditions)}: {outcome_text(outcome)}."
             for index, (conditions, outcome) in enumerate(rules[:-1], 1)]
    lines.append(f"{len(rules)}. Otherwise: {outcome_text(rules[-1][1])}.")
    return "\n".join(lines) + "\n"


def _overlap(records: list, rules: list) -> bool:
    """Some record matches two non-default rules with different outcomes, so the order matters."""
    for record in records:
        hits = {outcome for conditions, outcome in rules[:-1] if first_match(record, [(conditions, outcome), ((), None)])}
        if len(hits) > 1:
            return True
    return False


def _draw_rules(rng: random.Random, count: int, fields: dict, outcomes: tuple, default) -> list:
    """`count - 1` rules of one or two conditions over `fields` (name -> (ops, values)), then the default."""
    rules = []
    for _ in range(count - 1):
        names = rng.sample(sorted(fields), rng.choice((1, 2)))
        conditions = []
        for name in names:
            ops, values = fields[name]
            conditions.append((name, rng.choice(ops), rng.choice(values)))
        rules.append((tuple(conditions), rng.choice(outcomes)))
    return rules + [((), default)]


def _build_rule_table_apply(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    count, size = _level(RULE_LEVELS, level)
    fields = {"tier": (("is",), ("gold", "silver", "bronze")), "visits": (("over", "under"), (5, 10, 15, 20)),
              "region": (("is",), ("north", "south", "east"))}
    ids = [f"C-{number}" for number in rng.sample(range(100, 999), size)]
    customers = [{"id": i, "tier": rng.choice(fields["tier"][1]), "visits": rng.randint(0, 30),
                  "region": rng.choice(fields["region"][1])} for i in ids]
    for _ in range(100):  # from level 3 on, the order of the rules must change some outcome
        rules = _draw_rules(rng, count, fields, (5, 10, 15, 20, 25, 30), 0)
        if level < 3 or _overlap(customers, rules):
            break
    _write(root, "rules.txt", _rule_lines(rules, lambda outcome: f"discount={outcome}"))
    record = _record(root, "customers.json", {"customers": customers})
    discounts = {c["id"]: first_match(c, rules) for c in customers}
    return Task(**_common(ctx),
                prompt=("Read rules.txt and customers.json. For each customer, the first rule whose conditions all "
                        "hold gives the discount; later rules do not apply to that customer. Write discounts.txt "
                        "with one line per customer, id=discount, sorted by id; read it back and reply only APPLIED."),
                tools=UPDATE_TOOLS,
                expect=Expect(answer=Answer("APPLIED", "required"), files={"discounts.txt": _lines(discounts)},
                              evidence=_observed(record) + (f"{len(rules)}. Otherwise: discount=0.",)),
                claim_tokens=("APPLIED",),
                debug={"reads": ["rules.txt", "customers.json"], "level": level, "size": count * size, "rules": rules})


PAYROLL_CAPS = (None, 400, 400, 350, 300)
PAYROLL_BUDGET_FROM = 4


def adjust_pay(employees: list, rules: list, *, cap: int | None, budgets: dict[str, int]) -> dict[str, int]:
    """New salaries: the first matching rule's percent, rounded down, at most `cap`; a department
    with a budget grants raises by rating (highest first, ties by id) and gives 0 to one that no longer fits."""
    raises = {}
    for e in employees:
        amount = e["salary"] * first_match(e, rules) // 100
        raises[e["id"]] = min(amount, cap) if cap is not None else amount
    for dept, budget in budgets.items():
        left = budget
        for e in sorted((e for e in employees if e["dept"] == dept), key=lambda e: (-e["rating"], e["id"])):
            if raises[e["id"]] <= left:
                left -= raises[e["id"]]
            else:
                raises[e["id"]] = 0
    return {e["id"]: e["salary"] + raises[e["id"]] for e in employees}


def _build_payroll_rule_adjust(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    count, size = _level(RULE_LEVELS, level)
    fields = {"dept": (("is",), ("ops", "sales", "eng")), "rating": (("at_least", "under"), (2, 3, 4, 5))}
    ids = [f"E-{number}" for number in rng.sample(range(100, 999), size)]
    employees = [{"id": i, "dept": rng.choice(fields["dept"][1]), "rating": rng.randint(1, 5),
                  "salary": rng.randint(30, 90) * 100} for i in ids]
    rules = _draw_rules(rng, count, fields, (3, 4, 5, 6, 8), 1)
    cap = _level(PAYROLL_CAPS, level)
    budgets = {}
    if level >= PAYROLL_BUDGET_FROM:
        dept = rng.choice(fields["dept"][1])
        planned = adjust_pay([e for e in employees if e["dept"] == dept], rules, cap=cap, budgets={})
        wanted = sum(planned[i] for i in planned) - sum(e["salary"] for e in employees if e["dept"] == dept)
        budgets = {dept: max(1, wanted * 3 // 5)}
    _write(root, "rules.txt", _rule_lines(rules, lambda outcome: f"raise {outcome}%"))
    limits = []
    if cap is not None:
        limits.append(f"No raise is more than {cap}.")
    for dept, budget in budgets.items():
        limits.append(f"Raises in {dept} may total at most {budget}: grant them in order of rating (highest first, "
                      "ties by id), and a raise that no longer fits becomes 0.")
    _write(root, "limits.txt", "\n".join(limits) + "\n" if limits else "No limits.\n")
    record = _record(root, "employees.json", {"employees": employees})
    salaries = adjust_pay(employees, rules, cap=cap, budgets=budgets)
    return Task(**_common(ctx),
                prompt=("Read rules.txt, limits.txt and employees.json. Each employee's raise is the first matching "
                        "rule's percent of their salary, rounded down to a whole number; then apply limits.txt. "
                        "Write new_salaries.txt with one line per employee, id=new salary, sorted by id; read it "
                        "back and reply only ADJUSTED."),
                tools=(*UPDATE_TOOLS, "calculator"),
                expect=Expect(answer=Answer("ADJUSTED", "required"), files={"new_salaries.txt": _lines(salaries)},
                              evidence=_observed(record)),
                claim_tokens=("ADJUSTED",),
                debug={"reads": ["rules.txt", "limits.txt", "employees.json"], "level": level, "size": count * size})


# --- transfer_to_targets (val): balance stock against targets by a stated order ---------------

TRANSFER_LEVELS = ((3, 1), (4, 1), (5, 2), (6, 2), (8, 3))  # (warehouses, items)
WAREHOUSES = ("akron", "boise", "camden", "dover", "eugene", "fresno", "gary", "helena", "irvine", "joliet")
ITEMS = ("bolts", "cable", "gears")


def plan_transfers(stock: dict, targets: dict) -> list[tuple[str, str, str, int]]:
    """(item, from, to, quantity) moves: per item in name order, the largest surplus goes to the
    largest deficit (ties to the alphabetically first warehouse), until every warehouse is at target."""
    moves = []
    for item in sorted(next(iter(stock.values()))):
        balance = {w: stock[w][item] - targets[w][item] for w in stock}
        while any(balance.values()):
            source = min((w for w in balance if balance[w] > 0), key=lambda w: (-balance[w], w))
            sink = min((w for w in balance if balance[w] < 0), key=lambda w: (balance[w], w))
            quantity = min(balance[source], -balance[sink])
            moves.append((item, source, sink, quantity))
            balance[source] -= quantity
            balance[sink] += quantity
    return moves


def _build_transfer_to_targets(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    count, kinds = _level(TRANSFER_LEVELS, level)
    places, items = sorted(rng.sample(WAREHOUSES, count)), ITEMS[:kinds]
    targets = {w: {i: rng.randint(10, 60) for i in items} for w in places}
    moves = []
    while not moves:  # move stock away from its targets, keeping each item's total
        stock = json.loads(json.dumps(targets))
        for item in items:
            for _ in range(count):
                a, b = rng.sample(places, 2)
                amount = rng.randint(1, max(1, stock[a][item]))
                if amount <= stock[a][item]:
                    stock[a][item] -= amount
                    stock[b][item] += amount
        moves = plan_transfers(stock, targets)
    stock_record = _record(root, "stock.json", stock)
    target_record = _record(root, "targets.json", targets)
    transfers = "".join(f"{item}: {source} -> {sink} {quantity}\n" for item, source, sink, quantity in moves)
    return Task(**_common(ctx),
                prompt=("Read stock.json and targets.json; each item's total stock equals its total target. For each "
                        "item in name order, repeatedly move stock from the warehouse with the largest surplus to the "
                        "warehouse with the largest deficit (ties go to the alphabetically first warehouse), moving "
                        "as much as both allow, until every warehouse is at its target. Write transfers.txt with one "
                        "line per move, in the order made: item: from -> to quantity. Read it back and reply only "
                        "TRANSFERRED."),
                tools=(*UPDATE_TOOLS, "calculator"),
                expect=Expect(answer=Answer("TRANSFERRED", "required"), files={"transfers.txt": transfers},
                              evidence=_observed(stock_record, target_record)),
                claim_tokens=("TRANSFERRED",),
                debug={"reads": ["stock.json", "targets.json"], "level": level, "size": count * kinds})


RULE_SHAPES = [
    Shape("ordered_migration", "updates", "train", FINAL_SEEDS, False, _build_ordered_migration,
          _solution_update_records),
    Shape("capped_allocation", "updates", "train", FINAL_SEEDS, False, _build_capped_allocation,
          _solution_update_records),
    Shape("rule_table_apply", "updates", "train", FINAL_SEEDS, False, _build_rule_table_apply,
          _solution_update_records),
    Shape("transfer_to_targets", "updates", "val", FINAL_SEEDS, False, _build_transfer_to_targets,
          _solution_update_records),
    Shape("payroll_rule_adjust", "updates", "hardtest", FINAL_SEEDS, False, _build_payroll_rule_adjust,
          _solution_update_records),
]

SHAPES.extend(RULE_SHAPES)
