"""Stage 10 shapes, multi-turn dialogues (docs/STAGE10_HARD_TASKS_PLAN.md, section 3, Task 3).

One structure in three domains. Turn 1 asks about a group of two records: the act side (even
seed) asks for the one that stands out, the ask side (odd seed) for both. Then level - 1 turns
edit other, named records. The last turn edits "the one from my first question": clear on the
act side; ambiguous on the ask side, where the right reply changes nothing in that turn and
gives the ask token. A pair (seeds 2k, 2k + 1) shares its records; only turn 1's wording and
the outcome differ, so the referent carried across turns is the only thing that decides.
"""

from dataclasses import dataclass
from datetime import date, timedelta
import random
from pathlib import Path

from .skeletons import SHAPES, Shape, _common, _observed, _pair_rng, _record, _write, answer, call
from .spec import Answer, BuildContext, Expect, Task

UPDATE_TOOLS = ("read_file", "write_file", "edit_file")
FINAL_SEEDS = tuple(range(20))  # ten act/ask pairs


@dataclass(frozen=True)
class Words:
    """A domain's file names and sentences; {names} is the group as the question gives it."""

    source: str  # JSON records, read in turn 1
    target: str  # the text file the edits change
    act_first: str  # format(group=...): reply the one record
    ask_first: str  # format(group=...): reply both
    between: str  # format(name=..., value=...): an edit of a named record
    final: str  # format(value=...): an edit of the record from turn 1
    between_token: str
    act_token: str
    ask_token: str


def _dialogue(ctx: BuildContext, root: Path, level: int, words: Words, records: list, group: str,
              chosen: str, pair: list[str], state: dict, render, edits: list, final_edit) -> Task:
    """Assemble the turns, the expected final file and the scripted solution.

    `state` maps a record name to its editable value, `render(state)` writes the target file,
    `edits` are level - 1 (name, value_text, new_value) edits of records outside the pair, and `final_edit` is
    (value_text, new_value) for the record from turn 1.
    """
    act = ctx.seed % 2 == 0
    source = _record(root, words.source, {"records": records})
    before = render(state)
    _write(root, words.target, before)
    first = (words.act_first if act else words.ask_first).format(group=group)
    turns = [first]
    script = [[("read", words.source), ("answer", chosen if act else ",".join(sorted(pair)))]]
    for name, value_text, new_value in edits:
        turns.append(words.between.format(name=name, value=value_text))
        state = {**state, name: new_value}
        script.append([("read", words.target), ("write", words.target, render(state)),
                       ("read", words.target), ("answer", words.between_token)])
    value_text, new_value = final_edit
    turns.append(words.final.format(value=value_text))
    if act:
        state = {**state, chosen: new_value}
        script.append([("read", words.target), ("write", words.target, render(state)),
                       ("read", words.target), ("answer", words.act_token)])
    else:
        script.append([("read", words.target), ("answer", words.ask_token)])
    changed = render(state) != before
    return Task(**_common(ctx), prompt=turns[0], followups=tuple(turns[1:]), tools=UPDATE_TOOLS,
                expect=Expect(answer=Answer(words.act_token if act else words.ask_token, "required"),
                              files={words.target: render(state)} if changed else {},
                              evidence=_observed(source),
                              process=() if changed else ("no_write_attempts",)),
                claim_tokens=(words.act_token,), max_iterations=12,
                debug={"outcome": "act" if act else "ask", "level": level, "size": len(turns),
                       "chosen": chosen, "pair": sorted(pair), "script": script})


def _solution_dialogue(task: Task) -> list:
    """Each turn's reads, write and read-back, then its reply, as the builder planned them."""
    responses = []
    for turn in task.debug["script"]:
        for step in turn:
            if step[0] == "read":
                responses.append(call("read_file", path=step[1]))
            elif step[0] == "write":
                responses.append(call("write_file", path=step[1], content=step[2]))
            else:
                responses.append(answer(step[1]))
    return responses


def _groups(rng: random.Random, names: list[str], labels: tuple) -> tuple[dict[str, str], str]:
    """Each label to two of the six `names`; returns (name -> label, the turn-1 label)."""
    shuffled = rng.sample(names, len(names))
    label_of = {name: labels[index // 2] for index, name in enumerate(shuffled)}
    return label_of, rng.choice(labels)


# --- long_edit_dialogue (train): a shopping cart -----------------------------------------------

CART_WORDS = Words(
    source="catalog.json", target="cart.txt",
    act_first="Read catalog.json and reply only the name of the cheapest {group} item.",
    ask_first="Read catalog.json and reply only the names of the two {group} items, comma-separated.",
    between="Set the quantity of {name} in cart.txt to {value}, keeping every other line, and reply only SET.",
    final=("Set the quantity of the item from my first question to {value} in cart.txt, keeping every other "
           "line; read it back and reply UPDATED. If it is not clear which single item I mean, change nothing "
           "in this step and reply only NEED_INPUT."),
    between_token="SET", act_token="UPDATED", ask_token="NEED_INPUT")
ITEM_NAMES = ("lamp", "mug", "scarf", "kettle", "clock", "towel", "vase", "pillow", "jar", "fan")


def _build_long_edit_dialogue(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    rng = _pair_rng(ctx)
    names = rng.sample(ITEM_NAMES, 6)
    color_of, group = _groups(rng, names, ("red", "blue", "green"))
    prices = dict(zip(names, rng.sample(range(5, 80), 6)))
    records = [{"name": n, "color": color_of[n], "price": prices[n]} for n in names]
    pair = [n for n in names if color_of[n] == group]
    chosen = min(pair, key=prices.get)
    others = [n for n in names if n not in pair]
    state = {n: rng.randint(1, 5) for n in names}
    edits = []
    for index in range(level - 1):
        name = others[index % len(others)]
        value = rng.choice([q for q in range(0, 9) if q != state[name]])
        edits.append((name, str(value), value))
    final = rng.choice([q for q in range(0, 9) if q != state[chosen]])
    return _dialogue(ctx, root, level, CART_WORDS, records, group, chosen, pair, state,
                     lambda s: "".join(f"{n}={s[n]}\n" for n in names), edits, (str(final), final))


# --- itinerary_dialogue (val): a trip's stops and dates ----------------------------------------

TRIP_WORDS = Words(
    source="trip.json", target="itinerary.txt",
    act_first=("Read trip.json and reply only the city of the earlier of the two stops where I stay {group} "
               "nights."),
    ask_first="Read trip.json and reply only the two cities where I stay {group} nights, comma-separated.",
    between="Move {name} {value} in itinerary.txt, keeping every other line, and reply only OK.",
    final=("Move the stop from my first question {value} in itinerary.txt, keeping every other line; read it "
           "back and reply MOVED. If it is not clear which single stop I mean, change nothing in this step and "
           "reply only UNSURE."),
    between_token="OK", act_token="MOVED", ask_token="UNSURE")
TRIP_CITIES = ("Lisbon", "Porto", "Seville", "Lyon", "Turin", "Ghent", "Bruges", "Basel")


def _shift(days: int) -> str:
    return f"{abs(days)} day{'s' if abs(days) > 1 else ''} {'later' if days > 0 else 'earlier'}"


def _build_itinerary_dialogue(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    rng = _pair_rng(ctx)
    cities = rng.sample(TRIP_CITIES, 6)
    nights_of, group = _groups(rng, cities, ("2", "3", "4"))
    start, day = date(2026, 11, 1), 0
    arrivals = {}
    for city in cities:  # in trip order: each stop starts when the one before it ends
        arrivals[city] = start + timedelta(days=day)
        day += int(nights_of[city])
    records = [{"city": c, "arrive": arrivals[c].isoformat(), "nights": int(nights_of[c])} for c in cities]
    pair = [c for c in cities if nights_of[c] == group]
    chosen = min(pair, key=arrivals.get)
    others = [c for c in cities if c not in pair]
    edits, moved = [], dict(arrivals)
    for index in range(level - 1):  # a stop moved twice moves from where the first edit left it
        city, days = others[index % len(others)], rng.choice((1, 2, 3))
        moved[city] += timedelta(days=days)
        edits.append((city, _shift(days), moved[city]))
    days = -rng.choice((1, 2))  # the turn-1 stop is never moved in between
    return _dialogue(ctx, root, level, TRIP_WORDS, records, group, chosen, pair, arrivals,
                     lambda s: "".join(f"{c} {s[c].isoformat()}\n" for c in cities), edits,
                     (_shift(days), arrivals[chosen] + timedelta(days=days)))


# --- support_dialogue (hardtest): support tickets ---------------------------------------------

TICKET_WORDS = Words(
    source="tickets.json", target="tickets.txt",
    act_first="Read tickets.json and reply only the id of the higher-priority of the two tickets from {group}.",
    ask_first="Read tickets.json and reply only the ids of the two tickets from {group}, comma-separated.",
    between="Set the status of {name} in tickets.txt to {value}, keeping every other line, and reply only NOTED.",
    final=("Set the status of the ticket from my first question to {value} in tickets.txt, keeping every other "
           "line; read it back and reply ESCALATED. If it is not clear which single ticket I mean, change "
           "nothing in this step and reply only UNCLEAR."),
    between_token="NOTED", act_token="ESCALATED", ask_token="UNCLEAR")
CUSTOMERS = ("Acme", "Birchwood", "Corvid")


def _build_support_dialogue(rng: random.Random, root: Path, ctx: BuildContext, level: int) -> Task:
    rng = _pair_rng(ctx)
    ids = [f"T-{number}" for number in rng.sample(range(100, 999), 6)]
    customer_of, group = _groups(rng, ids, CUSTOMERS)
    priorities = dict(zip(ids, rng.sample(range(1, 9), 6)))
    records = [{"id": i, "customer": customer_of[i], "priority": priorities[i]} for i in ids]
    pair = [i for i in ids if customer_of[i] == group]
    chosen = max(pair, key=priorities.get)
    others = [i for i in ids if i not in pair]
    state = {i: "open" for i in ids}
    edits = []
    for index in range(level - 1):
        ticket = others[index % len(others)]
        status = rng.choice([s for s in ("pending", "closed", "waiting") if s != state[ticket]])
        edits.append((ticket, status, status))
    return _dialogue(ctx, root, level, TICKET_WORDS, records, group, chosen, pair, state,
                     lambda s: "".join(f"{i} status={s[i]}\n" for i in ids), edits, ("escalated", "escalated"))


DIALOGUE_SHAPES = [
    Shape("long_edit_dialogue", "updates", "train", FINAL_SEEDS, False, _build_long_edit_dialogue,
          _solution_dialogue),
    Shape("itinerary_dialogue", "updates", "val", FINAL_SEEDS, False, _build_itinerary_dialogue,
          _solution_dialogue),
    Shape("support_dialogue", "updates", "hardtest", FINAL_SEEDS, False, _build_support_dialogue,
          _solution_dialogue),
]

SHAPES.extend(DIALOGUE_SHAPES)
