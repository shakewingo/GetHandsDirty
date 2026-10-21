"""Stage 10 rule and arithmetic shapes: every level keeps the gates, the rules hold on worked
examples, and the slips each shape targets (a skipped step, an off-by-one, last match wins) fail."""

import json
import unittest

from agent_from_scratch.evals.bench.shapes_rules import (RULE_SHAPES, adjust_pay, allocate, apply_migration,
                                                         first_match, plan_transfers)
from agent_from_scratch.evals.bench.skeletons import answer, call
from agent_from_scratch.tests.shape_harness import ShapeGates, calib


class RuleShapeTests(ShapeGates, unittest.TestCase):
    SHAPE_NAMES = tuple(shape.name for shape in RULE_SHAPES)

    def deliver(self, name, level, path, content, token):
        """Read the sources, write `content` to `path`, read it back, claim `token`."""
        skeleton, ctx = calib(name, level)
        _, task = self.build(skeleton, ctx)
        script = [call("read_file", path=p) for p in task.debug["reads"]]
        script += [call("write_file", path=path, content=content), call("read_file", path=path), answer(token)]
        return task, self.run_script(skeleton, ctx, script)

    def test_a_migration_that_skips_its_last_step_fails(self):
        root, task = self.build(*calib("ordered_migration", 5))
        partial = apply_migration(json.loads((root / "data.json").read_text()), task.debug["ops"][:-1])
        self.assertNotEqual(partial, json.loads(task.expect.files["data.json"]))
        _, record = self.deliver("ordered_migration", 5, "data.json", json.dumps(partial), "MIGRATED")
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["files"])

    def test_an_allocation_off_by_one_unit_fails(self):
        _, task = self.build(*calib("capped_allocation", 5))
        lines = task.expect.files["allocations.txt"].splitlines()
        team, amount = lines[0].split("=")
        wrong = "\n".join([f"{team}={int(amount) + 1}", *lines[1:]]) + "\n"
        _, record = self.deliver("capped_allocation", 5, "allocations.txt", wrong, "ALLOCATED")
        self.assertFalse(record["checks"]["files"])

    def test_a_rule_table_applied_last_match_wins_fails(self):
        root, task = self.build(*calib("rule_table_apply", 5))
        customers = json.loads((root / "customers.json").read_text())["customers"]
        rules = task.debug["rules"]
        last = {c["id"]: first_match(c, [*reversed(rules[:-1]), rules[-1]]) for c in customers}
        wrong = "".join(f"{key}={value}\n" for key, value in sorted(last.items()))
        self.assertNotEqual(wrong, task.expect.files["discounts.txt"])
        _, record = self.deliver("rule_table_apply", 5, "discounts.txt", wrong, "APPLIED")
        self.assertFalse(record["checks"]["files"])

    def test_apply_migration_worked_example(self):
        ops = [("rename", "limit", "cap"), ("multiply", "cap", 3), ("append", "tags", "b"), ("delete", "owner", None)]
        self.assertEqual(apply_migration({"limit": 10, "tags": ["a"], "owner": "x"}, ops), {"cap": 30, "tags": ["a", "b"]})

    def test_allocate_worked_examples(self):
        requests = {"a": 50, "b": 30, "c": 20}
        self.assertEqual(allocate(60, requests), {"a": 30, "b": 18, "c": 12})
        self.assertEqual(allocate(60, requests, cap=25), {"a": 25, "b": 18, "c": 12})
        self.assertEqual(allocate(60, requests, cap=25, floor=15), {"a": 25, "b": 18, "c": 15})
        # Two units left: a is at the cap, b has the largest unmet request (12, then 11).
        self.assertEqual(allocate(60, requests, cap=25, floor=15, leftover=True), {"a": 25, "b": 20, "c": 15})
        self.assertEqual(allocate(60, requests, priority="a"), {"a": 50, "b": 6, "c": 4})

    def test_first_match_takes_the_first_rule_that_holds(self):
        rules = [((("tier", "is", "gold"),), 15), ((("visits", "over", 10),), 5), ((), 0)]
        self.assertEqual([first_match({"tier": t, "visits": v}, rules) for t, v in
                          (("gold", 20), ("silver", 20), ("silver", 3))], [15, 5, 0])

    def test_plan_transfers_worked_examples(self):
        self.assertEqual(plan_transfers({"a": {"x": 10}, "b": {"x": 0}, "c": {"x": 5}},
                                        {"a": {"x": 3}, "b": {"x": 7}, "c": {"x": 5}}), [("x", "a", "b", 7)])
        # Ties go to the alphabetically first warehouse, on both sides.
        self.assertEqual(plan_transfers({w: {"x": s} for w, s in zip("abcd", (6, 6, 0, 0))},
                                        {w: {"x": 3} for w in "abcd"}),
                         [("x", "a", "c", 3), ("x", "b", "d", 3)])

    def test_adjust_pay_worked_example(self):
        employees = [{"id": "E-1", "dept": "eng", "rating": 5, "salary": 5000},
                     {"id": "E-2", "dept": "eng", "rating": 3, "salary": 4000}]
        rules = [((("rating", "at_least", 4),), 6), ((), 2)]
        self.assertEqual(adjust_pay(employees, rules, cap=None, budgets={}), {"E-1": 5300, "E-2": 4080})
        # A budget of 320 fits E-1's 300 (higher rating first) but not E-2's 80.
        self.assertEqual(adjust_pay(employees, rules, cap=None, budgets={"eng": 320}), {"E-1": 5300, "E-2": 4000})
        self.assertEqual(adjust_pay(employees, rules, cap=250, budgets={}), {"E-1": 5250, "E-2": 4080})


if __name__ == "__main__":
    unittest.main()
