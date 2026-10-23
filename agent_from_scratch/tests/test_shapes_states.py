"""Stage 10 mixed-state and fault shapes: every level keeps the gates; touching a record that
must stay (locked, already right, pinned) fails; a fault left unretried fails."""

from types import SimpleNamespace
import unittest

from agent_from_scratch.evals.bench.shapes_states import _solution_planned
from agent_from_scratch.tests.shape_harness import ShapeGates, calib


def responses(steps: list) -> list:
    return _solution_planned(SimpleNamespace(debug={"script": steps}))


def without_retry(steps: list) -> list:
    """The planned steps with the retry (a step repeating the one before it) left out."""
    index = next(i for i in range(1, len(steps)) if steps[i] == steps[i - 1])
    return steps[:index] + steps[index + 1:]


class StateShapeTests(ShapeGates, unittest.TestCase):
    SHAPE_NAMES = ("mixed_state_batch", "flag_rollout", "cert_renewal_batch")

    def touched(self, name: str, keep_state: str) -> dict:
        """Run the solution, plus one write that changes a record the task says to leave."""
        skeleton, ctx = calib(name, 3)
        root, task = self.build(skeleton, ctx)
        held = task.debug["held"]
        path = next(p for p in sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.is_file())
                    if p not in task.expect.files and "/" in p
                    and ((keep_state == "held") == any(h in p for h in held)))
        content = (root / path).read_text() + "touched=yes\n"
        steps = task.debug["script"]
        steps = steps[:-1] + [("write", path, content), steps[-1]]
        return self.run_script(skeleton, ctx, responses(steps))

    def test_changing_a_held_record_fails(self):
        for name in self.SHAPE_NAMES:
            record = self.touched(name, "held")
            with self.subTest(shape=name):
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["unchanged"])

    def test_changing_a_record_that_is_already_right_fails(self):
        for name in self.SHAPE_NAMES:
            record = self.touched(name, "ok")
            with self.subTest(shape=name):
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["unchanged"])

    def test_a_save_that_fails_and_is_not_retried_fails(self):
        skeleton, ctx = calib("flag_rollout", 3)
        _, task = self.build(skeleton, ctx)
        self.assertEqual(task.fault.tool, "write_file")
        record = self.run_script(skeleton, ctx, responses(without_retry(task.debug["script"])))
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["files"])


class ReconcileShapeTests(ShapeGates, unittest.TestCase):
    SHAPE_NAMES = ("fault_amid_reconcile", "ledger_audit_fault")  # the second is the hard test's (Stage 11)
    CONDITIONS = ("clean", "fault")

    def test_the_fault_moves_later_with_the_level(self):
        for name in self.SHAPE_NAMES:
            calls = [self.build(*calib(name, level, condition="fault"))[1].fault.on_call for level in (1, 2, 3, 4, 5)]
            with self.subTest(shape=name):
                self.assertEqual(calls, sorted(calls))
                self.assertGreater(calls[-1], calls[0])

    def test_a_read_that_fails_and_is_not_retried_fails(self):
        for name in self.SHAPE_NAMES:
            skeleton, ctx = calib(name, 3, condition="fault")
            _, task = self.build(skeleton, ctx)
            record = self.run_script(skeleton, ctx, responses(without_retry(task.debug["script"])))
            with self.subTest(shape=name):
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["evidence"])
                self.assertTrue(record["checks"]["fault_observed"])

    def test_the_audit_writes_amounts_with_cents(self):
        _, task = self.build(*calib("ledger_audit_fault", 2, condition="clean"))
        line = task.expect.files["audit.txt"].splitlines()[0]
        self.assertRegex(line, r"^R-\d{3}: ledger \d+\.\d{2}, bank \d+\.\d{2}$")


if __name__ == "__main__":
    unittest.main()
