"""Stage 10 dialogue shapes: every level keeps the gates; an act/ask pair shares its records and
differs only in the outcome; acting on the ask side, asking on the act side, and editing the
wrong record of the pair all fail."""

import unittest

from agent_from_scratch.evals.bench.shapes_dialogue import DIALOGUE_SHAPES
from agent_from_scratch.evals.bench.skeletons import LEVELS, answer, call
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.tests.shape_harness import ShapeGates, calib


class DialogueShapeTests(ShapeGates, unittest.TestCase):
    SHAPE_NAMES = tuple(shape.name for shape in DIALOGUE_SHAPES)

    def pair(self, name, level):
        """(skeleton, ctx, task) for the act side (seed 100) and the ask side (seed 101)."""
        sides = []
        for seed in (100, 101):
            skeleton, ctx = calib(name, level, seed)
            root, task = self.build(skeleton, ctx)
            sides.append((skeleton, ctx, task, snapshot(root)))
        return sides

    @staticmethod
    def script(task, last_turn) -> list:
        """The scripted solution's turns before the last, then `last_turn`."""
        responses = []
        for turn in task.debug["script"][:-1]:
            for step in turn:
                responses.append(call("read_file", path=step[1]) if step[0] == "read"
                                 else call("write_file", path=step[1], content=step[2]) if step[0] == "write"
                                 else answer(step[1]))
        return responses + last_turn

    def test_each_pair_shares_its_records_and_differs_in_outcome(self):
        for name in self.SHAPE_NAMES:
            for level in LEVELS:
                (_, act_ctx, act, act_files), (_, _, ask, ask_files) = self.pair(name, level)
                with self.subTest(id=act_ctx.id):
                    self.assertEqual(act_files, ask_files)
                    self.assertEqual((act.debug["outcome"], ask.debug["outcome"]), ("act", "ask"))
                    self.assertEqual(act.followups, ask.followups)
                    self.assertNotEqual(act.prompt, ask.prompt)
                    self.assertEqual(len(act.followups), level)
                    self.assertIn(act.debug["chosen"], act.debug["pair"])

    def test_acting_on_the_ask_side_fails(self):
        for name in self.SHAPE_NAMES:
            (_, _, act, _), (skeleton, ctx, ask, _) = self.pair(name, 3)
            target, content = next(iter(act.expect.files.items()))
            acted = self.script(ask, [call("read_file", path=target), call("write_file", path=target, content=content),
                                      call("read_file", path=target), answer(act.expect.answer.value)])
            record = self.run_script(skeleton, ctx, acted)
            with self.subTest(shape=name):
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["files"])

    def test_asking_on_the_act_side_fails(self):
        for name in self.SHAPE_NAMES:
            (skeleton, ctx, act, _), (_, _, ask, _) = self.pair(name, 3)
            target = next(iter(act.expect.files))
            asked = self.script(act, [call("read_file", path=target), answer(ask.expect.answer.value)])
            record = self.run_script(skeleton, ctx, asked)
            with self.subTest(shape=name):
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["files"])

    def test_editing_the_other_record_of_the_pair_fails(self):
        for name in self.SHAPE_NAMES:
            (skeleton, ctx, act, _), (_, _, ask, _) = self.pair(name, 3)
            target, acted = next(iter(act.expect.files.items()))
            before = ask.expect.files[target]  # the in-between edits only
            chosen = act.debug["chosen"]
            other = next(record for record in act.debug["pair"] if record != chosen)
            new_value = next(line for line in acted.splitlines() if line.startswith(chosen))[len(chosen):]
            wrong = "".join((other + new_value if line.startswith(other) else line) + "\n"
                            for line in before.splitlines())
            script = self.script(act, [call("read_file", path=target), call("write_file", path=target, content=wrong),
                                       call("read_file", path=target), answer(act.expect.answer.value)])
            record = self.run_script(skeleton, ctx, script)
            with self.subTest(shape=name):
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["files"])
                self.assertTrue(record["false_completion"])


if __name__ == "__main__":
    unittest.main()
