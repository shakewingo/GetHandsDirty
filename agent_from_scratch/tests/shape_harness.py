"""Checks every Stage 10 shape shares: each level keeps the Item 1 gates, and difficulty grows
with the level. Mixed into a TestCase that sets SHAPE_NAMES (docs/STAGE10_HARD_TASKS_PLAN.md)."""

from pathlib import Path
import random
from tempfile import TemporaryDirectory

from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import LEVELS, answer
from agent_from_scratch.tests.test_bench_skeletons import _model


def calib(name: str, level: int, seed: int = 100, condition: str | None = None):
    """The calibration (skeleton, ctx) of `name` at `level`; recovery shapes need a condition."""
    return next((s, c) for s, c in specs("calib")
                if s.name == f"{name}@L{level}" and c.seed == seed and c.condition == condition)


class ShapeGates:
    SHAPE_NAMES: tuple[str, ...] = ()
    CONDITIONS: tuple = (None,)  # ("clean", "fault") for a recovery shape
    GATE_SEEDS = (100, 101)

    def build(self, skeleton, ctx):
        root = Path(self.enterContext(TemporaryDirectory()))
        return root, skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), root, ctx)

    def run_script(self, skeleton, ctx, responses) -> dict:
        return run_case(_model(responses), skeleton, ctx, Path(self.enterContext(TemporaryDirectory())) / "run")

    def test_every_level_keeps_the_gates(self):
        for name in self.SHAPE_NAMES:
            for level in LEVELS:
                for seed in self.GATE_SEEDS:
                    for condition in self.CONDITIONS:
                        skeleton, ctx = calib(name, level, seed, condition)
                        with self.subTest(id=ctx.id):
                            _, task = self.build(skeleton, ctx)
                            correct = self.run_script(skeleton, ctx, skeleton.solution(task))
                            self.assertTrue(correct["passed"], correct["checks"])
                            turns = 1 + len(task.followups)
                            only = self.run_script(skeleton, ctx, [answer(task.expect.answer.value)] * turns)
                            self.assertFalse(only["passed"], "answer-only passed")
                            wrong = self.run_script(skeleton, ctx, skeleton.solution(task)[:-1] + [answer("WRONG")])
                            self.assertFalse(wrong["passed"], "wrong answer passed")

    def test_difficulty_grows_with_level(self):
        for name in self.SHAPE_NAMES:
            with self.subTest(shape=name):
                sizes = [self.build(*calib(name, level, condition=self.CONDITIONS[0]))[1].debug["size"]
                         for level in LEVELS]
                self.assertEqual(sizes, sorted(set(sizes)), sizes)
