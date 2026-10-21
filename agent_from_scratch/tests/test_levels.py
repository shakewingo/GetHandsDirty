"""Stage 10 levels: calibration covers every level on its own seeds, and a shape enters train,
val or hardtest only at the level levels.json names; the Stage 8-9 splits do not move."""

import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.evals.bench import run, skeletons
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import CALIB_SEEDS, LEVELS, SHAPES, Shape, answer, call
from agent_from_scratch.evals.bench.spec import Answer, Expect, Task
from agent_from_scratch.tests.test_bench_skeletons import _model

SPLITS = json.loads((Path(skeletons.__file__).parent / "splits.json").read_text())


def _toy_build(rng, root, ctx, level):
    """A file of level + 2 lines; the answer is how many."""
    lines = [f"item-{rng.randint(100, 999)}" for _ in range(level + 2)]
    (root / "items.txt").write_text("\n".join(lines) + "\n")
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
                prompt="Read items.txt and reply only how many lines it has.", tools=("read_file",),
                expect=Expect(answer=Answer(str(len(lines)), "required"), evidence=(lines[0],)),
                pair_id=ctx.pair_id, condition=ctx.condition, debug={"level": level})


def _toy_solution(task):
    return [call("read_file", path="items.txt"), answer(task.expect.answer.value)]


TOY = Shape("toy_count", "inspection", "train", tuple(range(20)), False, _toy_build, _toy_solution)


class LevelTests(unittest.TestCase):
    def use(self, shapes, levels):
        self.enterContext(patch.object(skeletons, "SHAPES", shapes))
        self.enterContext(patch.object(skeletons, "load_levels", return_value=levels))

    def build(self, skeleton, ctx):
        root = Path(self.enterContext(TemporaryDirectory()))
        return skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), root, ctx)

    def test_calibration_covers_every_level_on_its_own_seeds(self):
        self.use([TOY], {})
        entries = specs("calib")
        self.assertEqual(len(entries), len(LEVELS) * len(CALIB_SEEDS))
        self.assertEqual({s.name for s, _ in entries}, {f"toy_count@L{level}" for level in LEVELS})
        self.assertEqual({c.seed for _, c in entries}, set(CALIB_SEEDS))
        self.assertTrue(all(c.split == "calib" for _, c in entries))
        skeleton, ctx = next((s, c) for s, c in entries if s.name == "toy_count@L4")
        self.assertEqual((ctx.id, self.build(skeleton, ctx).debug["level"]), ("toy_count@L4-100", 4))
        self.assertNotIn("toy_count@L4-100", {c.id for _, c in specs()})  # never outside calib

    def test_an_unleveled_shape_is_not_registered(self):
        self.use([TOY], {})
        self.assertEqual([c.id for _, c in specs() if c.name == "toy_count"], [])

    def test_a_leveled_shape_registers_at_that_level(self):
        self.use([TOY], {"toy_count": 3})
        entries = [(s, c) for s, c in specs("train") if c.name == "toy_count"]
        self.assertEqual([c.seed for _, c in entries], list(range(20)))
        skeleton, ctx = entries[0]
        task = self.build(skeleton, ctx)
        self.assertEqual((task.debug["level"], task.split, skeleton.strict), (3, "train", True))
        out = Path(self.enterContext(TemporaryDirectory())) / "run"
        record = run_case(_model(skeleton.solution(task)), skeleton, ctx, out)
        self.assertTrue(record["passed"], record["checks"])

    def test_an_unknown_shape_or_level_is_refused(self):
        for levels in ({"nope": 2}, {"toy_count": 6}, {"toy_count": 0}):
            with self.subTest(levels=levels):
                self.use([TOY], levels)
                with self.assertRaises(ValueError):
                    specs("train")

    def test_the_stage_8_and_9_splits_are_unchanged(self):
        self.enterContext(patch.object(skeletons, "load_levels", return_value={}))
        counts = {split: len(specs(split)) for split in ("train", "val", "test", "hardtest")}
        self.assertEqual(counts, {"train": 420, "val": 104, "test": 140, "hardtest": 0})

    def test_shapes_match_splits_json(self):
        names = [shape.name for shape in SHAPES]
        self.assertEqual(len(names), len(set(names)), "duplicate shape name")
        self.assertFalse(set(names) & {skeleton.name for skeleton in skeletons.SKELETONS})
        self.assertEqual({split: sorted(listed) for split, listed in SPLITS["shapes"].items()},
                         {split: sorted(s.name for s in SHAPES if s.split == split)
                          for split in ("train", "val", "hardtest")})

    def test_hardtest_needs_final(self):
        out = Path(self.enterContext(TemporaryDirectory())) / "out"
        argv = ["bench", "--split", "hardtest", "--backend", "vllm", "--output", str(out)]
        with patch.object(sys, "argv", argv), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            run.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
