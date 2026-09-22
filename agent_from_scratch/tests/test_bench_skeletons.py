"""Every registered skeleton's scripted solution passes; a fake claim fails; seeds reproduce."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import SKELETONS, answer
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.llm import LLM

SPLITS_PATH = Path(__file__).resolve().parents[1] / "evals" / "bench" / "splits.json"


def _model(responses):
    model = Mock(spec=LLM)
    model.settings.return_value = {}
    model.read_usage.side_effect = LLM.read_usage
    model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 1000,
        "window_tokens": 32768, "response_reserve": 2048, "remaining_tokens": 29720}
    model.generate.side_effect = list(responses)
    return model


class SkeletonGateTests(unittest.TestCase):
    def _task(self, skeleton, ctx):
        directory = Path(self.enterContext(TemporaryDirectory()))
        rng = random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
        return skeleton.build(rng, directory, ctx)

    def test_every_scripted_solution_passes(self):
        for index, (skeleton, ctx) in enumerate(specs()):
            with self.subTest(id=ctx.id):
                task = self._task(skeleton, ctx)
                output = Path(self.enterContext(TemporaryDirectory())) / f"run-{index}"
                record = run_case(_model(skeleton.solution(task)), skeleton, ctx, output)
                self.assertTrue(record["passed"], record["checks"])

    def test_a_fake_claim_with_no_tool_calls_fails(self):
        for index, (skeleton, ctx) in enumerate(specs()):
            with self.subTest(id=ctx.id):
                task = self._task(skeleton, ctx)
                claim = (task.claim_tokens[0] if task.claim_tokens else
                        task.expect.answer.value if task.expect.answer else "done")
                output = Path(self.enterContext(TemporaryDirectory())) / f"fake-{index}"
                record = run_case(_model([answer(claim)]), skeleton, ctx, output)
                self.assertFalse(record["passed"], record["checks"])

    def test_the_same_seed_rebuilds_an_identical_workspace(self):
        for skeleton, ctx in specs():
            with self.subTest(id=ctx.id):
                a = Path(self.enterContext(TemporaryDirectory()))
                b = Path(self.enterContext(TemporaryDirectory()))
                skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), a, ctx)
                skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), b, ctx)
                self.assertEqual(snapshot(a), snapshot(b))

    def test_registered_skeletons_match_splits_json_exactly(self):
        names = [skeleton.name for skeleton in SKELETONS]
        self.assertEqual(len(names), len(set(names)), "duplicate skeleton name")
        splits = json.loads(SPLITS_PATH.read_text())
        listed = {name for group in (splits["dev"], splits["test"], splits["train"]) for name in group}
        self.assertEqual(set(names), listed)
        for skeleton in SKELETONS:
            self.assertIn(skeleton.name, splits[skeleton.split])
        self.assertEqual(len([s for s in SKELETONS if s.split == "dev"]), 4)
        self.assertEqual(len([s for s in SKELETONS if s.split == "test"]), 10)
        dev_total = sum(len(s.seeds) * (2 if s.recovery else 1) for s in SKELETONS if s.split == "dev")
        test_total = sum(len(s.seeds) * (2 if s.recovery else 1) for s in SKELETONS if s.split == "test")
        self.assertEqual((dev_total, test_total), (15, 60))


if __name__ == "__main__":
    unittest.main()
