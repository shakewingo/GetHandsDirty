"""Train-task gates exercise the real agent, file tools and independent verifier."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.bench.manifest import build_manifest, MANIFEST_PATH
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import answer, call
from agent_from_scratch.evals.bench.spec import BuildContext
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.tests.test_bench_skeletons import _model


class TrainSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.entries = [(s, ctx) for s, ctx in specs("train") if s.name == "inventory_reconciliation"]
        self.assertEqual(len(self.entries), 20, "The v2 pool must generate ten clean/fault pairs.")
        self.skeleton = self.entries[0][0]

    def build(self, ctx, rng=None):
        root = Path(self.enterContext(TemporaryDirectory()))
        rng = rng or random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}")
        return root, self.skeleton.build(rng, root, ctx)

    def run_script(self, ctx, responses):
        output = Path(self.enterContext(TemporaryDirectory())) / "run"
        return run_case(_model(responses), self.skeleton, ctx, output)

    def test_scripted_solutions_pass_with_observed_faults(self):
        for _, ctx in self.entries:
            with self.subTest(id=ctx.id):
                _, task = self.build(ctx)
                record = self.run_script(ctx, self.skeleton.solution(task))
                self.assertTrue(record["passed"], record["checks"])
                self.assertEqual(record["changed_paths"], [])
                self.assertEqual(record["fault_encountered"], ctx.condition == "fault")

    def test_correct_answer_without_reading_fails_evidence(self):
        for _, ctx in self.entries:
            with self.subTest(id=ctx.id):
                _, task = self.build(ctx)
                record = self.run_script(ctx, [answer(task.expect.answer.value)])
                self.assertTrue(record["checks"]["answer_content"])
                self.assertFalse(record["checks"]["evidence"])
                self.assertFalse(record["passed"])

    def test_reading_both_files_with_wrong_answer_fails(self):
        for _, ctx in self.entries:
            with self.subTest(id=ctx.id):
                _, task = self.build(ctx)
                responses = self.skeleton.solution(task)[:-1] + [answer("WRONG")]
                record = self.run_script(ctx, responses)
                self.assertTrue(record["checks"]["evidence"])
                self.assertFalse(record["checks"]["answer_content"])
                self.assertFalse(record["passed"])

    def test_reading_only_one_source_fails_even_with_correct_answer(self):
        ctx = next(ctx for _, ctx in self.entries if ctx.condition == "clean")
        _, task = self.build(ctx)
        for path in ("inventory.json", "shipment.json"):
            with self.subTest(path=path):
                record = self.run_script(ctx, [call("read_file", path=path),
                                               answer(task.expect.answer.value)])
                self.assertFalse(record["checks"]["evidence"])
                self.assertFalse(record["passed"])

    def test_path_metadata_and_clipped_reads_are_not_source_evidence(self):
        for _, ctx in self.entries:
            if ctx.condition != "clean":
                continue
            root, task = self.build(ctx)
            for clipping in ("first_line", "last_character"):
                with self.subTest(id=ctx.id, clipping=clipping):
                    responses = []
                    for path in ("inventory.json", "shipment.json"):
                        lines = (root / path).read_text().splitlines()
                        arguments = {"path": path, "limit": 1}
                        if clipping == "last_character":
                            arguments.update(offset=len(lines), column=len(lines[-1]) - 1)
                        responses.append(call("read_file", **arguments))
                    record = self.run_script(ctx, responses + [answer(task.expect.answer.value)])
                    if clipping == "first_line" and len(lines) == 1:
                        # A full one-line JSON read contains every input fact.
                        self.assertTrue(record["passed"], record["checks"])
                    else:
                        self.assertFalse(record["checks"]["evidence"])
                        self.assertFalse(record["passed"])

    def test_skipping_the_failed_read_does_not_count_as_recovery(self):
        ctx = next(ctx for _, ctx in self.entries if ctx.condition == "fault")
        _, task = self.build(ctx)
        record = self.run_script(ctx, [call("read_file", path="inventory.json"),
                                      call("read_file", path="shipment.json"),
                                      answer(task.expect.answer.value)])
        self.assertTrue(record["fault_encountered"])
        self.assertFalse(record["checks"]["evidence"])
        self.assertFalse(record["passed"])

    def test_empty_single_and_multiple_missing_ids_have_exact_sorted_answers(self):
        # Random(42) samples 754, 214, 125, 859, 381, 350 from range(100, 999).
        for seed, expected in ((0, "NONE"), (1, "SKU-859"), (2, "SKU-381,SKU-859")):
            with self.subTest(seed=seed):
                ctx = BuildContext(self.skeleton.name, self.skeleton.family, "train", seed, None)
                root, task = self.build(ctx, random.Random(42))
                self.assertEqual(task.expect.answer.value, expected)
                self.assertEqual(task.expect.answer.format, "required")
                inventory = json.loads((root / "inventory.json").read_text())
                shipment = json.loads((root / "shipment.json").read_text())
                self.assertEqual(set(inventory["ids"]), {"SKU-754", "SKU-214", "SKU-125"})
                self.assertEqual(set(shipment["ids"]), {"SKU-125", *expected.split(",")} - {"NONE"})

    def test_seed_and_condition_reproduce_files_and_expectations(self):
        for _, ctx in self.entries:
            with self.subTest(id=ctx.id):
                a, task_a = self.build(ctx)
                b, task_b = self.build(ctx)
                self.assertEqual(snapshot(a), snapshot(b))
                self.assertEqual(task_a, task_b)

    def test_dev_test_tasks_and_protocol_match_the_frozen_manifest(self):
        frozen = json.loads(MANIFEST_PATH.read_text())
        current = build_manifest()
        eval_tasks = [task for task in current["tasks"] if task["split"] in ("dev", "test")]
        self.assertEqual(eval_tasks, frozen["tasks"])
        for key in ("model", "decoding", "seeds", "n_ctx", "limits", "memory",
                    "system_prompt_sha256", "chat_template_sha256", "verifier_sha256"):
            self.assertEqual(current[key], frozen[key], key)


if __name__ == "__main__":
    unittest.main()
