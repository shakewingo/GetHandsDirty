"""Stage 9 task contracts and negative gates; all tools and scoring are real."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import answer, call
from agent_from_scratch.evals.bench.spec import BuildContext
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.tests.test_bench_skeletons import _model


class HeldOutTaskTests(unittest.TestCase):
    split = "holdout"
    expected_names = {"three_source_join", "merge_record_requests", "backup_record_lookup", "unit_complete_total"}

    def setUp(self):
        self.entries = specs(self.split)
        self.assertEqual(len(self.entries), 80)
        self.assertEqual({s.name for s, _ in self.entries}, self.expected_names)

    def build(self, skeleton, ctx):
        root = Path(self.enterContext(TemporaryDirectory()))
        task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), root, ctx)
        return root, task

    def run_script(self, skeleton, ctx, responses):
        output = Path(self.enterContext(TemporaryDirectory())) / "run"
        return run_case(_model(responses), skeleton, ctx, output)

    def test_correct_scripts_pass_and_faults_are_observed(self):
        for skeleton, ctx in self.entries:
            with self.subTest(id=ctx.id):
                _, task = self.build(skeleton, ctx)
                record = self.run_script(skeleton, ctx, skeleton.solution(task))
                self.assertTrue(record["passed"], record["checks"])
                if ctx.condition == "fault":
                    self.assertTrue(record["fault_encountered"])

    def test_answer_only_is_rejected(self):
        for skeleton, ctx in self.entries:
            with self.subTest(id=ctx.id):
                _, task = self.build(skeleton, ctx)
                record = self.run_script(skeleton, ctx, [answer(task.expect.answer.value)])
                self.assertFalse(record["passed"])

    def test_wrong_answer_after_correct_actions_is_rejected(self):
        for skeleton, ctx in self.entries:
            with self.subTest(id=ctx.id):
                _, task = self.build(skeleton, ctx)
                record = self.run_script(skeleton, ctx, skeleton.solution(task)[:-1] + [answer("WRONG")])
                self.assertFalse(record["passed"])
                self.assertFalse(record["checks"]["answer_content"])

    def test_reproducible_inputs_and_contracts(self):
        for skeleton, ctx in self.entries:
            with self.subTest(id=ctx.id):
                a, task_a = self.build(skeleton, ctx)
                b, task_b = self.build(skeleton, ctx)
                self.assertEqual(snapshot(a), snapshot(b))
                self.assertEqual(task_a, task_b)

    def test_clean_fault_pairs_share_the_base_records(self):
        for skeleton, clean in self.entries:
            if clean.condition != "clean":
                continue
            fault = next(ctx for s, ctx in self.entries if s.name == skeleton.name
                         and ctx.seed == clean.seed and ctx.condition == "fault")
            a, _ = self.build(skeleton, clean)
            b, _ = self.build(skeleton, fault)
            # Backup lookup deliberately removes the primary; backup data stays matched.
            a_files, b_files = snapshot(a), snapshot(b)
            if skeleton.name == "backup_record_lookup":
                a_files.pop("current.json")
            self.assertEqual(a_files, b_files)

    def test_metadata_and_clipped_content_cannot_supply_evidence(self):
        for skeleton, ctx in self.entries:
            if ctx.condition == "fault":
                continue
            root, task = self.build(skeleton, ctx)
            responses = []
            for path in root.iterdir():
                lines = path.read_text().splitlines()
                responses.append(call("read_file", path=path.name, offset=len(lines),
                                      column=max(len(lines[-1]) - 1, 0), limit=1))
            record = self.run_script(skeleton, ctx, responses + [answer(task.expect.answer.value)])
            self.assertFalse(record["checks"]["evidence"])
            self.assertFalse(record["passed"])


class TrainingTaskTests(HeldOutTaskTests):
    split = "train"
    expected_names = {"inventory_reconciliation", "eligibility_record_update",
                      "address_conflict_stop", "points_save_recovery"}

    def test_write_tasks_require_the_actual_change_and_a_readback(self):
        for skeleton, ctx in self.entries:
            root, task = self.build(skeleton, ctx)
            if not task.expect.files:
                continue
            with self.subTest(id=ctx.id):
                reads = [call("read_file", path=p.name) for p in root.iterdir()]
                missing_write = self.run_script(skeleton, ctx, reads + [answer(task.expect.answer.value)])
                self.assertFalse(missing_write["passed"])
                self.assertFalse(missing_write["checks"]["files"])
                script = skeleton.solution(task)
                # Drop reads of modified files that occur after the first mutation.
                saw_write = False
                no_readback = []
                for response in script:
                    names = [c.name for c in response.tool_calls]
                    if "write_file" in names or "edit_file" in names:
                        saw_write = True
                    if saw_write and any(c.name == "read_file" and c.arguments["path"] in task.expect.files
                                         for c in response.tool_calls):
                        continue
                    no_readback.append(response)
                record = self.run_script(skeleton, ctx, no_readback)
                self.assertFalse(record["passed"])

    def test_blocked_cases_reject_even_an_unauthorized_write_attempt(self):
        for skeleton, ctx in self.entries:
            root, task = self.build(skeleton, ctx)
            if task.expect.answer.value not in ("BLOCKED", "NEED_INPUT"):
                continue
            with self.subTest(id=ctx.id):
                script = skeleton.solution(task)[:-1] + [call("write_file", path="extra.txt", content="x"),
                                                         answer(task.expect.answer.value)]
                record = self.run_script(skeleton, ctx, script)
                self.assertFalse(record["checks"]["process"])
                self.assertFalse(record["passed"])


class TaskOracleTests(unittest.TestCase):
    def build(self, name, seed=0):
        skeleton = next(s for s, _ in specs() if s.name == name)
        root = Path(self.enterContext(TemporaryDirectory()))
        task = skeleton.build(random.Random(42), root,
                              BuildContext(name, skeleton.family, skeleton.split, seed, None))
        return root, task

    def test_join_and_unit_total_have_independent_known_answers(self):
        _, join = self.build("three_source_join")
        self.assertEqual(join.expect.answer.value, "person-754@example.test|city-125")
        _, total = self.build("unit_complete_total")
        self.assertEqual(total.expect.answer.value, "756")  # 754 cents + 2 cents.
        _, missing = self.build("unit_complete_total", 1)
        self.assertEqual(missing.expect.answer.value, "NEED_UNIT")

    def test_budget_boundary_and_ineligibility(self):
        _, over = self.build("eligibility_record_update", 0)
        self.assertEqual(over.expect.answer.value, "BLOCKED")  # 91 requested, 90 available.
        _, boundary = self.build("eligibility_record_update", 1)
        self.assertEqual(boundary.expect.files["application.txt"],
                         "owner=person-214\nstatus=APPROVED\nnote=keep\n")
        _, ineligible = self.build("eligibility_record_update", 7)
        self.assertEqual(ineligible.expect.answer.value, "BLOCKED")

    def test_points_updates_preserve_other_lines_and_apply_once(self):
        _, task = self.build("points_save_recovery")
        self.assertEqual(task.expect.files, {
            "account_a.txt": "person=person-859\nbalance=120\nnote=keep\n",  # 103 + 17.
            "account_b.txt": "person=person-350\nbalance=443\nnote=keep\n",  # 435 + 8.
        })

    def test_address_agreement_and_conflict(self):
        _, agreement = self.build("address_conflict_stop", 0)
        _, conflict = self.build("address_conflict_stop", 1)
        self.assertEqual(agreement.expect.answer.value, "street-754")
        self.assertEqual(conflict.expect.answer.value, "NEED_INPUT")

    def test_compatible_requests_and_conflict_do_not_change_unrelated_lines(self):
        _, compatible = self.build("merge_record_requests", 0)
        _, conflict = self.build("merge_record_requests", 1)
        self.assertEqual(compatible.expect.files["contact.txt"],
                         "name=customer-125\ncity=CITY-754\nphone=PHONE-214\nnote=keep\n")
        self.assertEqual(conflict.expect.answer.value, "NEED_INPUT")
        self.assertEqual(conflict.expect.files, {})

    def test_correct_fallback_without_observing_primary_failure_is_rejected(self):
        skeleton, ctx = next((s, ctx) for s, ctx in specs("holdout")
                             if s.name == "backup_record_lookup" and ctx.condition == "fault")
        root = Path(self.enterContext(TemporaryDirectory()))
        task = skeleton.build(random.Random(42), root, ctx)
        output = Path(self.enterContext(TemporaryDirectory())) / "run"
        record = run_case(_model([call("read_file", path="archive.json"), answer(task.expect.answer.value)]),
                          skeleton, ctx, output)
        self.assertTrue(record["checks"]["evidence"])
        self.assertTrue(record["checks"]["answer_content"])
        self.assertFalse(record["checks"]["fault_observed"])
        self.assertFalse(record["passed"])

    def test_other_files_remain_protected_after_successful_updates(self):
        for name in ("eligibility_record_update", "points_save_recovery", "merge_record_requests"):
            skeleton, ctx = next((s, ctx) for s, ctx in specs() if s.name == name and ctx.seed == 0)
            if name == "eligibility_record_update":
                skeleton, ctx = next((s, ctx) for s, ctx in specs("train") if s.name == name and ctx.seed == 1)
            root = Path(self.enterContext(TemporaryDirectory()))
            task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), root, ctx)
            script = skeleton.solution(task)[:-1] + [call("write_file", path="unrelated.txt", content="x"),
                                                     answer(task.expect.answer.value)]
            output = Path(self.enterContext(TemporaryDirectory())) / "run"
            record = run_case(_model(script), skeleton, ctx, output)
            self.assertFalse(record["checks"]["unchanged"])
            self.assertFalse(record["passed"])

    def test_fallback_requires_failure_of_the_designated_primary_path(self):
        skeleton, ctx = next((s, ctx) for s, ctx in specs("holdout")
                             if s.name == "backup_record_lookup" and ctx.condition == "fault")
        root = Path(self.enterContext(TemporaryDirectory()))
        task = skeleton.build(random.Random(42), root, ctx)
        for path, accepted in (("typo.json", False), ("./current.json", True),
                               ("unused/../current.json", True)):
            with self.subTest(path=path):
                output = Path(self.enterContext(TemporaryDirectory())) / "run"
                script = [call("read_file", path=path), call("read_file", path="archive.json"),
                          answer(task.expect.answer.value)]
                record = run_case(_model(script), skeleton, ctx, output)
                self.assertEqual(record["checks"]["fault_observed"], accepted)
                self.assertEqual(record["fault_encountered"], accepted)
                self.assertEqual(record["passed"], accepted)

    def test_reading_decision_sources_after_an_oracle_write_is_rejected(self):
        skeleton, ctx = next((s, ctx) for s, ctx in specs("train")
                             if s.name == "eligibility_record_update" and ctx.seed == 1)
        root = Path(self.enterContext(TemporaryDirectory()))
        task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), root, ctx)
        script = [call("read_file", path="application.txt"),
                  call("write_file", path="application.txt", content=task.expect.files["application.txt"]),
                  call("read_file", path="policy.json"), call("read_file", path="request.json"),
                  call("read_file", path="application.txt"), answer("UPDATED")]
        output = Path(self.enterContext(TemporaryDirectory())) / "run"
        record = run_case(_model(script), skeleton, ctx, output)
        self.assertTrue(record["checks"]["files"])
        self.assertTrue(record["checks"]["evidence"])
        self.assertFalse(record["passed"])
        self.assertTrue(record["false_completion"])

    def test_alternative_read_order_and_targeted_edit_are_valid(self):
        skeleton, ctx = next((s, ctx) for s, ctx in specs("train")
                             if s.name == "eligibility_record_update" and ctx.seed == 1)
        script = [call("read_file", path="request.json"), call("read_file", path="application.txt"),
                  call("read_file", path="policy.json"),
                  call("edit_file", path="application.txt", old_text="status=PENDING", new_text="status=APPROVED"),
                  call("read_file", path="application.txt"), answer("UPDATED")]
        output = Path(self.enterContext(TemporaryDirectory())) / "run"
        record = run_case(_model(script), skeleton, ctx, output)
        self.assertTrue(record["passed"], record["checks"])


if __name__ == "__main__":
    unittest.main()
