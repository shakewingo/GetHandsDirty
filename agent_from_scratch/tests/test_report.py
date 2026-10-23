"""Stage 11's report module: pass@1, pairing by task and seed, task-level bootstrap, missing seeds."""

import unittest

from agent_from_scratch.evals.report import bootstrap_diff, pass_at_1, paired, table
from agent_from_scratch.evals.trajectory import compare


def run(outcomes: dict[str, bool], skeleton: str = "s") -> list[dict]:
    return [{"id": task, "skeleton": skeleton, "passed": passed} for task, passed in outcomes.items()]


class ResultsTests(unittest.TestCase):
    def test_pass_at_1_counts_every_task_and_seed(self):
        self.assertAlmostEqual(pass_at_1([run({"a": True, "b": False}), run({"a": True, "b": True})]), 0.75)

    def test_pairs_are_matched_by_task_and_seed(self):
        a = [run({"a": True, "b": False}), run({"a": False, "b": False})]
        b = [run({"a": True, "b": True}), run({"a": True, "b": False})]
        self.assertEqual({k: paired(a, b)[k] for k in ("pairs", "wins", "losses", "both", "neither")},
                         {"pairs": 4, "wins": 2, "losses": 0, "both": 1, "neither": 1})

    def test_mcnemar_matches_compare(self):
        a = [run({f"t{i}": i % 3 == 0 for i in range(12)})]
        b = [run({f"t{i}": i % 2 == 0 for i in range(12)})]
        self.assertEqual(paired(a, b)["mcnemar_p"], compare(a[0], b[0])["mcnemar_p"])

    def test_bootstrap_resamples_tasks_not_runs(self):
        # Task "a" flips from fail to pass on all three seeds; "b" never changes. Resampling tasks
        # gives differences in {0, 0.5, 1}; resampling runs would give thirds as well.
        a = [run({"a": False, "b": True})] * 3
        b = [run({"a": True, "b": True})] * 3
        diff, low, high = bootstrap_diff(a, b, resamples=2000)
        self.assertAlmostEqual(diff, 0.5)
        self.assertIn(low, (0.0, 0.5, 1.0))
        self.assertIn(high, (0.0, 0.5, 1.0))
        self.assertLessEqual(low, diff)
        self.assertGreaterEqual(high, diff)

    def test_the_bootstrap_is_reproducible(self):
        a = [run({f"t{i}": i % 2 == 0 for i in range(20)})]
        b = [run({f"t{i}": i % 3 != 0 for i in range(20)})]
        self.assertEqual(bootstrap_diff(a, b), bootstrap_diff(a, b))

    def test_a_missing_seed_is_reported(self):
        arms = {"full": [run({"a": True})] * 3, "short": [run({"a": True})] * 2}
        text = table(arms, {"all": None})
        self.assertIn("2 of 3 seeds", text)
        self.assertNotIn("3 of 3", text)
        with self.assertRaisesRegex(ValueError, "Seed counts differ"):
            paired(arms["full"], arms["short"])

    def test_subsets_filter_by_skeleton(self):
        runs = [run({"a": True}, "hard") + run({"b": False}, "easy")]
        text = table({"x": runs}, {"all": None, "hard": {"hard"}})
        self.assertIn("| x | 0.500 (2) | 1.000 (1) |", text)


if __name__ == "__main__":
    unittest.main()
