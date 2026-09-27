"""k sampled runs combine into per-task pass counts and pass@1 / pass^k / pass@k."""

import unittest

from agent_from_scratch.evals.bench.aggregate import aggregate


def record(task_id, family, passed, false_completion=None):
    return {"id": task_id, "skeleton": task_id.rsplit("-", 1)[0], "family": family,
            "passed": passed, "false_completion": false_completion}


class AggregateTests(unittest.TestCase):
    def test_rates_over_three_samples(self):
        runs = [[record("a-0", "inspection", True), record("b-0", "updates", True, False)],
                [record("a-0", "inspection", True), record("b-0", "updates", False, True)],
                [record("a-0", "inspection", True), record("b-0", "updates", False, True)]]
        report = aggregate(runs, [0, 1, 2])
        self.assertEqual(report["overall"], {"tasks": 2, "mean_pass@1": round(4 / 6, 4),
                                             "pass^k": 0.5, "pass@k": 1.0, "mixed": 1})
        self.assertEqual(report["by_family"]["updates"]["mixed"], 1)
        self.assertEqual(report["false_completion"], {"count": 2, "claims": 3})
        self.assertEqual(report["tasks"]["b-0"]["passes"], 1)

    def test_runs_must_share_tasks_and_differ_in_seed(self):
        a, b = [record("a-0", "inspection", True)], [record("b-0", "inspection", True)]
        with self.assertRaises(ValueError):
            aggregate([a, b], [0, 1])
        with self.assertRaises(ValueError):
            aggregate([a, a], [0, 0])


if __name__ == "__main__":
    unittest.main()
