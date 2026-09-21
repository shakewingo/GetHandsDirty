"""Trajectory metrics describe run shape with tool categories, not coding stages."""

import unittest
from unittest.mock import Mock

from agent_from_scratch.agent import Agent
from agent_from_scratch.evals.trajectory import compare, profile
from agent_from_scratch.evals.verify import metrics
from agent_from_scratch.llm import LLM, ResponseError, ResponseErrorCode
from agent_from_scratch.tests.test_turn import answer, call


class TrajectoryTests(unittest.TestCase):
    def run_script(self, *responses):
        model = Mock(spec=LLM)
        model.settings.return_value = {}
        model.read_usage.side_effect = LLM.read_usage
        model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 3744,
            "window_tokens": 8000, "response_reserve": 512, "remaining_tokens": 3744}
        model.generate.side_effect = list(responses)
        return Agent(model).run_turn("2+2")

    def test_metrics_label_each_actor_request(self):
        record = metrics(self.run_script(ResponseError(ResponseErrorCode.INVALID_RESPONSE),
                                         call(), answer("4")))
        self.assertEqual(record["actions"], ["error", "execute", "answer"])
        self.assertEqual(record["peak_context_ratio"], 0.5)
        self.assertEqual((record["elided_messages"], record["plan_updates"],
                          record["stuck_reminders"]), (0, 0, 0))

    def test_profile_reports_survival_mix_and_overflow(self):
        records = [{"id": "a", "passed": True, "stop_reason": "final_response",
                    "actions": ["explore", "answer"], "peak_context_ratio": 0.4,
                    "elided_messages": 1, "compact_requests": 0, "plan_updates": 0,
                    "stuck_reminders": 0, "model_requests": 2, "usage": {"total_tokens": 10}},
                   {"id": "b", "passed": False, "stop_reason": "context_limit",
                    "actions": ["explore"], "peak_context_ratio": 0.9,
                    "elided_messages": 3, "compact_requests": 2, "plan_updates": 0,
                    "stuck_reminders": 1, "model_requests": 3, "usage": {"total_tokens": None}}]
        summary = profile(records)
        self.assertEqual(summary["context_limit_rate"], 0.5)
        self.assertEqual(summary["survival"][0], {"request": 1, "active": 1.0, "mix": {"explore": 2}})
        self.assertEqual(summary["survival"][1], {"request": 2, "active": 0.5, "mix": {"answer": 1}})
        self.assertEqual(summary["mean"]["elided_messages"], 2)
        self.assertIsNone(summary["mean"]["total_tokens"])

    def test_compare_pairs_tasks_and_uses_the_exact_mcnemar_test(self):
        a = [{"id": str(i), "passed": False} for i in range(6)]
        b = [{"id": str(i), "passed": i < 5} for i in range(6)]
        result = compare(a, b)
        self.assertEqual((result["both"], result["only_a"], result["only_b"], result["neither"]),
                         (0, 0, 5, 1))
        self.assertAlmostEqual(result["mcnemar_p"], 0.0625)


if __name__ == "__main__":
    unittest.main()
