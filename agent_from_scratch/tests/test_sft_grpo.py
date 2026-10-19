"""GRPO's reward is strict, advantages follow the closed form, and rows hold what was sampled."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.config import CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.run import save
from agent_from_scratch.evals.sft.grpo import (advantages, collect_groups, iteration_tasks, reward,
                                               sampled_rows, train_pool)
from agent_from_scratch.llm import compile_chat_template, render_chat

TEMPLATE_TEXT = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
TEMPLATE = compile_chat_template(TEMPLATE_TEXT)
TEMPLATE_SHA256 = sha256(TEMPLATE_TEXT.encode("utf-8")).hexdigest()
END = "<|im_end|>"
MESSAGES = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Go."}]


def trace_with(text: str, *, finish: str = "stop", status: str = "completed", shape: str = "text") -> dict:
    """A schema-5 trace with one agent request whose reply was `text`.

    `shape` "text" is vLLM's completion response; "message" is the chat shape a parse error keeps.
    """
    choice = ({"text": text, "finish_reason": finish} if shape == "text"
              else {"message": {"role": "assistant", "content": text}, "finish_reason": finish})
    return {"schema_version": 5, "messages": MESSAGES, "model_requests": [{
        "purpose": "agent", "status": status, "input_message_count": 2, "input_messages": MESSAGES,
        "tools": {}, "finish_reason": finish, "raw_response": {"choices": [choice]}}]}


def fake_run(root: Path, name: str, seed: int, outcomes: dict, *, split: str = "train",
             adapter: str = "M1") -> Path:
    """A bench-shaped run directory; `outcomes` maps task ID -> (passed, format_exact)."""
    out = root / name
    records = []
    for task_id, (passed, exact) in outcomes.items():
        run_id = f"{task_id}-s{seed}"
        runs = out / task_id / "state" / "runs"
        runs.mkdir(parents=True)
        (runs / f"{run_id}.jsonl").write_text(json.dumps(trace_with(f"answer {seed}")))
        records.append({"id": task_id, "skeleton": task_id.rsplit("-", 1)[0], "family": "updates",
                        "split": split, "run_id": run_id, "passed": passed,
                        "checks": {"format_exact": exact}})
    save(out / "results.json", records)
    save(out / "metadata.json", {"split": split, "seed": seed, "settings": {
        "backend": "vllm", "adapter": adapter, "chat_template_sha256": TEMPLATE_SHA256}})
    return out


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = train_pool()

    def test_pool_is_the_whole_train_split(self):
        self.assertEqual(len(self.pool), 21)
        self.assertEqual({len(ids) for ids in self.pool.values()}, {20})

    def test_one_pass_covers_the_pool_once(self):
        skeleton_of = {task_id: name for name, ids in self.pool.items() for task_id in ids}
        chunks = [iteration_tasks(self.pool, 1, i) for i in range(10)]
        every = [task_id for chunk in chunks for task_id in chunk]
        self.assertEqual(sorted(every), sorted(skeleton_of))
        for chunk in chunks:
            self.assertEqual(len(chunk), 42)
            self.assertEqual(set(Counter(skeleton_of[task_id] for task_id in chunk).values()), {2})

    def test_rotation_differs_between_passes(self):
        self.assertNotEqual(iteration_tasks(self.pool, 1, 0), iteration_tasks(self.pool, 2, 0))
        self.assertEqual(iteration_tasks(self.pool, 1, 0), iteration_tasks(self.pool, 1, 0))

    def test_an_iteration_past_the_pass_is_refused(self):
        with self.assertRaises(ValueError):
            iteration_tasks(self.pool, 1, 10)


class RewardTests(unittest.TestCase):
    def test_strict_reward(self):
        self.assertEqual(reward({"passed": True, "checks": {"format_exact": False}}), 0)
        self.assertEqual(reward({"passed": True, "checks": {"format_exact": True}}), 1)
        self.assertEqual(reward({"passed": True, "checks": {}}), 1)  # no expected answer
        self.assertEqual(reward({"passed": False, "checks": {"format_exact": True}}), 0)

    def test_advantages_match_the_closed_form(self):
        weights = advantages([1] + [0] * 7)
        self.assertAlmostEqual(weights[0], 7 ** 0.5, places=4)        # 2.6458
        self.assertAlmostEqual(weights[1], -(1 / 7) ** 0.5, places=4)  # -0.3780
        self.assertIsNone(advantages([1] * 8))
        self.assertIsNone(advantages([0] * 8))
        self.assertIsNone(advantages([1]))


class SampledRowsTests(unittest.TestCase):
    def test_the_prompt_is_the_rendered_input_and_the_target_is_the_sampled_text(self):
        [row] = sampled_rows(trace_with("Done."), TEMPLATE)
        self.assertEqual(row["prompt"], render_chat(TEMPLATE, MESSAGES, {}))
        self.assertEqual(row["completion"], "Done." + END)

    def test_parse_error_sample_is_read_from_the_chat_shape(self):
        trace = trace_with("<tool_call>{bad", status="parse_error", shape="message")
        self.assertEqual(sampled_rows(trace, TEMPLATE)[0]["completion"], "<tool_call>{bad" + END)

    def test_truncated_sample_gets_no_end_token(self):
        self.assertEqual(sampled_rows(trace_with("half a rep", finish="length"), TEMPLATE)[0]["completion"],
                         "half a rep")

    def test_an_answered_request_without_sampled_text_is_an_error(self):
        trace = trace_with("x")
        trace["model_requests"][0]["raw_response"] = None
        with self.assertRaises(ValueError):
            sampled_rows(trace, TEMPLATE)

    def test_requests_the_model_did_not_answer_are_skipped(self):
        trace = trace_with("x", status="model_error")
        self.assertEqual(sampled_rows(trace, TEMPLATE), [])


class CollectGroupsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_group_of_seven_still_counts(self):
        runs = [fake_run(self.root, f"s{seed}", seed, {"shape_a-0": (seed == 0, True)}) for seed in range(7)]
        rows, metadata = collect_groups(runs)
        self.assertEqual((metadata["rollouts"], metadata["mixed"], len(rows)), (7, 1, 7))
        self.assertAlmostEqual(sum(row["weight"] for row in rows), 0.0, places=9)
        first = next(row for row in rows if row["provenance"]["seed"] == 0)
        self.assertAlmostEqual(first["weight"], 6 ** 0.5)
        self.assertEqual((first["completion"], first["provenance"]["reward"]), ("answer 0" + END, 1))

    def test_all_pass_and_all_fail_groups_are_skipped(self):
        outcomes = {"shape_a-0": (True, True), "shape_b-0": (False, True)}
        runs = [fake_run(self.root, f"s{seed}", seed, outcomes) for seed in range(8)]
        rows, metadata = collect_groups(runs)
        self.assertEqual((metadata["all_pass"], metadata["all_fail"], metadata["mixed"], rows), (1, 1, 0, []))

    def test_a_loose_answer_counts_as_a_failure(self):
        runs = [fake_run(self.root, f"s{seed}", seed, {"shape_a-0": (True, seed != 0)}) for seed in range(8)]
        rows, metadata = collect_groups(runs)
        self.assertEqual((metadata["mixed"], metadata["mean_reward"]), (1, 0.875))
        self.assertLess(next(r for r in rows if r["provenance"]["seed"] == 0)["weight"], 0)

    def test_runs_from_two_adapters_are_refused(self):
        runs = [fake_run(self.root, "s0", 0, {"shape_a-0": (True, True)}, adapter="grpo"),
                fake_run(self.root, "s1", 1, {"shape_a-0": (False, True)}, adapter="M1")]
        with self.assertRaisesRegex(ValueError, "different adapters"):
            collect_groups(runs)

    def test_only_train_runs_are_accepted(self):
        runs = [fake_run(self.root, f"s{seed}", seed, {"shape_a-0": (seed == 0, True)}, split="val")
                for seed in range(2)]
        with self.assertRaisesRegex(ValueError, "train split"):
            collect_groups(runs)

    def test_the_same_task_at_the_same_seed_twice_is_an_error(self):
        runs = [fake_run(self.root, name, 0, {"shape_a-0": (True, True)}) for name in ("a", "b")]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            collect_groups(runs)


if __name__ == "__main__":
    unittest.main()
