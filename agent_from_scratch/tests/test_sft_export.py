"""Verified train rollouts become (prompt, completion) pairs that are exactly what the model saw."""

import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.config import CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import answer, call
from agent_from_scratch.evals.run import save
from agent_from_scratch.evals.sft import export
from agent_from_scratch.evals.sft.export import (collect, examples_from_trace, flawed_turns, rejection,
                                                 select)
from agent_from_scratch.llm import (LLM, LLMResponse, ResponseError, ResponseErrorCode, ResponseType,
                                    compile_chat_template, render_chat)
from agent_from_scratch.tests.test_bench_skeletons import _model
from agent_from_scratch.tools.base import ToolCall

TEMPLATE = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
END = "<|im_end|>"
ENTRIES = {ctx.id: (skeleton, ctx) for skeleton, ctx in specs("train")}


def rollout(root: Path, task_id: str, *, before=(), script=None, overrides=None, name="case"):
    """Run one train task with its scripted solution; returns (record, task, trace).

    `before` prepends responses; `script` rewrites the solution's response list.
    """
    skeleton, ctx = ENTRIES[task_id]
    with TemporaryDirectory() as temporary:
        task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"),
                              Path(temporary), ctx)
    out = root / name
    responses = [*before, *(script or list)(skeleton.solution(task))]
    record = run_case(_model(responses), skeleton, ctx, out, overrides=overrides)
    trace = json.loads((out / "state" / "runs" / f"{record['run_id']}.jsonl").read_text())
    return record, json.loads((out / "task.json").read_text()), trace


def handmade(arguments: dict, *, content: str = "", raw_text: str | None = None) -> dict:
    """The smallest schema-5 trace with one completed tool-call request."""
    messages = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Go."},
                {"role": "assistant", "content": content, "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "write_file", "arguments": json.dumps(arguments)}}]}]
    return {"schema_version": 5, "messages": messages, "model_requests": [{
        "purpose": "agent", "status": "completed", "input_message_count": 2, "call_ids": ["c1"],
        "input_messages": messages[:2], "tools": {},
        "raw_response": None if raw_text is None else {"choices": [{"text": raw_text}]}}]}


class ExamplesFromTraceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_one_example_per_completed_actor_request(self):
        record, _, trace = rollout(self.root, "inventory_reconciliation-0-fault")
        examples = examples_from_trace(trace, TEMPLATE)
        self.assertEqual(len(examples), record["actor_requests"])
        self.assertEqual([e["request_index"] for e in examples], list(range(len(examples))))
        self.assertEqual(examples[0]["completion"],
                         '<tool_call>\n{"name": "read_file", "arguments": {"path": "inventory.json"}}\n</tool_call>' + END)
        self.assertEqual(examples[-1]["completion"], record["final_answer"] + END)
        for example in examples:
            self.assertTrue(example["prompt"].endswith("<|im_start|>assistant\n"))
            self.assertEqual(example["completion"].count(END), 1)

    def test_prompt_is_byte_identical_to_the_inference_prompt(self):
        _, _, trace = rollout(self.root, "points_save_recovery-0-fault")
        for example in examples_from_trace(trace, TEMPLATE):
            request = trace["model_requests"][example["request_index"]]
            self.assertEqual(example["prompt"],
                             render_chat(TEMPLATE, request["input_messages"], request["tools"]))

    def test_each_target_is_how_the_next_prompt_shows_that_turn(self):
        _, _, trace = rollout(self.root, "points_save_recovery-0-fault")
        examples = examples_from_trace(trace, TEMPLATE)
        for earlier, later in zip(examples, examples[1:]):
            self.assertTrue(later["prompt"].startswith(earlier["prompt"] + earlier["completion"] + "\n"))

    def test_target_parses_back_to_the_action_that_was_taken(self):
        _, _, trace = rollout(self.root, "points_save_recovery-0-fault")
        for example in examples_from_trace(trace, TEMPLATE):
            request = trace["model_requests"][example["request_index"]]
            recorded = trace["messages"][request["input_message_count"]]
            parsed = LLM.parse_response({"choices": [{"message": {
                "role": "assistant", "content": example["completion"].removesuffix(END)}}]})
            self.assertEqual([(c.name, c.arguments) for c in parsed.tool_calls],
                             [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                              for c in recorded.get("tool_calls", [])])

    def test_a_parse_error_request_is_never_a_target(self):
        record, _, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                   before=[ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)])
        self.assertTrue(record["passed"], record["checks"])
        examples = examples_from_trace(trace, TEMPLATE)
        self.assertEqual(len(examples), record["actor_requests"] - 1)
        self.assertNotIn(0, [e["request_index"] for e in examples])
        self.assertIn("[Runtime feedback]", examples[0]["prompt"])

    def test_escaped_arguments_and_narration_survive_the_round_trip(self):
        arguments = {"path": "a<b.txt", "content": "值 <tool_call> \"q\"\n"}
        example = examples_from_trace(handmade(arguments, content="Saving it."), TEMPLATE)[0]
        self.assertTrue(example["completion"].startswith("Saving it.\n<tool_call>\n"))
        self.assertIn("a\\u003cb.txt", example["completion"])
        parsed = LLM.parse_response({"choices": [{"message": {
            "role": "assistant", "content": example["completion"].removesuffix(END)}}]})
        self.assertEqual((parsed.content, parsed.tool_calls[0].arguments), ("Saving it.", arguments))

    def test_a_request_that_does_not_line_up_with_its_message_is_an_error(self):
        trace = handmade({"path": "a.txt", "content": "x"})
        trace["model_requests"][0]["call_ids"] = ["other"]
        with self.assertRaises(ValueError):
            examples_from_trace(trace, TEMPLATE)

    def test_sampled_text_is_compared_with_the_canonical_target(self):
        arguments = {"path": "a.txt", "content": "x"}
        same = '<tool_call>\n{"name": "write_file", "arguments": ' + json.dumps(arguments) + '}\n</tool_call>'
        for raw_text, expected in ((same, True), (same.replace(": {", ":{", 1), False), (None, None)):
            with self.subTest(raw_text=raw_text):
                example = examples_from_trace(handmade(arguments, raw_text=raw_text), TEMPLATE)[0]
                self.assertIs(example["raw_matches_canonical"], expected)


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_clean_and_injected_fault_runs_are_eligible(self):
        for task_id in ("inventory_reconciliation-0-clean", "inventory_reconciliation-0-fault",
                        "points_save_recovery-1-fault"):
            with self.subTest(task_id=task_id):
                self.assertIsNone(rejection(*rollout(self.root, task_id, name=task_id)))

    def test_a_failed_run_is_rejected(self):
        record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean")
        self.assertEqual(rejection({**record, "passed": False}, task, trace), "failed")

    def test_a_passed_run_with_the_models_own_mistakes_is_rejected(self):
        cases = {"parse": [ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)],
                 "wrong_path": [call("read_file", path="no-such-file.json")],
                 "bad_arguments": [call("read_file", nope=1)]}
        for name, before in cases.items():
            with self.subTest(name=name):
                record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                              before=before, name=name)
                self.assertTrue(record["passed"], record["checks"])
                self.assertEqual(rejection(record, task, trace), "own_error")

    def test_a_view_that_is_not_the_raw_prefix_is_rejected(self):
        record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                      overrides={"planning": True})
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(rejection(record, task, trace), "edited_view")


def batch(*paths):
    """One turn that reads several paths at once."""
    return LLMResponse("assistant", "", ResponseType.tool_call,
                       tool_calls=[ToolCall("read_file", {"path": path}) for path in paths])


def wrong_first(solution):
    """The scripted solution after one read of a file that does not exist."""
    return [call("read_file", path="no-such-file.json"), *solution]


class FlawedTurnTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_only_the_turn_holding_the_models_own_failed_call_is_flawed(self):
        # The shape seen in teacher data: one batch with a good read and a read of a directory.
        record, task, trace = rollout(
            self.root, "inventory_reconciliation-0-clean",
            script=lambda solution: [batch("inventory.json", "."), *solution[1:]])
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(flawed_turns(record, task, trace), {0: "own_error"})
        self.assertEqual(rejection(record, task, trace), "own_error")
        self.assertIsNone(rejection(record, task, trace, flawed="drop"))

    def test_the_injected_fault_and_a_call_skipped_after_it_are_not_flaws(self):
        # The fault hits the first call of the batch, so the second is skipped.
        record, task, trace = rollout(
            self.root, "inventory_reconciliation-0-fault",
            script=lambda solution: [batch("inventory.json", "shipment.json"), *solution[1:]])
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual([error["error_code"] for error in record["tool_errors"]],
                         ["execution_error", "skipped"])
        self.assertEqual(flawed_turns(record, task, trace), {})
        self.assertIsNone(rejection(record, task, trace))

    def test_a_loose_final_answer_flaws_the_last_turn_only(self):
        loose = lambda solution: [*solution[:-1], answer(solution[-1].content + "\n\nBoth balances verified.")]
        record, task, trace = rollout(self.root, "points_save_recovery-0-clean", script=loose)
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(flawed_turns(record, task, trace), {record["actor_requests"] - 1: "loose_answer"})
        self.assertEqual(rejection(record, task, trace), "loose_answer")
        self.assertIsNone(rejection(record, task, trace, flawed="drop"))

    def test_a_parse_error_rejects_the_whole_rollout_in_either_mode(self):
        record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                      before=[ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)])
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(rejection(record, task, trace, flawed="drop"), "own_error")


def trajectory(task_id, seed, completions, run_dir="r"):
    return {"task_id": task_id, "seed": seed, "run_dir": run_dir,
            "examples": [{"completion": text} for text in completions]}


class SelectTests(unittest.TestCase):
    def test_identical_trajectories_of_a_task_collapse_to_the_lowest_seed(self):
        kept = select([trajectory("t", 2, ["a", "b"]), trajectory("t", 1, ["a", "b"])], 5)
        self.assertEqual([(t["task_id"], t["seed"]) for t in kept], [("t", 1)])

    def test_cap_keeps_the_shortest_distinct_trajectories_per_task(self):
        pool = [trajectory("t", 0, ["a", "b", "c"]), trajectory("t", 1, ["a", "c"]),
                trajectory("t", 2, ["b", "c"]), trajectory("u", 0, ["a", "b", "c"])]
        kept = select(pool, 2)
        self.assertEqual([(t["task_id"], t["seed"]) for t in kept], [("t", 1), ("t", 2), ("u", 0)])

    def test_the_same_actions_on_different_tasks_are_both_kept(self):
        self.assertEqual(len(select([trajectory("t", 0, ["a"]), trajectory("u", 0, ["a"])], 1)), 2)

    def test_a_trajectory_without_dropped_turns_comes_first(self):
        flawed = trajectory("t", 0, ["x", "a"])
        flawed["examples"][0]["dropped"] = "own_error"
        kept = select([flawed, trajectory("t", 1, ["a", "b", "c"])], 1)
        self.assertEqual([t["seed"] for t in kept], [1])

    def test_a_dropped_answer_ranks_before_a_dropped_failure(self):
        # A failed call stays in the history of every later example; a loose answer touches nothing else.
        loose = trajectory("t", 0, ["a", "b", "x"])
        loose["examples"][2]["dropped"] = "loose_answer"
        failed = trajectory("t", 1, ["x", "a"])
        failed["examples"][0]["dropped"] = "own_error"
        self.assertEqual([t["seed"] for t in select([failed, loose], 1)], [0])


class CollectTests(unittest.TestCase):
    TASKS = ("inventory_reconciliation-0-clean", "inventory_reconciliation-1-clean")

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def run_dir(self, name, seed, *, split="train", backend="vllm", tasks=TASKS, script=None):
        """A bench-shaped run directory whose rollouts come from scripted solutions."""
        out = self.root / name
        out.mkdir()
        records = [rollout(out, task_id, name=task_id, script=script)[0] for task_id in tasks]
        save(out / "results.json", records)
        save(out / "metadata.json", {"suite": f"bench-{split}-v2", "split": split, "seed": seed,
                                     "git_revision": "0" * 40,
                                     "settings": {"backend": backend, "chat_template_sha256": None}})
        return out

    def test_repeated_rollouts_collapse_and_examples_carry_provenance(self):
        examples, metadata = collect([self.run_dir("s0", 0), self.run_dir("s1", 1)], "rft")
        funnel = metadata["funnel"]
        self.assertEqual((funnel["rollouts"], funnel["eligible"], funnel["kept_trajectories"],
                          funnel["tasks_covered"]), (4, 4, 2, 2))
        self.assertEqual((funnel["examples_before_dedup"], funnel["examples"], len(examples)), (6, 6, 6))
        self.assertEqual(len({e["id"] for e in examples}), len(examples))
        provenance = examples[0]["provenance"]
        self.assertEqual((provenance["track"], provenance["seed"], provenance["run_dir"],
                          provenance["skeleton"]), ("rft", 0, "s0", "inventory_reconciliation"))
        trace = json.loads((self.root / "s0" / provenance["task_id"] / "state" / "runs"
                            / f"{provenance['run_id']}.jsonl").read_text())
        self.assertEqual(trace["model_requests"][provenance["request_index"]]["status"], "completed")
        self.assertEqual(metadata["funnel"]["by_skeleton"],
                         {"inventory_reconciliation": {"trajectories": 2, "tasks": 2}})

    def test_two_distinct_trajectories_share_their_common_prefix_once(self):
        # Seed 1 re-reads the second source before answering: same first two decisions, then it
        # diverges. Prompts never match across runs (each names its own temporary workspace),
        # so duplicates are found by task and action history, not by prompt text.
        task = self.TASKS[:1]
        again = lambda solution: [*solution[:2], solution[1], solution[2]]
        examples, metadata = collect([self.run_dir("s0", 0, tasks=task),
                                      self.run_dir("s1", 1, tasks=task, script=again)], "rft")
        funnel = metadata["funnel"]
        self.assertEqual((funnel["kept_trajectories"], funnel["examples_before_dedup"],
                          funnel["examples"]), (2, 7, 5))
        self.assertEqual([e["provenance"]["seed"] for e in examples], [0, 0, 0, 1, 1])
        self.assertEqual(collect([self.run_dir("t0", 0, tasks=task),
                                  self.run_dir("t1", 1, tasks=task, script=again)], "rft",
                                 max_per_task=1)[1]["funnel"]["examples"], 3)

    def test_drop_mode_keeps_a_flawed_rollout_without_its_flawed_turn(self):
        run = self.run_dir("s0", 0, tasks=self.TASKS[:1], script=wrong_first)
        examples, metadata = collect([run], "rft")
        self.assertEqual((examples, metadata["funnel"]["rejected"]["own_error"], metadata["flawed_turns"]),
                         ([], 1, "reject"))
        examples, metadata = collect([run], "rft", flawed="drop")
        funnel = metadata["funnel"]
        self.assertEqual((metadata["flawed_turns"], funnel["eligible"], funnel["examples_before_dedup"],
                          funnel["examples"]), ("drop", 1, 3, 3))
        self.assertEqual(funnel["dropped_turns"], {"own_error": 1, "loose_answer": 0})
        self.assertEqual([e["provenance"]["request_index"] for e in examples], [1, 2, 3])
        self.assertEqual([e["provenance"]["dropped_turns"] for e in examples], [[0]] * 3)
        # The failed call stays in the history the later turns saw; it is never a target.
        self.assertIn("no-such-file.json", examples[0]["prompt"])
        self.assertFalse(any("no-such-file.json" in e["completion"] for e in examples))

    def test_a_flawed_trajectory_ranks_after_a_clean_one_and_is_not_its_duplicate(self):
        task = self.TASKS[:1]
        runs = [self.run_dir("s0", 0, tasks=task, script=wrong_first), self.run_dir("s1", 1, tasks=task)]
        examples, metadata = collect(runs, "rft", flawed="drop")
        self.assertEqual((metadata["funnel"]["kept_trajectories"], metadata["funnel"]["examples"]), (2, 6))
        self.assertEqual([e["provenance"]["seed"] for e in examples], [1, 1, 1, 0, 0, 0])
        self.assertEqual([e["provenance"]["dropped_turns"] for e in examples], [[]] * 3 + [[0]] * 3)
        only = collect(runs, "rft", flawed="drop", max_per_task=1)[0]
        self.assertEqual({e["provenance"]["seed"] for e in only}, {1})

    def test_only_the_train_split_can_become_training_data(self):
        for split in ("dev", "test", "holdout"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train"):
                collect([self.run_dir(split, 0, split=split)], "rft")

    def test_the_backend_must_match_the_track_unless_marked_pilot(self):
        local = self.run_dir("local", 0, backend="llama_cpp")
        with self.assertRaisesRegex(ValueError, "backend"):
            collect([local], "rft")
        self.assertTrue(collect([local], "rft", pilot=True)[1]["pilot"])
        with self.assertRaisesRegex(ValueError, "backend"):
            collect([self.run_dir("api", 0, backend="deepseek")], "rft")
        with self.assertRaisesRegex(ValueError, "[Ss]cripted"):
            collect([self.run_dir("gate", 0, backend="scripted")], "rft", pilot=True)

    def test_the_same_task_at_the_same_seed_twice_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            collect([self.run_dir("a", 0), self.run_dir("b", 0)], "rft")

    def test_a_run_sampled_under_another_template_is_an_error(self):
        run = self.run_dir("old", 0)
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["settings"]["chat_template_sha256"] = "f" * 64
        save(run / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "template"):
            collect([run], "rft")

    def test_command_writes_once_and_never_overwrites(self):
        run, out = self.run_dir("s0", 0), self.root / "sft"
        argv = ["export", "--track", "rft", "--output", str(out), str(run)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            export.main()
        lines = (out / "examples.jsonl").read_text().splitlines()
        metadata = json.loads((out / "metadata.json").read_text())
        self.assertEqual(len(lines), metadata["funnel"]["examples"])
        self.assertEqual(set(json.loads(lines[0])), {"id", "prompt", "completion", "provenance"})
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as error:
            export.main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((out / "examples.jsonl").read_text().splitlines(), lines)

    def test_command_takes_the_flawed_turn_mode(self):
        run, out = self.run_dir("s0", 0), self.root / "sft"
        argv = ["export", "--track", "rft", "--flawed-turns", "drop", "--output", str(out), str(run)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            export.main()
        self.assertEqual(json.loads((out / "metadata.json").read_text())["flawed_turns"], "drop")


if __name__ == "__main__":
    unittest.main()
