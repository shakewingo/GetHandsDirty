"""A task with follow-up turns is played as one conversation, scored after its last turn, and
exported with one example per model request of every turn."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.bench.run import run_case
from agent_from_scratch.evals.bench.skeletons import Skeleton, answer, call
from agent_from_scratch.evals.bench.spec import Answer, BuildContext, Expect, Task
from agent_from_scratch.evals.run import save
from agent_from_scratch.evals.sft.export import collect
from agent_from_scratch.tests.test_bench_skeletons import _model


def _build(rng, root, ctx):
    (root / "note.txt").write_text("title=note\ncode=K123\n")
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
                prompt="Read note.txt and reply only its code.",
                followups=("Now save that code to answer.txt and reply SAVED.",),
                tools=("read_file", "write_file"),
                expect=Expect(answer=Answer("SAVED"), files={"answer.txt": "K123\n"}, evidence=("code=K123",)))


def _solution(task):
    return [call("read_file", path="note.txt"), answer("K123"),
            call("write_file", path="answer.txt", content="K123\n"), answer("SAVED")]


PROBE = Skeleton("two_turn_probe", "updates", "train", (0,), False, _build, _solution, strict=True)
CTX = BuildContext("two_turn_probe", "updates", "train", 0, None)


class MultiTurnRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def run_script(self, responses, name="run"):
        return run_case(_model(responses), PROBE, CTX, self.root / name)

    def test_both_turns_are_played_in_one_conversation(self):
        record = self.run_script(_solution(None))
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(len(record["run_ids"]), 2)
        self.assertEqual(record["model_requests"], 4)
        second = json.loads((self.root / "run" / "state" / "runs" / f"{record['run_ids'][1]}.jsonl").read_text())
        users = [m["content"] for m in second["messages"] if m["role"] == "user"]
        self.assertEqual(users, ["Read note.txt and reply only its code.",
                                 "Now save that code to answer.txt and reply SAVED."])

    def test_evidence_from_an_earlier_turn_counts(self):
        # The code is read only in turn 1; turn 2 writes without reading again.
        record = self.run_script(_solution(None))
        self.assertTrue(record["checks"]["evidence"])

    def test_a_turn_that_does_not_end_normally_fails_the_task_and_ends_it(self):
        # Turn 1 never answers: it repeats a call until the request limit.
        stuck = [call("read_file", path="note.txt")] * 12
        record = self.run_script(stuck)
        self.assertFalse(record["checks"]["normal_finish"])
        self.assertFalse(record["passed"])
        self.assertEqual(len(record["run_ids"]), 1)

    def test_asking_instead_of_acting_in_the_last_turn_fails(self):
        script = _solution(None)[:2] + [answer("Which code do you mean?")]
        record = self.run_script(script)
        self.assertFalse(record["checks"]["files"])
        self.assertFalse(record["passed"])


class MultiTurnExportTests(unittest.TestCase):
    def test_every_turn_becomes_examples_and_later_prompts_hold_earlier_turns(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        run = root / "s0"
        run.mkdir()
        record = run_case(_model(_solution(None)), PROBE, CTX, run / CTX.id)
        save(run / "results.json", [record])
        save(run / "metadata.json", {"suite": "bench-train-v2", "split": "train", "seed": 0,
                                     "git_revision": "0" * 40,
                                     "settings": {"backend": "vllm", "chat_template_sha256": None}})
        examples, metadata = collect([run], "rft")
        self.assertEqual([e["provenance"]["turn"] for e in examples], [0, 0, 1, 1])
        self.assertEqual(examples[-1]["completion"], "SAVED<|im_end|>")
        self.assertIn("Read note.txt and reply only its code.", examples[2]["prompt"])
        self.assertIn("K123<|im_end|>", examples[2]["prompt"])  # turn 1's reply is history in turn 2


class MultiTurnManifestTests(unittest.TestCase):
    def test_follow_ups_are_frozen_only_where_a_task_has_them(self):
        from agent_from_scratch.evals.bench import manifest
        from unittest.mock import patch

        with patch.object(manifest, "specs", return_value=[(PROBE, CTX)]):
            tasks = manifest.build_manifest()["tasks"]
        self.assertIn("followups_sha256", tasks[0])
        frozen = json.loads(manifest.MANIFEST_PATH.read_text())["tasks"]
        self.assertTrue(all("followups_sha256" not in task for task in frozen))


if __name__ == "__main__":
    unittest.main()
