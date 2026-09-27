"""Declarative task shapes are frozen and JSON-serializable."""

from dataclasses import FrozenInstanceError, asdict
import json
import unittest

from agent_from_scratch.evals.bench.spec import Answer, BuildContext, Expect, Fault, Task


class SpecTests(unittest.TestCase):
    def test_answer_rejects_an_unknown_format(self):
        Answer(value="X")  # default "advisory" is fine
        with self.assertRaises(ValueError):
            Answer(value="X", format="exact")

    def test_task_is_frozen_and_asdict_is_json_serializable(self):
        task = Task(id="t-0", skeleton="t", family="inspection", split="dev",
                    prompt="p", tools=("read_file",), expect=Expect(answer=Answer(value="X")))
        with self.assertRaises(FrozenInstanceError):
            task.id = "other"
        json.dumps(asdict(task))  # must not raise

    def test_build_context_computes_id_and_pair_id(self):
        plain = BuildContext(name="s", family="f", split="dev", seed=2, condition=None)
        self.assertEqual((plain.id, plain.pair_id), ("s-2", None))
        paired = BuildContext(name="s", family="f", split="dev", seed=2, condition="fault")
        self.assertEqual((paired.id, paired.pair_id), ("s-2-fault", "s-2"))


if __name__ == "__main__":
    unittest.main()
