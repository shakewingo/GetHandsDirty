"""Training rows carry loss on the completion only, and nothing is truncated or guessed."""

import re
import unittest

from agent_from_scratch.evals.sft.tensorize import IGNORE, encode_example, select_examples

PROMPT = "<|im_start|>user\nRead a.json<|im_end|>\n<|im_start|>assistant\n"
CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.json"}}\n</tool_call><|im_end|>'
MARKERS = r"<\|im_start\|>|<\|im_end\|>|</?tool_call>"


def tokenizer(pattern: str):
    """A stand-in tokenizer with character offsets; each distinct piece gets its own id."""
    vocabulary: dict[str, int] = {}

    def tokenize(text: str):
        pieces = list(re.finditer(pattern, text, re.S))
        return [vocabulary.setdefault(m.group(), len(vocabulary)) for m in pieces], [m.span() for m in pieces]
    return tokenize


WORDS = tokenizer(MARKERS + r"|\n|\w+|.")
# This one glues the newline that ends the prompt to a capitalised word that starts the completion.
GLUED = tokenizer(MARKERS + r"|\n[A-Z]+|\n|\w+|.")


def example(completion=CALL, prompt=PROMPT, **provenance):
    return {"id": "e1", "prompt": prompt, "completion": completion, "provenance": provenance}


class EncodeExampleTests(unittest.TestCase):
    def test_labels_cover_the_completion_and_nothing_else(self):
        row = encode_example(example(), WORDS)
        prompt_tokens = len(WORDS(PROMPT)[0])
        self.assertEqual(row["id"], "e1")
        self.assertEqual(row["input_ids"], WORDS(PROMPT + CALL)[0])
        self.assertEqual(row["labels"][:prompt_tokens], [IGNORE] * prompt_tokens)
        self.assertEqual(row["labels"][prompt_tokens:], row["input_ids"][prompt_tokens:])
        self.assertEqual(len(row["labels"]), len(row["input_ids"]))

    def test_the_end_marker_is_the_last_trained_token(self):
        row = encode_example(example("BLOCKED<|im_end|>"), WORDS)
        self.assertEqual([label for label in row["labels"] if label != IGNORE],
                         WORDS("BLOCKED<|im_end|>")[0][-2:])
        for broken in ("BLOCKED", "BLOCKED<|im_end|>\n"):
            with self.subTest(completion=broken), self.assertRaisesRegex(ValueError, "end"):
                encode_example(example(broken), WORDS)

    def test_a_token_across_the_boundary_is_an_error(self):
        self.assertEqual(encode_example(example("NONE<|im_end|>"), WORDS)["labels"][-2:],
                         WORDS("NONE<|im_end|>")[0])
        with self.assertRaisesRegex(ValueError, "boundary"):
            encode_example(example("NONE<|im_end|>"), GLUED)

    def test_a_row_longer_than_the_limit_is_refused_not_truncated(self):
        tokens = len(WORDS(PROMPT + CALL)[0])
        self.assertEqual(len(encode_example(example(), WORDS, max_tokens=tokens)["input_ids"]), tokens)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            encode_example(example(), WORDS, max_tokens=tokens - 1)

    def test_a_tokenizer_that_loses_offsets_is_an_error(self):
        with self.assertRaises(ValueError):
            encode_example(example(), lambda text: ([1, 2, 3], [(0, 1)]))


def rows(*trajectories):
    """Examples in file order; each trajectory is (skeleton, task, number of examples)."""
    return [{"id": f"{task}-{index}", "prompt": "", "completion": "",
             "provenance": {"skeleton": skeleton, "task_id": task, "run_dir": "s0"}}
            for skeleton, task, count in trajectories for index in range(count)]


class SelectExamplesTests(unittest.TestCase):
    def test_no_cap_keeps_everything_in_order(self):
        examples = rows(("a", "a-0", 2), ("b", "b-0", 3))
        self.assertEqual(select_examples(examples), examples)

    def test_cap_keeps_whole_trajectories_per_skeleton(self):
        examples = rows(("a", "a-0", 2), ("a", "a-1", 2), ("a", "a-2", 2), ("b", "b-0", 7), ("b", "b-1", 3))
        kept = select_examples(examples, max_per_skeleton=5)
        self.assertEqual([e["provenance"]["task_id"] for e in kept], ["a-0", "a-0", "a-1", "a-1", "b-1", "b-1", "b-1"])

    def test_the_same_task_from_two_runs_is_two_trajectories(self):
        examples = rows(("a", "a-0", 2)) + [{**e, "provenance": {**e["provenance"], "run_dir": "s1"}}
                                             for e in rows(("a", "a-0", 2))]
        self.assertEqual(len(select_examples(examples, max_per_skeleton=3)), 2)

    def test_limit_takes_the_first_examples_after_the_cap(self):
        examples = rows(("a", "a-0", 2), ("a", "a-1", 2), ("b", "b-0", 2))
        self.assertEqual([e["id"] for e in select_examples(examples, max_per_skeleton=2, limit=3)],
                         ["a-0-0", "a-0-1", "b-0-0"])


if __name__ == "__main__":
    unittest.main()
