"""The mix repeats our examples a whole number of times, keeps every public one, and trains unchanged."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.config import CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.run import save
from agent_from_scratch.evals.sft.mix import combine, repeat_factor
from agent_from_scratch.evals.sft.train import load_dataset

TEMPLATE_SHA256 = sha256(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def examples(prefix: str, count: int) -> list[dict]:
    return [{"id": f"{prefix}{n}", "prompt": "p", "completion": "c<|im_end|>",
             "provenance": {"skeleton": prefix, "task_id": f"t{n}", "run_dir": prefix}} for n in range(count)]


def dataset(root: Path, name: str, rows: list[dict], *, track: str, template=TEMPLATE_SHA256, pilot=False) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    save(directory / "metadata.json", {"track": track, "pilot": pilot, "flawed_turns": "drop" if track != "public" else None,
                                       "max_per_task": 1 if track != "public" else None,
                                       "chat_template_sha256": template})
    return directory


class RepeatFactorTests(unittest.TestCase):
    def test_a_third_of_the_rows_from_the_approved_sizes(self):
        # 5,000 public rows: the own rows should number 2,500.
        self.assertEqual(repeat_factor(1000, 5000, 1 / 3), 2)  # 2.5 rounds to even: 2,000 own rows
        self.assertEqual(repeat_factor(800, 5000, 1 / 3), 3)   # 3.125: 2,400 own rows
        self.assertEqual(repeat_factor(5000, 5000, 1 / 3), 1)  # never fewer than one copy

    def test_an_impossible_share_is_refused(self):
        for share in (0, 1, 1.5):
            with self.subTest(share=share), self.assertRaises(ValueError):
                repeat_factor(10, 10, share)


class CombineTests(unittest.TestCase):
    def test_own_rows_appear_k_times_and_public_rows_once(self):
        own, public = examples("own", 4), [examples("a", 10), examples("b", 14)]
        rows, k = combine(own, public, share=1 / 3, seed=0)
        self.assertEqual(k, 3)
        counts = Counter(row["id"] for row in rows)
        self.assertEqual({counts[e["id"]] for e in own}, {3})
        self.assertEqual({counts[e["id"]] for e in public[0] + public[1]}, {1})
        self.assertEqual(sorted(row["provenance"]["copy"] for row in rows if row["id"] == "own0"), [0, 1, 2])

    def test_the_seed_alone_decides_the_order(self):
        own, public = examples("own", 4), [examples("a", 10)]
        first, _ = combine(own, public, share=1 / 3, seed=1)
        again, _ = combine(own, public, share=1 / 3, seed=1)
        other, _ = combine(own, public, share=1 / 3, seed=2)
        self.assertEqual(first, again)
        self.assertNotEqual([r["id"] for r in first], [r["id"] for r in other])


class MixCommandTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def run_mix(self, own, *public):
        out = self.root / "mix"
        result = subprocess.run([sys.executable, "-m", "agent_from_scratch.evals", "mix", "--own", str(own),
                                 "--public", *map(str, public), "--output", str(out)], capture_output=True, text=True)
        return result, out

    def test_the_output_loads_as_a_training_dataset_with_its_sources(self):
        own = dataset(self.root, "own", examples("own", 4), track="teacher")
        public = dataset(self.root, "pub", examples("a", 16), track="public")
        result, out = self.run_mix(own, public)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows, metadata = load_dataset(out)
        self.assertEqual((len(rows), metadata["repeat_own"], metadata["own_share"]), (24, 2, 0.3333))
        self.assertEqual(metadata["own"]["directory"], "own")
        self.assertEqual([p["examples"] for p in metadata["public"]], [16])

    def test_a_pilot_or_a_foreign_template_or_swapped_roles_is_refused(self):
        own = dataset(self.root, "own", examples("own", 4), track="teacher")
        cases = {"pilot": dataset(self.root, "pilot", examples("a", 4), track="public", pilot=True),
                 "template": dataset(self.root, "foreign", examples("a", 4), track="public", template="0" * 64),
                 "roles": dataset(self.root, "teacher2", examples("a", 4), track="teacher")}
        for name, public in cases.items():
            with self.subTest(name=name):
                result, out = self.run_mix(own, public)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
