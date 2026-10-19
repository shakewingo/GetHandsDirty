"""The training loop moves only the adapter, learns the masked targets, and survives a reload."""

from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.config import CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.run import save
from agent_from_scratch.evals.sft import train
from agent_from_scratch.evals.sft.tensorize import IGNORE
from agent_from_scratch.evals.sft.train import add_lora, fit, load_dataset, policy_step

TEMPLATE_SHA256 = sha256(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def dataset(root: Path, *, pilot=False, template=TEMPLATE_SHA256) -> Path:
    """An export-shaped directory with two examples."""
    directory = root / "sft"
    directory.mkdir()
    examples = [{"id": f"e{n}", "prompt": "p", "completion": "c<|im_end|>",
                 "provenance": {"skeleton": "s", "task_id": f"t{n}", "run_dir": "r"}} for n in range(2)]
    (directory / "examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    save(directory / "metadata.json", {"track": "teacher", "pilot": pilot, "flawed_turns": "drop",
                                       "max_per_task": 1, "chat_template_sha256": template})
    return directory


class LoadDatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_examples_and_metadata_are_read(self):
        examples, metadata = load_dataset(dataset(self.root))
        self.assertEqual(([e["id"] for e in examples], metadata["track"]), (["e0", "e1"], "teacher"))

    def test_a_line_separator_inside_a_string_does_not_split_a_row(self):
        directory = dataset(self.root)
        row = {"id": "e9", "prompt": "a b", "completion": "c <|im_end|>",
               "provenance": {"skeleton": "s", "task_id": "t9", "run_dir": "r"}}
        with (directory / "examples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        examples, _ = load_dataset(directory)
        self.assertEqual(examples[-1], row)
        self.assertEqual(len(examples), 3)

    def test_a_pilot_dataset_needs_the_pilot_flag(self):
        directory = dataset(self.root, pilot=True)
        with self.assertRaisesRegex(ValueError, "pilot"):
            load_dataset(directory)
        self.assertEqual(len(load_dataset(directory, pilot=True)[0]), 2)

    def test_a_dataset_exported_under_another_template_is_refused(self):
        with self.assertRaisesRegex(ValueError, "template"):
            load_dataset(dataset(self.root, template="f" * 64))

    def test_command_refuses_before_any_model_is_loaded(self):
        directory, existing = dataset(self.root, pilot=True), self.root / "taken"
        existing.mkdir()
        for output in (self.root / "out", existing):
            argv = ["train", "--examples", str(directory), "--output", str(output)]
            with self.subTest(output=output.name), patch.object(sys, "argv", argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    train.main()
                self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "out").exists())

    def test_the_module_loads_without_the_training_libraries(self):
        code = ("import sys; import agent_from_scratch.evals.sft.train; "
                "sys.exit(any(name in sys.modules for name in ('torch', 'transformers', 'peft')))")
        repository = Path(__file__).resolve().parents[2]
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=repository).returncode, 0)


def tiny_model(seed: int = 0):
    """A two-layer Qwen3-shaped model with random weights; small enough for a CPU test."""
    import torch
    import transformers

    torch.manual_seed(seed)
    # The output head stays frozen, so its random weights must be large enough for an adapter
    # to produce confident logits through it: at the default 0.02 the loss cannot fall far.
    config = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64, initializer_range=0.5)
    return transformers.AutoModelForCausalLM.from_config(config)


def toy_rows():
    """Four rows that share a prefix; only the last three tokens of each carry a label."""
    rows = []
    for n in range(4):
        ids = [1, 2, 3, 4, 5 + n, 10 + n, 20 + n, 30 + n]
        rows.append({"id": f"r{n}", "input_ids": ids, "labels": [IGNORE] * 5 + ids[5:]})
    return rows


class FitTests(unittest.TestCase):
    def setUp(self):
        try:
            import peft  # noqa: F401
            import torch
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("torch, transformers and peft are not installed (requirements-train.txt)")
        # A GPU pod reports the host's cores (128) under a much smaller CPU quota (13.6): torch's
        # threads then spin and get throttled, and these 11-second tests ran past 47 minutes.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.torch = torch

    def adapted(self):
        return add_lora(tiny_model(), rank=8, alpha=16, dropout=0.0)

    def test_one_record_per_optimizer_step_and_the_loss_falls(self):
        model, epochs = self.adapted(), []
        records = fit(model, toy_rows(), epochs=60, lr=1e-2, accumulate=2, seed=0, on_epoch=epochs.append)
        self.assertEqual(len(records), 60 * 2)
        self.assertEqual((records[0]["step"], records[0]["epoch"], records[0]["rows"]), (1, 1, 2))
        self.assertEqual(epochs, list(range(1, 61)))
        self.assertLess(records[-1]["loss"], 0.05 * records[0]["loss"])
        self.assertLess(records[-1]["lr"], records[len(records) // 2]["lr"])  # decays after warmup
        self.assertLess(records[0]["lr"], records[11]["lr"])                  # rises during warmup

    def test_a_short_last_group_is_averaged_over_its_own_rows(self):
        records = fit(self.adapted(), toy_rows()[:3], epochs=1, lr=1e-3, accumulate=2, seed=0)
        self.assertEqual([record["rows"] for record in records], [2, 1])

    def test_the_recorded_loss_is_the_mean_of_the_rows_own_losses(self):
        model, rows = self.adapted(), toy_rows()
        with self.torch.no_grad():
            losses = [model(input_ids=self.torch.tensor([row["input_ids"]]),
                            labels=self.torch.tensor([row["labels"]])).loss.item() for row in rows]
        record = fit(model, rows, epochs=1, lr=0.0, accumulate=4, seed=0)[0]
        self.assertAlmostEqual(record["loss"], sum(losses) / len(losses), places=5)

    def test_the_rate_warms_up_from_the_first_step_and_decays_toward_zero(self):
        # 20 optimizer steps, so 2 of warmup.
        records = fit(self.adapted(), toy_rows(), epochs=5, lr=1e-3, accumulate=1, seed=0)
        self.assertAlmostEqual(records[0]["lr"], 5e-4)
        self.assertAlmostEqual(records[1]["lr"], 1e-3)
        self.assertTrue(0.0 < records[-1]["lr"] < 1e-4, records[-1]["lr"])

    def test_the_seed_alone_decides_the_row_order(self):
        runs = []
        for state in (1, 2):
            random.seed(state)  # the global generator must not matter
            runs.append(fit(self.adapted(), toy_rows(), epochs=2, lr=1e-2, accumulate=1, seed=0))
        self.assertEqual(runs[0], runs[1])
        other = fit(self.adapted(), toy_rows(), epochs=2, lr=1e-2, accumulate=1, seed=1)
        self.assertNotEqual([record["loss"] for record in other], [record["loss"] for record in runs[0]])

    def test_only_the_adapter_moves(self):
        model = self.adapted()
        before = {name: value.clone() for name, value in model.named_parameters()}
        fit(model, toy_rows(), epochs=3, lr=1e-2, accumulate=2, seed=0)
        moved = {name for name, value in model.named_parameters() if not self.torch.equal(value, before[name])}
        self.assertTrue(moved)
        self.assertTrue(all("lora_" in name for name in moved), sorted(moved)[:3])

    def optimizer(self, model, lr):
        return self.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                      weight_decay=0.0)

    def test_unit_weights_match_one_fit_step(self):
        rows = toy_rows()
        fitted = self.adapted()
        fit(fitted, rows, epochs=1, lr=1e-2, accumulate=len(rows), seed=0)  # one step at the full rate
        stepped = self.adapted()
        result = policy_step(stepped, self.optimizer(stepped, 1e-2), [{**row, "weight": 1.0} for row in rows],
                             lr=1e-2)
        for (name, a), (_, b) in zip(fitted.named_parameters(), stepped.named_parameters()):
            self.assertTrue(self.torch.allclose(a, b, atol=1e-6), name)
        self.assertEqual((result["rows"], result["tokens"], result["mean_weight"]), (4, 12, 1.0))

    def test_a_negative_weight_lowers_that_rows_likelihood(self):
        model, rows = self.adapted(), toy_rows()[:2]

        def losses():
            with self.torch.no_grad():
                return [model(input_ids=self.torch.tensor([row["input_ids"]]),
                              labels=self.torch.tensor([row["labels"]])).loss.item() for row in rows]
        before = losses()
        optimizer = self.optimizer(model, 1e-2)
        weighted = [{**rows[0], "weight": 1.0}, {**rows[1], "weight": -1.0}]
        for _ in range(5):
            policy_step(model, optimizer, weighted, lr=1e-2)
        after = losses()
        self.assertLess(after[0], before[0])
        self.assertGreater(after[1], before[1])

    def test_only_the_adapter_moves_in_a_policy_step(self):
        model = self.adapted()
        before = {name: value.clone() for name, value in model.named_parameters()}
        rows = [{**row, "weight": weight} for row, weight in zip(toy_rows(), (1.0, -1.0, 0.5, -0.5))]
        policy_step(model, self.optimizer(model, 1e-2), rows, lr=1e-2)
        moved = {name for name, value in model.named_parameters() if not self.torch.equal(value, before[name])}
        self.assertTrue(moved)
        self.assertTrue(all("lora_" in name for name in moved), sorted(moved)[:3])

    def test_masked_positions_carry_no_loss(self):
        rows = toy_rows()
        # Same inputs, but labels on the prompt tokens too: a different objective, so different losses.
        unmasked = [{**row, "labels": list(row["input_ids"])} for row in rows]
        masked_loss = fit(self.adapted(), rows, epochs=1, lr=0.0, accumulate=4, seed=0)[0]["loss"]
        unmasked_loss = fit(self.adapted(), unmasked, epochs=1, lr=0.0, accumulate=4, seed=0)[0]["loss"]
        self.assertNotAlmostEqual(masked_loss, unmasked_loss, places=3)
        # Changing a label that is masked changes nothing.
        relabelled = [{**row, "labels": [IGNORE] * 5 + row["labels"][5:]} for row in unmasked]
        self.assertAlmostEqual(fit(self.adapted(), relabelled, epochs=1, lr=0.0, accumulate=4, seed=0)[0]["loss"],
                               masked_loss, places=6)

    def test_a_saved_adapter_reloads_onto_a_fresh_base_with_the_same_outputs(self):
        from peft import PeftModel

        model, rows = self.adapted(), toy_rows()
        fit(model, rows, epochs=60, lr=1e-2, accumulate=2, seed=0)
        model.eval()
        ids = self.torch.tensor([rows[0]["input_ids"]])
        with TemporaryDirectory() as temporary, self.torch.no_grad():
            model.save_pretrained(temporary)
            files = {path.name for path in Path(temporary).iterdir()}
            self.assertLessEqual({"adapter_config.json", "adapter_model.safetensors"}, files)
            reloaded = PeftModel.from_pretrained(tiny_model(), temporary).eval()
            self.assertTrue(self.torch.allclose(model(input_ids=ids).logits, reloaded(input_ids=ids).logits, atol=1e-5))
            base = tiny_model().eval()
            self.assertFalse(self.torch.allclose(base(input_ids=ids).logits, reloaded(input_ids=ids).logits, atol=1e-3))
            # The overfitted adapter writes each row's targets; the base alone does not.
            predicted = reloaded(input_ids=ids).logits[0, 4:7].argmax(-1).tolist()
            self.assertEqual(predicted, rows[0]["input_ids"][5:])
            self.assertNotEqual(base(input_ids=ids).logits[0, 4:7].argmax(-1).tolist(), rows[0]["input_ids"][5:])


if __name__ == "__main__":
    unittest.main()
