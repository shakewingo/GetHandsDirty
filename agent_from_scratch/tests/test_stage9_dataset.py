"""Public generation and scripted-check entry points require no model/API."""

import json
from pathlib import Path
import subprocess
import sys
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.evals.bench import run
from agent_from_scratch.evals.bench.manifest import build_manifest
from agent_from_scratch.config import BENCH_DECODING, BENCH_MODEL, BENCH_N_CTX
from agent_from_scratch.tests.test_bench_skeletons import _model


class DatasetCommandTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, "-m", "agent_from_scratch.evals", "dataset", *map(str, args)],
                              capture_output=True, text=True)

    def test_generates_only_requested_split_with_private_contracts(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "data"
            result = self.command("--split", "train", "--output", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["task_counts"], {"train": 320})
            tasks = list((out / "tasks").glob("*/task.json"))
            self.assertEqual(len(tasks), 320)
            for path in tasks:
                task = json.loads(path.read_text())
                self.assertEqual(task["split"], "train")
                self.assertNotIn("task.json", [p.name for p in (path.parent / "workspace").iterdir()])

    def test_scripted_check_accepts_solutions_and_rejects_negative_runs(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "data"
            result = self.command("--output", out, "--check")
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            # 320 train and 140 test tasks.
            self.assertEqual(summary["correct"], {"passed": 460, "attempts": 460})
            self.assertEqual(summary["answer_only"], {"passed": 0, "attempts": 460})
            self.assertEqual(summary["wrong_answer"], {"passed": 0, "attempts": 460})
            self.assertTrue(summary["gates_passed"])

    def test_existing_output_is_never_overwritten(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            sentinel = out / "keep.txt"
            sentinel.write_text("keep")
            result = self.command("--output", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_test_split_runs_require_final_and_a_manifest(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(json.dumps(build_manifest()))
            base = ["bench", "--split", "test", "--output", str(Path(temporary) / "run")]
            for flags in ([], ["--final"], ["--manifest", str(manifest)],
                          ["--final", "--manifest", str(manifest), "--limits", '{"planning": true}']):
                with self.subTest(flags=flags), patch.object(sys, "argv", base + flags):
                    with self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_supplied_manifest_drift_blocks_test_before_loading_model(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            manifest = Path(temporary) / "manifest.json"
            content = build_manifest()
            content["memory"] = "on"
            manifest.write_text(json.dumps(content))
            args = ["bench", "--split", "test", "--final", "--manifest", str(manifest),
                    "--output", str(Path(temporary) / "run")]
            with patch.object(sys, "argv", args), self.assertRaises(SystemExit):
                run.main()
            model.assert_not_called()

    def test_train_and_versioned_test_reach_the_existing_runner(self):
        for split in ("train", "test"):
            with self.subTest(split=split), TemporaryDirectory() as temporary:
                root = Path(temporary)
                skeleton, ctx = run.specs(split)[0]
                fixture = root / "fixture"
                fixture.mkdir()
                task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), fixture, ctx)
                model = _model(skeleton.solution(task))
                model.settings.return_value = {"backend": "vllm", "model_root": BENCH_MODEL["id"],
                                               "n_ctx": BENCH_N_CTX, **BENCH_DECODING}
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps(build_manifest()))
                output = root / "run"
                args = ["bench", "--split", split, "--tasks", ctx.id, "--backend", "vllm",
                        "--output", str(output)]
                if split == "test":
                    args += ["--final", "--manifest", str(manifest)]
                with patch.object(sys, "argv", args), patch.object(run, "load_model", return_value=model):
                    run.main()
                record = json.loads((output / ctx.id / "result.json").read_text())
                self.assertTrue(record["passed"], record["checks"])

    def test_invalid_manifest_is_a_cli_error_before_loading_model(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            manifest = Path(temporary) / "manifest.json"
            args = ["bench", "--split", "test", "--final", "--manifest", str(manifest),
                    "--output", str(Path(temporary) / "run")]
            for content in ("{broken", "[]", "null", '{"tasks": {}}', '{"tasks": [null]}'):
                with self.subTest(content=content):
                    manifest.write_text(content)
                    with patch.object(sys, "argv", args), self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
