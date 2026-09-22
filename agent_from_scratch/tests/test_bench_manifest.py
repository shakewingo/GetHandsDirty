"""The freeze manifest is reproducible and reports drift when a frozen file changes."""

from pathlib import Path
from unittest.mock import patch
import unittest

from agent_from_scratch.evals.bench.manifest import build_manifest, manifest_drift, MANIFEST_PATH


class ManifestTests(unittest.TestCase):
    def test_build_manifest_is_reproducible_and_has_no_model_dependent_fields(self):
        first, second = build_manifest(), build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["memory"], "off")
        self.assertEqual(len(first["tasks"]), 75)  # 15 dev + 60 test

    def test_manifest_drift_is_empty_before_any_freeze(self):
        with patch("agent_from_scratch.evals.bench.manifest.MANIFEST_PATH", Path("/nonexistent.json")):
            self.assertEqual(manifest_drift(), {})

    def test_manifest_drift_reports_a_changed_field(self):
        frozen = build_manifest()
        frozen["memory"] = "on"  # simulate a manifest that no longer matches the tree
        with patch("agent_from_scratch.evals.bench.manifest.MANIFEST_PATH") as path:
            path.exists.return_value = True
            import json
            path.read_text.return_value = json.dumps(frozen)
            drift = manifest_drift()
        self.assertIn("memory", drift)


if __name__ == "__main__":
    unittest.main()
