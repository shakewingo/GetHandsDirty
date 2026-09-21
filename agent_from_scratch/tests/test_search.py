"""Search tools find files by name pattern and by content, skipping build/cache trees."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.tools.register import default_registry
from agent_from_scratch.tools.search import GlobFilesTool, GrepTextTool


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        (self.root / "sub").mkdir()
        (self.root / "node_modules").mkdir()
        (self.root / "a.json").write_text('{"port": 80}\n')
        (self.root / "sub" / "b.json").write_text('{\n  "Port": 8080\n}\n')
        (self.root / "node_modules" / "c.json").write_text('{"port": 1}\n')
        (self.root / "blob.bin").write_bytes(b"port\0\0")

    def test_glob_lists_matching_files_outside_ignored_trees(self):
        result = GlobFilesTool(self.root).invoke({"pattern": "**/*.json"})
        self.assertEqual(result.output["matches"], ["a.json", "sub/b.json"])
        limited = GlobFilesTool(self.root).invoke({"pattern": "**/*.json", "max_matches": 1})
        self.assertEqual((limited.output["matches"], limited.output["truncated"]), (["a.json"], True))
        self.assertEqual(GlobFilesTool(self.root).invoke({"pattern": "*", "path": "a.json"}).error_code,
                         "execution_error")

    def test_grep_reports_path_and_line_and_skips_binary(self):
        tool = GrepTextTool(self.root)
        self.assertEqual(tool.invoke({"query": "port"}).output["matches"], ['a.json:1: {"port": 80}'])
        found = tool.invoke({"query": "port", "case_sensitive": False, "include": "*.json"})
        self.assertEqual(found.output["matches"], ['a.json:1: {"port": 80}', 'sub/b.json:2:   "Port": 8080'])
        limited = tool.invoke({"query": "port", "case_sensitive": False, "max_matches": 1})
        self.assertEqual((len(limited.output["matches"]), limited.output["truncated"]), (1, True))
        self.assertEqual(tool.invoke({"query": "("}).error_code, "invalid_arguments")

    def test_workspace_confinement_and_registration(self):
        tool = GrepTextTool(self.root / "sub", restrict_to_workspace=True)
        self.assertEqual(tool.invoke({"query": "port", "path": ".."}).error_code, "denied")
        self.assertTrue({"glob_files", "grep_text"} <= set(default_registry.schemas()))


if __name__ == "__main__":
    unittest.main()
