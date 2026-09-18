"""Frozen byte-based filesystem regressions; general tools are in test_general_files.py."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

from agent_from_scratch.tools.base import ToolErrorCode, ToolRegistry
from agent_from_scratch.evals.legacy_files import ListFilesTool, ReadFileTool, WriteFileTool
from agent_from_scratch.utils import file_version


class FileToolTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.registry = ToolRegistry([
            ListFilesTool(self.workspace, max_entries=2),
            ReadFileTool(self.workspace, max_bytes=8),
            WriteFileTool(self.workspace, max_bytes=8),
        ])

    def test_create_replace_and_read_utf8_file(self):
        for content in ("你好", "", "12345678"):
            with self.subTest(content=content):
                written = self.registry.invoke("write_file", {"path": "nested/a.txt", "content": content})
                self.assertTrue(written.ok, written.error_message)
                read = self.registry.invoke("read_file", {"path": "nested/a.txt"})
                self.assertTrue(read.ok, read.error_message)
                self.assertEqual(read.output["content"], content)
                self.assertTrue(read.output["eof"])
                self.assertIsNone(read.output["next_offset"])
                self.assertEqual((self.workspace / "nested/a.txt").read_bytes(), content.encode("utf-8"))
        self.assertEqual(self.registry.invoke("list_files", {"path": "nested"}).output,
                         {"entries": ["a.txt"], "truncated": False})

    def test_escapes_are_blocked_without_outside_changes(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        (self.workspace / "link").symlink_to(outside, target_is_directory=True)
        (self.workspace / "file-link").symlink_to(outside / "secret.txt")
        for path in ("../outside/secret.txt", str(outside / "secret.txt"),
                     "link/secret.txt", "file-link", "link/new/file.txt"):
            for name, arguments in (
                ("read_file", {"path": path}),
                ("write_file", {"path": path, "content": "changed"}),
                ("list_files", {"path": path}),
            ):
                with self.subTest(name=name, path=path):
                    result = self.registry.invoke(name, arguments, "blocked")
                    self.assertEqual(result.error_code, ToolErrorCode.EXECUTION_ERROR)
                    self.assertEqual(result.call_id, "blocked")
        self.assertEqual((outside / "secret.txt").read_text(), "secret")
        self.assertFalse((outside / "new").exists())

    def test_internal_symlinks_are_usable(self):
        (self.workspace / "a.txt").write_text("old")
        (self.workspace / "alias").symlink_to(self.workspace / "a.txt")
        self.assertTrue(self.registry.invoke("write_file", {"path": "alias", "content": "new"}).ok)
        self.assertEqual(self.registry.invoke("read_file", {"path": "alias"}).output["content"], "new")
        self.assertTrue((self.workspace / "alias").is_symlink())
        self.assertEqual((self.workspace / "a.txt").read_text(), "new")

    def test_size_limits_preserve_existing_files_and_bound_listing(self):
        (self.workspace / "a.txt").write_text("original")
        for path in ("a.txt", "new/sub/file.txt"):
            result = self.registry.invoke("write_file", {"path": path, "content": "你好世界"})
            self.assertFalse(result.ok)
        self.assertEqual((self.workspace / "a.txt").read_text(), "original")
        self.assertFalse((self.workspace / "new").exists())
        (self.workspace / "big.txt").write_text("123456789")
        read = self.registry.invoke("read_file", {"path": "big.txt"})
        self.assertTrue(read.ok)
        self.assertEqual(read.output["content"], "12345678")
        self.assertFalse(read.output["eof"])
        self.assertEqual(read.output["next_offset"], 8)
        (self.workspace / "c.txt").touch()
        listing = self.registry.invoke("list_files", {"path": "."})
        self.assertTrue(listing.output["truncated"])
        self.assertEqual(len(listing.output["entries"]), 2)

    def test_invalid_arguments_and_non_text_files_fail_then_recover(self):
        for arguments in ({"path": "a.txt"}, {"path": "a.txt", "content": 3},
                          {"path": "a.txt", "content": "x", "extra": True}):
            self.assertEqual(self.registry.invoke("write_file", arguments).error_code,
                             ToolErrorCode.INVALID_ARGUMENTS)
        self.assertFalse((self.workspace / "a.txt").exists())
        (self.workspace / "binary").write_bytes(b"\xff")
        os.mkfifo(self.workspace / "pipe")
        for path in ("", "missing", ".", "binary", "pipe"):
            self.assertFalse(self.registry.invoke("read_file", {"path": path}).ok)
        for path in ("", ".", "pipe"):
            self.assertFalse(self.registry.invoke("write_file", {"path": path, "content": "x"}).ok)
        self.assertTrue(self.registry.invoke("write_file", {"path": "ok", "content": "ok"}).ok)

    def test_failed_replacement_preserves_original_and_cleans_temporary_file(self):
        (self.workspace / "a.txt").write_text("original")
        with patch.object(Path, "replace", side_effect=OSError("write failure")):
            result = self.registry.invoke("write_file", {"path": "a.txt", "content": "new"})
        self.assertFalse(result.ok)
        self.assertEqual((self.workspace / "a.txt").read_text(), "original")
        self.assertEqual(list(self.workspace.iterdir()), [self.workspace / "a.txt"])

    def test_chunks_reconstruct_utf8_without_skipping_or_repeating_bytes(self):
        content = "abc你🙂好\r\nxyzé" * 5
        (self.workspace / "utf8.txt").write_bytes(content.encode("utf-8"))
        for size in (4, 5, 8):
            with self.subTest(chunk_size=size):
                chunks, offset = [], 0
                while True:
                    result = self.registry.invoke("read_file", {
                        "path": "utf8.txt", "offset": offset, "chunk_size": size,
                    })
                    self.assertTrue(result.ok, result.error_message)
                    chunk = result.output
                    chunks.append(chunk["content"])
                    consumed = len(chunk["content"].encode("utf-8"))
                    self.assertLessEqual(consumed, size)
                    self.assertIn(f"[{offset}, {offset + consumed})", chunk["read_status"])
                    if chunk["eof"]:
                        self.assertIsNone(chunk["next_offset"])
                        self.assertIn("End of file.", chunk["read_status"])
                        break
                    self.assertGreater(consumed, 0)
                    self.assertEqual(chunk["next_offset"], offset + consumed)
                    self.assertIn(f"offset={chunk['next_offset']}", chunk["read_status"])
                    offset = chunk["next_offset"]
                self.assertEqual("".join(chunks), content)

    def test_invalid_chunk_arguments_never_read_and_can_be_corrected(self):
        (self.workspace / "a.txt").write_text("12345678")
        for extra in ({"offset": -1}, {"offset": True}, {"offset": 1.5},
                      {"chunk_size": 0}, {"chunk_size": 3}, {"chunk_size": 9},
                      {"chunk_size": None}, {"chunk_size": "4"}, {"limit": 4}):
            with self.subTest(extra=extra), patch.object(Path, "open") as opened:
                result = self.registry.invoke("read_file", {"path": "a.txt", **extra})
                self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS)
                opened.assert_not_called()
        assert result.error_message is not None
        self.assertIn("Allowed fields", result.error_message)
        assert result.error_message is not None
        self.assertIn("chunk_size", result.error_message)
        result = self.registry.invoke("read_file", {"path": "a.txt", "chunk_size": 4})
        self.assertEqual(result.output["content"], "1234")
        self.assertFalse(self.registry.invoke("read_file", {"path": "a.txt", "offset": 9}).ok)
        end = self.registry.invoke("read_file", {"path": "a.txt", "offset": 8})
        self.assertEqual(end.output["content"], "")
        self.assertTrue(end.output["eof"])

    def test_chunking_does_not_hide_invalid_utf8(self):
        for raw, offset in ((b"abc\xffrest", 0), (b"abc\xe4\xbd", 0), ("你好".encode("utf-8"), 1)):
            (self.workspace / "bad.txt").write_bytes(raw)
            result = self.registry.invoke("read_file", {"path": "bad.txt", "offset": offset})
            self.assertFalse(result.ok)
        (self.workspace / "bad.txt").write_text("valid")
        self.assertTrue(self.registry.invoke("read_file", {"path": "bad.txt"}).ok)

    def test_read_metadata_matches_open_file_and_detects_mid_read_changes(self):
        target = self.workspace / "a.txt"
        target.write_text("abcdefghij")
        before = target.stat()
        result = self.registry.invoke("read_file", {"path": "a.txt"})
        self.assertEqual(result.output["size_bytes"], 10)
        self.assertEqual(result.output["version"], file_version(before))
        fields = {name: getattr(before, name) for name in (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns",
        )}
        changed = SimpleNamespace(**{**fields, "st_mtime_ns": before.st_mtime_ns + 1})
        with patch("agent_from_scratch.evals.legacy_files.fstat", side_effect=[before, changed]):
            result = self.registry.invoke("read_file", {"path": "a.txt"})
        self.assertFalse(result.ok)
        assert result.error_message is not None
        self.assertIn("changed", result.error_message)
        self.assertIsNone(result.output)

    def test_default_read_size_matches_schema_and_reduces_round_trips(self):
        content = "x" * 9562
        (self.workspace / "long.txt").write_text(content)
        tool = ReadFileTool(self.workspace)
        parameters = tool.to_schema()["function"]["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        schema = properties["chunk_size"]
        assert isinstance(schema, dict)
        first = tool.invoke({"path": "long.txt"}).output
        second = tool.invoke({"path": "long.txt", "offset": first["next_offset"]}).output
        self.assertEqual(schema["default"], len(first["content"].encode("utf-8")))
        self.assertEqual(first["next_offset"], 8192)
        self.assertTrue(second["eof"])
        self.assertEqual(first["content"] + second["content"], content)


if __name__ == "__main__":
    unittest.main()
