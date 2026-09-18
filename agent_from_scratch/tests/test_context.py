from __future__ import annotations
from typing import TYPE_CHECKING, Any

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.context import ContextBuilder, InstructionConfig, InstructionLoadError, load_instructions

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


class ContextTests(unittest.TestCase):
    def test_complete_components_are_independent_down_to_nested_calls(self):
        instructions: list[ChatCompletionRequestMessage] = [{"role": "system", "content": "System"}]
        history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Earlier request"},
                   {"role": "assistant", "content": "Earlier answer"}]
        current_turn: list[ChatCompletionRequestMessage] = [
            {"role": "user", "content": "Calculate"},
            {"role": "assistant", "content": "Checking", "tool_calls": [
                {"id": "one", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"left": 2, "right": 2}'}},
                {"id": "two", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"left": 3, "right": 3}'}},
            ]},
            {"role": "tool", "tool_call_id": "one", "content": '{"ok": false}'},
            {"role": "tool", "tool_call_id": "two", "content": '{"error_code": "skipped"}'},
            {"role": "user", "content": "[Runtime feedback] Retry."},
        ]
        expected = deepcopy(instructions + history + current_turn)
        prepared = ContextBuilder().build_messages(
            instructions=instructions, history=history, current_turn=current_turn,
        )
        self.assertEqual(prepared, expected)
        prepared[0]["content"] = "Changed instructions"
        prepared[1]["content"] = "Changed history"
        prepared[4].get("tool_calls", [])[0]["function"]["arguments"] = "{}"
        prepared[5]["content"] = "Changed observation"
        prepared.pop()
        self.assertEqual(instructions + history + current_turn, expected)


class InstructionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.prompts = self.root / "prompts"
        self.prompts.mkdir()
        (self.prompts / "system.md").write_text(" Core rules\n", encoding="utf-8")
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.user = self.root / "user.md"
        self.rules = self.workspace / "AGENTS.md"
        self.enterContext(patch("agent_from_scratch.context.PROMPTS_DIR", self.prompts))

    def test_system_only_and_absent_workspace_are_distinguishable(self):
        text, meta = load_instructions(InstructionConfig())
        self.assertEqual(text, "Core rules")
        self.assertEqual([s["status"] for s in meta["sources"]], ["loaded", "disabled", "disabled"])
        text, meta = load_instructions(InstructionConfig(workspace=self.workspace))
        self.assertEqual(text, "Core rules")
        self.assertEqual(meta["sources"][-1]["status"], "missing")

    def test_order_literal_text_and_hashes_without_ancestor_or_include_loading(self):
        self.user.write_text("Prefer short answers. {{ untouched }}", encoding="utf-8")
        self.rules.write_text("Read @extra.md\n{% include 'secret.md' %}", encoding="utf-8")
        (self.root / "AGENTS.md").write_text("PARENT RULE", encoding="utf-8")
        (self.workspace / "nested").mkdir()
        (self.workspace / "nested/AGENTS.md").write_text("NESTED RULE", encoding="utf-8")
        (self.workspace / "extra.md").write_text("INCLUDED RULE", encoding="utf-8")
        text, meta = load_instructions(InstructionConfig(workspace=self.workspace, user_path=self.user))
        self.assertEqual(text, "Core rules\n\n# User defaults\n\nPrefer short answers. {{ untouched }}"
                         "\n\n# Workspace rules\n\nRead @extra.md\n{% include 'secret.md' %}")
        included = ["Core rules", self.user.read_text(), self.rules.read_text()]
        for source, content in zip(meta["sources"], included):
            self.assertEqual(source["sha256"], sha256(content.encode()).hexdigest())
            self.assertEqual(source["byte_count"], len(content.encode()))
            self.assertTrue(Path(source["path"]).is_absolute())
        self.assertNotIn("PARENT RULE", text)
        self.assertNotIn("NESTED RULE", text)
        self.assertNotIn("INCLUDED RULE", text)

    def test_missing_required_empty_and_invalid_sources(self):
        config = InstructionConfig(user_path=self.user)
        with self.assertRaisesRegex(InstructionLoadError, "user"):
            load_instructions(config)
        for data in (b"\xff", b"x" * 8193):
            self.user.write_bytes(data)
            with self.subTest(data_length=len(data)), self.assertRaises(InstructionLoadError):
                load_instructions(config)
        self.user.write_bytes(b"")
        _, meta = load_instructions(config)
        self.assertEqual(meta["sources"][1]["byte_count"], 0)
        self.user.unlink()
        self.user.mkdir()
        with self.assertRaisesRegex(InstructionLoadError, "regular file"):
            load_instructions(config)
        (self.prompts / "system.md").write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(InstructionLoadError, "empty"):
            load_instructions(InstructionConfig())
        (self.prompts / "system.md").unlink()
        with self.assertRaisesRegex(InstructionLoadError, "system"):
            load_instructions(InstructionConfig())

    def test_workspace_links_and_unreadable_source_fail(self):
        config = InstructionConfig(workspace=self.workspace)
        self.rules.symlink_to(self.workspace / "missing")
        with self.assertRaises(InstructionLoadError):
            load_instructions(config)
        self.rules.unlink()
        self.user.write_text("External rules", encoding="utf-8")
        self.rules.symlink_to(self.user)
        with self.assertRaisesRegex(InstructionLoadError, "escapes"):
            load_instructions(config)
        self.rules.unlink()
        self.rules.write_text("Local rules", encoding="utf-8")
        real_open = Path.open
        def open_file(path, *args, **kwargs):
            if path == self.rules.resolve():
                raise PermissionError("denied")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", open_file), self.assertRaisesRegex(InstructionLoadError, "denied"):
            load_instructions(config)

    def test_utf8_source_and_assembled_caps_include_wrappers(self):
        (self.prompts / "system.md").write_text("Core", encoding="utf-8")
        self.user.write_text("中" * 4, encoding="utf-8")  # Exactly 12 bytes.
        config = InstructionConfig(user_path=self.user, max_source_bytes=12)
        text, _ = load_instructions(config)
        exact = len(text.encode())
        load_instructions(InstructionConfig(user_path=self.user, max_source_bytes=12, max_total_bytes=exact))
        with self.assertRaisesRegex(InstructionLoadError, "Assembled"):
            load_instructions(InstructionConfig(user_path=self.user, max_source_bytes=12, max_total_bytes=exact - 1))
        self.user.write_text("中" * 4 + "a", encoding="utf-8")
        with self.assertRaisesRegex(InstructionLoadError, "12 bytes"):
            load_instructions(config)

    def test_configuration_is_fixed_to_resolved_paths_and_valid_caps(self):
        invalid_configs: list[dict[str, Any]] = [
            {"max_source_bytes": 0}, {"max_total_bytes": -1}, {"max_source_bytes": True},
            {"workspace": self.root / "missing"},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(InstructionLoadError):
                InstructionConfig(**kwargs)
        config = InstructionConfig(workspace=self.workspace / ".." / "workspace", user_path=self.user)
        self.assertEqual(config.workspace, self.workspace.resolve())
        self.assertEqual(config.user_path, self.user.resolve())
        self.workspace.rmdir()
        with self.assertRaisesRegex(InstructionLoadError, "Workspace"):
            load_instructions(config)


if __name__ == "__main__":
    unittest.main()
