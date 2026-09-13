import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent_from_scratch.session import SessionStore
from agent_from_scratch.trace import RunStopReason, TraceStore


class SessionTests(unittest.TestCase):
    def test_all_operations_reject_escaping_ids_and_symlinks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            sessions.mkdir()
            victim = root / "victim.jsonl"
            original = '{"run_id": "old", "messages": []}\n'
            victim.write_text(original)
            (sessions / "linked.jsonl").symlink_to(victim)
            store = SessionStore(sessions)
            for session_id in ("../victim", str(root / "victim"), "nested/name", "", "linked"):
                for operation in (store.load_records, store.reset, lambda name: store.append(name, "run", [])):
                    with self.subTest(session_id=session_id, operation=operation), self.assertRaises(ValueError):
                        operation(session_id)
                    self.assertEqual(victim.read_text(), original)

    def test_load_rejects_invalid_records_and_messages_with_line_number(self):
        invalid = [None, [], {"run_id": "old"}, {"run_id": "old", "messages": None}]
        invalid += [{"run_id": "old", "messages": [message]} for message in (
            None, {}, {"role": "user", "content": 42},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "", "tool_calls": [None]},
        )]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            store = SessionStore(directory)
            for record in invalid:
                with self.subTest(record=record):
                    path.write_text('{"run_id": "valid", "messages": []}\n' + json.dumps(record) + '\n')
                    with self.assertRaisesRegex(ValueError, "line 2"):
                        store.load_records("broken")
            path.write_text('{broken json}\n')
            with self.assertRaisesRegex(ValueError, "line 1"):
                store.load_records("broken")

    def test_failed_append_preserves_previous_session(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(directory)
            messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
            store.append("session_1", "first", messages)
            path = Path(directory) / "session_1.jsonl"
            original = path.read_bytes()
            with patch("agent_from_scratch.utils.os.fsync", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.append("session_1", "second", messages)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                store.append("session_1", "bad", [{"role": "user", "content": 42}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(SessionStore(directory).load_records("session_1")[0]["messages"], messages)
            store.reset("session_1")
            self.assertEqual(store.load_records("session_1"), [])

    def test_legacy_and_new_runs_replay_only_completed_messages(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(directory)
            old = {"run_id": "old", "messages": [{"role": "user", "content": "Remember blue"}]}
            path = Path(directory) / "mixed.jsonl"
            path.write_text(json.dumps(old) + "\n")
            store.append("mixed", "failed", [], stop_reason=RunStopReason.MODEL_ERROR)
            store.append("mixed", "interrupted", [], stop_reason=RunStopReason.INTERRUPTED)
            latest = [{"role": "assistant", "content": "Blue"}]
            store.append("mixed", "new", latest, started_at="2026-09-12T00:00:00+00:00")
            self.assertEqual(store.load_history("mixed"), old["messages"] + latest)
            self.assertEqual(len(store.load_records("mixed")), 4)
            self.assertEqual(json.loads(path.read_text().splitlines()[0]), old)
            self.assertNotIn("started_at", store.load_records("mixed")[0])
            self.assertIsNone(TraceStore(Path(directory) / "runs").load_run("old"))

    def test_invalid_version_or_unfinished_history_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for changes in ({"schema_version": 2}, {"stop_reason": "unknown"},
                            {"started_at": 123}, {"stop_reason": "interrupted"}):
                record = {"schema_version": 1, "run_id": "run", "started_at": None,
                          "stop_reason": "final_response", "messages": [{"role": "user", "content": "Hi"}],
                          **changes}
                with self.subTest(changes=changes):
                    path.write_text(json.dumps(record) + "\n")
                    with self.assertRaisesRegex(ValueError, "line 1"):
                        SessionStore(directory).load_history("bad")

    def test_trace_lookup_handles_legacy_missing_and_escaping_references(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            legacy = {"run_id": "old", "messages": []}
            (runs / "old.jsonl").write_text(json.dumps(legacy) + "\n")
            (root / "outside.jsonl").write_text(json.dumps({"run_id": "outside"}))
            (runs / "linked.jsonl").symlink_to(root / "outside.jsonl")
            traces = TraceStore(runs)
            self.assertEqual(traces.load_run("old"), legacy)
            self.assertIsNone(traces.load_run("missing"))
            for run_id in ("../outside", "linked"):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    traces.load_run(run_id)


if __name__ == "__main__":
    unittest.main()
