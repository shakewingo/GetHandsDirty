"""The Stage 8 verifier: content-first scoring, automatic false-completion and fault checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.bench.spec import Answer, Expect, Task
from agent_from_scratch.evals.bench.verify import bench_summary, check
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.tools.base import ToolResult
from agent_from_scratch.trace import RunStopReason, TurnResult


def _row(name, output, *, ok=True, error_message=None, arguments=None, call_id="c1"):
    """One assistant tool-call message plus its tool-result message, in production shape."""
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments or {})}}]}
    result = ToolResult(call_id=call_id, tool_name=name, ok=ok, output=output,
                        error_code=None if ok else "execution_error",
                        error_message=error_message)
    return [assistant, result.to_message()]


def _turn(rows, final_answer, stop_reason=RunStopReason.FINAL_RESPONSE):
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]
    for row in rows:
        messages.extend(row)
    return TurnResult(messages=messages, final_answer=final_answer, stop_reason=stop_reason)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        (self.workspace / "config.json").write_text('{"output": "old.json"}\n')

    def test_a_correct_advisory_answer_wrapped_in_prose_still_passes(self):
        task = Task(id="t", skeleton="s", family="inspection", split="dev", prompt="p",
                    tools=("read_file",), expect=Expect(answer=Answer(value="FINISH-LIME")))
        before = snapshot(self.workspace)
        result = _turn([_row("read_file", {"content": "..."})], "The value is FINISH-LIME.")
        score = check(task, result, self.workspace, before)
        self.assertTrue(score["passed"], score["checks"])
        self.assertFalse(score["checks"]["format_exact"])

    def test_a_required_answer_needs_the_exact_format(self):
        task = Task(id="t", skeleton="s", family="stopping", split="dev", prompt="p",
                    tools=("read_file",),
                    expect=Expect(answer=Answer(value="UNCHANGED", format="required")))
        before = snapshot(self.workspace)
        wrapped = _turn([], "The answer is UNCHANGED.")
        exact = _turn([], "UNCHANGED")
        self.assertFalse(check(task, wrapped, self.workspace, before)["passed"])
        self.assertTrue(check(task, exact, self.workspace, before)["passed"])

    def test_answer_match_is_whole_token_not_substring(self):
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(answer=Answer(value="DONE", reject=("UNDONE",))))
        before = snapshot(self.workspace)
        self.assertFalse(check(task, _turn([], "UNDONE"), self.workspace, before)["passed"])
        self.assertTrue(check(task, _turn([], "DONE"), self.workspace, before)["passed"])

    def test_files_check_compares_json_by_value_and_text_after_stripping_newline(self):
        (self.workspace / "config.json").write_text('{"a": 1, "b": 2}\n')
        (self.workspace / "note.txt").write_text("hello\n")
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(files={"config.json": '{"b": 2, "a": 1}',
                                        "note.txt": "hello"}))
        before = {"config.json": "h", "note.txt": "h2"}
        self.assertTrue(check(task, _turn([], ""), self.workspace, before)["checks"]["files"])

    def test_unchanged_fails_on_an_unexpected_write(self):
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(files={"config.json": '{"output": "old.json"}'}))
        before = {"config.json": "OLDHASH", "other.txt": "SAMEHASH"}
        after_snapshot_note = "other.txt changed relative to `before`"  # documents the scenario
        # `check` re-snapshots the live workspace, so simulate drift by writing a new file.
        (self.workspace / "other.txt").write_text("changed")
        result = _turn([], "")
        score = check(task, result, self.workspace, before)
        self.assertFalse(score["checks"]["unchanged"], after_snapshot_note)
        self.assertIn("other.txt", score["changed_paths"])

    def test_evidence_requires_the_value_in_a_successful_observation(self):
        task = Task(id="t", skeleton="s", family="inspection", split="dev", prompt="p",
                    tools=("read_file",), expect=Expect(answer=Answer(value="X"),
                                                        evidence=("secret-42",)))
        before = {}
        missing = _turn([_row("read_file", {"content": "no clue here"})], "X")
        present = _turn([_row("read_file", {"content": "secret-42 is the value"})], "X")
        self.assertFalse(check(task, missing, self.workspace, before)["checks"]["evidence"])
        self.assertTrue(check(task, present, self.workspace, before)["checks"]["evidence"])

    def test_process_rules_no_write_attempts_and_check_before_write(self):
        task = Task(id="t", skeleton="s", family="stopping", split="dev", prompt="p", tools=(),
                    expect=Expect(process=("no_write_attempts",)))
        before = {}
        clean = _turn([_row("read_file", {})], "")
        wrote = _turn([_row("read_file", {}), _row("write_file", {}, call_id="c2")], "")
        self.assertTrue(check(task, clean, self.workspace, before)["checks"]["process"])
        self.assertFalse(check(task, wrote, self.workspace, before)["checks"]["process"])

        recovery = Task(id="t2", skeleton="s", family="recovery", split="dev", prompt="p",
                        tools=(), expect=Expect(process=("check_before_write", "passing_check_after_last_write")))
        bad_order = _turn([_row("write_file", {}, call_id="c1"),
                           _row("shell", {"exit_code": 0}, call_id="c2")], "")
        good_order = _turn([_row("shell", {"exit_code": 1}, ok=False,
                                 error_message="Command exited with code 1", call_id="c1"),
                            _row("write_file", {}, call_id="c2"),
                            _row("shell", {"exit_code": 0}, call_id="c3")], "")
        self.assertFalse(check(recovery, bad_order, self.workspace, before)["checks"]["process"])
        self.assertTrue(check(recovery, good_order, self.workspace, before)["checks"]["process"])

    def test_false_completion_flags_a_claimed_but_unmet_task(self):
        task = Task(id="t", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                    expect=Expect(files={"config.json": '{"output": "new.json"}'}),
                    claim_tokens=("DONE",))
        before = {"config.json": "h"}
        lied = _turn([], "DONE")  # config.json was never changed to new.json
        self.assertTrue(check(task, lied, self.workspace, before)["false_completion"])
        no_claim = Task(id="t2", skeleton="s", family="updates", split="dev", prompt="p", tools=(),
                        expect=Expect())
        self.assertIsNone(check(no_claim, _turn([], "anything"), self.workspace, before)["false_completion"])

    def test_fault_encountered_searches_the_whole_failed_row(self):
        task = Task(id="t", skeleton="s", family="recovery", split="dev", prompt="p",
                    tools=("read_file",), expect=Expect(answer=Answer(value="X")),
                    fault_signal=("read_file", "Simulated transient read failure."))
        before = {}
        hit = _turn([_row("read_file", {}, ok=False,
                          error_message="Simulated transient read failure."),
                    _row("read_file", {"content": "X"}, call_id="c2")], "X")
        miss = _turn([_row("read_file", {"content": "X"})], "X")
        self.assertTrue(check(task, hit, self.workspace, before)["fault_encountered"])
        self.assertFalse(check(task, miss, self.workspace, before)["fault_encountered"])

    def test_bench_summary_reports_by_skeleton_matched_recovery_and_tokens(self):
        usage_a = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        usage_b1 = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
        usage_b2 = {"prompt_tokens": 80, "completion_tokens": 8, "total_tokens": 88}
        records = [
            {"skeleton": "a", "family": "inspection", "split": "dev", "passed": True,
             "checks": {"format_exact": True}, "false_completion": None, "fault_encountered": None,
             "pair_id": None, "condition": None, "invalid_calls": 0, "model_requests": 2,
             "elapsed_seconds": 1.0, "usage": usage_a},
            {"skeleton": "b", "family": "recovery", "split": "dev", "passed": True,
             "checks": {}, "false_completion": False, "fault_encountered": False,
             "pair_id": "b-0", "condition": "clean", "invalid_calls": 0, "model_requests": 2,
             "elapsed_seconds": 1.0, "usage": usage_b1},
            {"skeleton": "b", "family": "recovery", "split": "dev", "passed": True,
             "checks": {}, "false_completion": False, "fault_encountered": True,
             "pair_id": "b-0", "condition": "fault", "invalid_calls": 1, "model_requests": 3,
             "elapsed_seconds": 2.0, "usage": usage_b2},
        ]
        summary = bench_summary(records)
        self.assertEqual(summary["by_skeleton"]["a"], {"passed": 1, "total": 1, "format_exact": 1})
        self.assertEqual(summary["matched_recovery"],
                         [{"pair_id": "b-0", "complete_pair": True, "clean_passed": True,
                           "fault_passed": True, "fault_encountered": True, "recovered": True}])
        self.assertEqual(summary["false_completion"], {"count": 0, "claimed": 2, "total": 3})
        self.assertEqual(summary["usage"], {"prompt_tokens": 230, "completion_tokens": 23,
                                            "total_tokens": 253})

    def test_bench_summary_usage_is_none_when_any_record_is_missing_it(self):
        records = [{"skeleton": "a", "family": "inspection", "split": "dev", "passed": True,
                   "checks": {}, "false_completion": None, "fault_encountered": None,
                   "pair_id": None, "condition": None, "invalid_calls": 0, "model_requests": 1,
                   "elapsed_seconds": 1.0, "usage": {"prompt_tokens": None,
                                                     "completion_tokens": None, "total_tokens": None}}]
        self.assertIsNone(bench_summary(records)["usage"]["total_tokens"])


if __name__ == "__main__":
    unittest.main()
