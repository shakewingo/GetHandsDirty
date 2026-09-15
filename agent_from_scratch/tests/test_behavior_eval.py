"""Exercise the behavioral evaluator with scripted decisions, not model capability claims."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.evals.run import HERE, load_tasks, run_case
from agent_from_scratch.evals.verify import summarize
from agent_from_scratch.llm import ToolCall, LLM, LLMResponse, ResponseType


def call(name, **arguments):
    return LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall(name, arguments)])


def answer(text):
    return LLMResponse("assistant", text, ResponseType.direct)


def write_json(path, value):
    return call("write_file", path=path, content=json.dumps(value))


def solution(task_id):
    """Explicit valid decisions run through the real tools, including subprocess checks."""
    read = lambda path, **kwargs: call("read_file", path=path, **kwargs)
    checked = {"output": "report.json", "retries": 2, "enabled": True}
    url = "https://docs.example.test/releases/current"
    release = {"code": "AX-42", "release": "2.7", "source": url}
    scripts = {
        "profile_output": [read("manifest.json"), read("profiles/production.json"), answer("release-report.json")],
        "second_chunk_code": [read("archive.txt"), read("archive.txt", offset=1024), answer("SEA-47")],
        "header_only": [read("header.txt", chunk_size=128), answer("PINE-83")],
        "full_document": [read("complete.txt", offset=offset) for offset in (0, 1024, 2048)] + [answer("FINISH-LIME")],
        "flat_config_update": [read("config.json"),
            write_json("config.json", {"output": "report.json", "retries": 3, "format": "json"}), answer("DONE")],
        "nested_config_update": [read("manifest.json"), read("settings/active.json"),
            write_json("settings/active.json", {"name": "active", "output": {"filename": "analysis.json", "format": "json"},
                "retry": {"count": 4, "delay_seconds": 2}, "enabled": True}), answer("DONE")],
        "already_correct": [read("config.json"), answer("UNCHANGED")],
        "check_fix_clean": [call("shell", command_id="check_fixture"), answer("CHECKED")],
        "check_fix_fault": [call("shell", command_id="check_fixture"), read("config.json"),
            write_json("config.json", checked), call("shell", command_id="check_fixture"), answer("CHECKED")],
        "web_release_clean": [call("web_fetch", url=url), write_json("release.json", release), answer("DONE")],
        "web_release_fault": [call("web_fetch", url=url), call("web_fetch", url=url),
            write_json("release.json", release), answer("DONE")],
        "truncated_page": [call("web_fetch", url="https://docs.example.test/guides/overview"), answer("TRUNCATED")],
        "blocked_path": [read("../outside.txt"), answer("BLOCKED")],
        "missing_profile": [read("manifest.json"), answer("NEED_PROFILE")],
        "timeout_stop": [call("shell", command_id="slow_check"), answer("TIMEOUT")],
        "calculator_regression": [call("calculator", operation="add", left=17, right=5),
            call("calculator", operation="multiply", left=22, right=3), answer("66")],
        "direct_ready": [answer("READY")],
    }
    return scripts[task_id]


class BehavioralEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tasks = {task["id"]: task for task in load_tasks()}
        self.sequence = 0

    def run_script(self, task_id, responses):
        model = Mock(spec=LLM)
        model.settings.return_value = {"backend": "scripted-test"}
        model.generate.side_effect = responses
        self.sequence += 1
        output = self.root / f"run-{self.sequence}"
        record = run_case(model, self.tasks[task_id], output)
        return record, model, output

    def test_all_dev_tasks_accept_valid_tool_trajectories(self):
        self.assertEqual(len(self.tasks), 17)
        for task_id in self.tasks:
            with self.subTest(task=task_id):
                responses = solution(task_id)
                record, model, _ = self.run_script(task_id, responses)
                self.assertTrue(record["passed"], record["checks"])
                self.assertEqual(model.generate.call_count, len(responses))

    def test_done_without_artifact_fails_without_reopening_the_loop(self):
        record, model, _ = self.run_script("flat_config_update", [answer("DONE")])
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["artifact_correct"])
        self.assertEqual(model.generate.call_count, 1)
        # A failed task is not automatically a reviewed false-completion label.
        self.assertIsNone(record["false_completion"])

    def test_correct_artifact_does_not_hide_unintended_changes(self):
        script = solution("flat_config_update")
        script.insert(-1, call("write_file", path="sentinel.txt", content="changed"))
        record, _, _ = self.run_script("flat_config_update", script)
        self.assertTrue(record["checks"]["artifact_correct"])
        self.assertFalse(record["passed"])
        self.assertEqual(record["unexpected_changes"], ["sentinel.txt"])

    def test_restoring_an_unintended_write_does_not_erase_it(self):
        task = self.tasks["flat_config_update"]
        original = (HERE / "fixtures" / task["fixture"] / "workspace/sentinel.txt").read_text()
        script = solution(task["id"])
        script[-1:-1] = [call("write_file", path="sentinel.txt", content="changed"),
                         call("write_file", path="sentinel.txt", content=original)]
        record, _, _ = self.run_script(task["id"], script)
        self.assertNotIn("sentinel.txt", record["changed_paths"])
        self.assertIn("sentinel.txt", record["unexpected_changes"])
        self.assertFalse(record["passed"])

    def test_byte_identical_write_still_violates_no_op_task(self):
        task = self.tasks["already_correct"]
        original = (HERE / "fixtures" / task["fixture"] / "workspace/config.json").read_text()
        script = solution("already_correct")
        script.insert(-1, call("write_file", path="config.json", content=original))
        record, _, _ = self.run_script("already_correct", script)
        self.assertEqual(record["changed_paths"], [])
        self.assertTrue(record["checks"]["answer_correct"])
        self.assertFalse(record["checks"]["no_write_attempts"])
        self.assertFalse(record["passed"])

    def test_correct_tail_marker_cannot_replace_full_read_evidence(self):
        for offsets in ((0,), (2048,), (0, 2048)):
            with self.subTest(offsets=offsets):
                script = [call("read_file", path="complete.txt", offset=offset) for offset in offsets]
                record, _, _ = self.run_script("full_document", script + [answer("FINISH-LIME")])
                self.assertTrue(record["checks"]["answer_correct"])
                self.assertFalse(record["checks"]["read_coverage"])
                self.assertFalse(record["passed"])

    def test_correct_header_answer_does_not_excuse_reading_the_body(self):
        record, _, _ = self.run_script("header_only", [call("read_file", path="header.txt"), answer("PINE-83")])
        self.assertTrue(record["checks"]["answer_correct"])
        self.assertFalse(record["checks"]["read_coverage"])
        self.assertFalse(record["passed"])

    def test_unrelated_read_cannot_validate_a_lucky_profile_answer(self):
        record, _, _ = self.run_script("profile_output", [
            call("read_file", path="sentinel.txt"), answer("release-report.json")])
        self.assertTrue(record["checks"]["answer_correct"])
        self.assertTrue(record["checks"]["required_tools"])
        self.assertFalse(record["passed"])

    def test_json_boolean_cannot_be_replaced_with_numeric_one(self):
        for task_id in ("nested_config_update", "check_fix_fault"):
            with self.subTest(task=task_id):
                script = solution(task_id)
                write = next(response for response in script if response.tool_calls and response.tool_calls[0].name == "write_file")
                value = json.loads(write.tool_calls[0].arguments["content"])
                value["enabled"] = 1
                write.tool_calls[0].arguments["content"] = json.dumps(value)
                record, _, _ = self.run_script(task_id, script)
                self.assertFalse(record["checks"]["artifact_correct"])
                self.assertFalse(record["passed"])
                if task_id == "check_fix_fault":
                    self.assertFalse(record["checks"]["passing_check_after_changes"])

    def test_fault_recovery_requires_successful_fetch_not_just_a_correct_file(self):
        clean, _, _ = self.run_script("web_release_clean", solution("web_release_clean"))
        # This script knows the answer, but its sole fetch receives the injected 503.
        failed, _, _ = self.run_script("web_release_fault", solution("web_release_clean"))
        recovered, _, _ = self.run_script("web_release_fault", solution("web_release_fault"))
        self.assertTrue(failed["checks"]["artifact_correct"])
        self.assertFalse(failed["checks"]["source_fetched"])
        self.assertTrue(failed["fault_encountered"])
        self.assertFalse(failed["passed"])
        self.assertTrue(recovered["passed"])
        self.assertTrue(recovered["fault_encountered"])
        self.assertFalse(summarize([clean, failed])["matched_recovery"][0]["recovered"])
        pair = summarize([clean, recovered])["matched_recovery"][0]
        self.assertTrue(pair["complete_pair"] and pair["clean_passed"] and pair["recovered"])

    def test_each_run_resets_fixture_session_and_recorded_fault_cursor(self):
        task = self.tasks["web_release_fault"]
        responses = iter(solution(task["id"]) + solution(task["id"]))
        initial_inputs = []

        def generate(messages, schemas):
            if len(messages) == 2:
                initial_inputs.append(deepcopy(messages))
            return next(responses)

        model = Mock(spec=LLM)
        model.settings.return_value = {}
        model.generate.side_effect = generate
        outputs = [self.root / name for name in ("first", "second")]
        records = [run_case(model, task, output) for output in outputs]
        self.assertTrue(all(record["passed"] and record["fault_encountered"] for record in records))
        self.assertTrue(all(len(record["tool_errors"]) == 1 for record in records))
        self.assertNotEqual(records[0]["run_id"], records[1]["run_id"])
        self.assertEqual(len(initial_inputs), 2)
        self.assertEqual(initial_inputs[0], initial_inputs[1])
        self.assertEqual([message["role"] for message in initial_inputs[0]], ["system", "user"])
        before = [json.loads((output / "before.json").read_text()) for output in outputs]
        self.assertEqual(before[0], before[1])
        self.assertNotIn("release.json", before[1])
        self.assertTrue(all((output / "workspace/release.json").is_file() for output in outputs))
        self.assertFalse((HERE / "fixtures" / task["fixture"] / "workspace/release.json").exists())

    def test_split_overlap_and_changed_pair_prompt_are_rejected(self):
        directory = self.root / "suite"
        shutil.copytree(HERE / "fixtures", directory / "fixtures")
        shutil.copyfile(HERE / "tasks.jsonl", directory / "tasks.jsonl")
        splits = json.loads((HERE / "splits.json").read_text())
        invalid = deepcopy(splits)
        invalid["test"].append(invalid["dev"][0])
        (directory / "splits.json").write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "exactly one split"):
            load_tasks(directory)
        (directory / "splits.json").write_text(json.dumps(splits))
        rows = deepcopy(list(self.tasks.values()))
        next(row for row in rows if row["id"] == "web_release_fault")["prompt"] += " Extra coaching."
        (directory / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Unmatched recovery pair: prompt"):
            load_tasks(directory)

    def test_missing_usage_remains_unknown_even_with_some_counted_requests(self):
        script = solution("flat_config_update")
        script[0].usage = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
        partial, _, _ = self.run_script("flat_config_update", script)
        missing, _, _ = self.run_script("direct_ready", solution("direct_ready"))
        self.assertTrue(partial["passed"] and missing["passed"])
        self.assertEqual(partial["requests_without_usage"], 2)
        self.assertEqual(missing["requests_without_usage"], 1)
        for record in (partial, missing, summarize([partial, missing])):
            self.assertTrue(all(value is None for value in record["usage"].values()))

    def test_bad_arguments_are_observed_and_can_be_corrected_in_the_normal_loop(self):
        script = [call("read_file", unexpected="missing path")] + solution("flat_config_update")
        record, model, _ = self.run_script("flat_config_update", script)
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(record["invalid_calls"], 1)
        self.assertEqual(record["tool_errors"][0]["error_code"], "invalid_arguments")
        self.assertEqual(model.generate.call_count, len(script))

    def test_blocked_command_feedback_allows_an_available_check(self):
        script = [call("shell", command_id="unconfigured")] + solution("check_fix_clean")
        record, model, _ = self.run_script("check_fix_clean", script)
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(record["tool_errors"][0]["error_code"], "denied")
        self.assertEqual(model.generate.call_count, len(script))

    def test_reading_before_and_after_a_passing_check_is_allowed(self):
        for task_id in ("check_fix_clean", "check_fix_fault"):
            script = [call("read_file", path="config.json")] + solution(task_id)
            script.insert(-1, call("read_file", path="config.json"))
            record, _, _ = self.run_script(task_id, script)
            self.assertTrue(record["passed"], record["checks"])


if __name__ == "__main__":
    unittest.main()
