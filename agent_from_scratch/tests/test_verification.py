import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM, LLMResponse
from agent_from_scratch.session import SessionStore
from agent_from_scratch.tools.files import ReadFileTool
from agent_from_scratch.tools.register import ToolRegistry
from agent_from_scratch.verification import full_file_check, missing_ranges


def read(offset=0, path="a.txt"):
    return LLMResponse("assistant", "", "tool_call", "read_file", {"path": path, "offset": offset})


def answer(text="Read the whole file."):
    return LLMResponse("assistant", text, "direct")


class VerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.target = self.workspace / "a.txt"
        self.target.write_text("abcdefghij")
        self.registry = ToolRegistry([ReadFileTool(self.workspace, max_bytes=4)])
        self.check = full_file_check(self.workspace, "a.txt")

    def observations(self, *offsets, path="a.txt"):
        messages = []
        for index, offset in enumerate(offsets):
            response = read(offset, path)
            response.call_id = str(index)
            result = self.registry.invoke("read_file", response.tool_params, str(index))
            messages.extend([response.to_message(), {"role": "tool", "tool_call_id": str(index),
                                                      "content": json.dumps(asdict(result))}])
        return messages

    def agent(self, *responses):
        model = Mock(spec=LLM)
        model.settings.return_value = {}
        model.generate.side_effect = responses
        return Agent(model, str(self.workspace / "state"), registry=self.registry)

    def test_first_or_last_chunk_is_not_full_coverage(self):
        for offsets, missing in [((0,), [(4, 10)]), ((8,), [(0, 8)])]:
            result = self.check(self.observations(*offsets))
            self.assertEqual(result.status, "pending")
            self.assertEqual(result.evidence["missing_ranges"], missing)

    def test_overlapping_and_out_of_order_reads_cover_once(self):
        result = self.check(self.observations(8, 4, 0, 0, 2))
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["covered_bytes"], 10)
        self.assertEqual(missing_ranges(10, [(0, 6), (2, 8)]), [(8, 10)])

    def test_empty_file_requires_a_real_read(self):
        self.target.write_bytes(b"")
        check = full_file_check(self.workspace, self.target)
        self.assertEqual(check([]).status, "pending")
        self.assertEqual(check(self.observations(0)).status, "passed")

    def test_utf8_counts_bytes_not_characters(self):
        self.target.write_text("你🙂好")
        check = full_file_check(self.workspace, self.target)
        result = check(self.observations(0, 3, 7))
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.evidence["covered_bytes"], 10)

    def test_file_replacement_blocks_old_coverage(self):
        messages = self.observations(0, 4, 8)
        replacement = self.workspace / "replacement"
        replacement.write_text("ABCDEFGHIJ")
        replacement.replace(self.target)
        self.assertEqual(self.check(messages).status, "blocked")

    def test_other_file_and_failed_reads_do_not_count(self):
        (self.workspace / "b.txt").write_text("abcdefghij")
        result = self.check(self.observations(0, 4, 8, path="b.txt"))
        self.assertEqual(result.evidence["covered_bytes"], 0)
        self.assertEqual(self.check(self.observations(20)).evidence["covered_bytes"], 0)

    def test_forged_version_range_or_unpaired_observation_is_rejected(self):
        for field, value in [("version", "old"), ("next_offset", 7), ("eof", True), ("offset", 1)]:
            messages = self.observations(0)
            observed = json.loads(messages[-1]["content"])
            observed["output"][field] = value
            messages[-1]["content"] = json.dumps(observed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(messages)
        with self.assertRaises(ValueError):
            self.check(self.observations(0)[1:])

    def test_early_candidate_continues_within_same_turn(self):
        agent = self.agent(read(), answer("Would you like me to continue?"), read(4), read(8), answer())
        result = agent.run_turn("Read a.txt", completion_check=self.check)
        self.assertEqual(result.stop_reason, "final_response")
        self.assertEqual(result.final_answer, "Read the whole file.")
        self.assertEqual(result.completion_check["status"], "passed")
        self.assertEqual(len(result.model_requests), 5)
        self.assertTrue(any("offset=4" in m.get("content", "") for m in result.messages))

    def test_rejected_candidate_stays_in_raw_trace_but_not_the_next_prompt(self):
        candidate = answer("LONG REJECTED SOURCE EXCERPT")
        candidate.raw_response = {"choices": [{"message": candidate.to_message(), "finish_reason": "stop"}]}
        agent = self.agent(read(), candidate, read(4), read(8), answer())
        result = agent.run_turn("Read a.txt", completion_check=self.check)
        self.assertFalse(any(m.get("content") == candidate.content for m in result.messages))
        request = result.model_requests[1]
        event = json.loads((self.workspace / "state/runs" / request.response_file).read_text())
        self.assertEqual(event["raw_response"]["choices"][0]["message"]["content"], candidate.content)
        # Historical request prefixes still end before any removed tail candidate.
        self.assertEqual(result.messages[:request.input_message_count][-1]["role"], "tool")

    def test_history_does_not_satisfy_current_turn_and_budget_is_not_reset(self):
        agent = self.agent(answer())
        agent.max_iterations = 1
        result = agent.run_turn("Read a.txt", self.observations(0, 4, 8),
                                completion_check=self.check, session_id="test")
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIsNone(result.final_answer)
        self.assertEqual(result.completion_check["evidence"]["covered_bytes"], 0)
        store = SessionStore(self.workspace / "state/sessions")
        self.assertEqual(store.load_history("test"), [])

    def test_partial_read_without_full_contract_can_finish(self):
        agent = self.agent(read(), answer("Read the requested prefix."))
        result = agent.run_turn("Read only first 4 bytes")
        self.assertEqual(result.stop_reason, "final_response")
        self.assertIsNone(result.completion_check)

    def test_checker_failure_does_not_accept_candidate(self):
        agent = self.agent(answer())
        result = agent.run_turn("Read a.txt", completion_check=Mock(side_effect=ValueError("invalid evidence")),
                                session_id="test")
        self.assertEqual(result.stop_reason, "check_failed")
        self.assertIsNone(result.final_answer)
        self.assertIn("invalid evidence", result.error_message)
        self.assertEqual(SessionStore(self.workspace / "state/sessions").load_history("test"), [])

    def test_budget_end_refreshes_evidence_without_an_extra_model_call(self):
        agent = self.agent(read(), read(4), read(8))
        agent.max_iterations = 3
        result = agent.run_turn("Read a.txt", completion_check=self.check)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(result.completion_check["status"], "passed")
        self.assertIsNone(result.final_answer)

    def test_candidate_rejections_have_a_separate_limit(self):
        agent = self.agent(answer(), answer(), answer(), read())
        result = agent.run_turn("Read a.txt", completion_check=self.check)
        self.assertEqual((result.stop_reason, len(result.model_requests)), ("check_failed", 3))
        self.assertIsNone(result.final_answer)
        self.assertIn("retry limit", result.error_message)

    def test_read_command_accepts_space_in_path_and_verifies_before_printing(self):
        target = self.workspace / "a file.txt"
        target.write_text("1234")
        agent = self.agent(read(path="a file.txt"), answer("Read the file."))
        with patch("builtins.input", side_effect=["/read a file.txt", "exit"]), patch("builtins.print") as output:
            agent.run_repl(workspace=self.workspace)
        self.assertTrue(output.call_args.args[0].endswith("Read the file."))
        self.assertEqual(agent.llm.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
