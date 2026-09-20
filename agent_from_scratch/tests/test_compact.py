"""Compaction changes model inputs, never raw evidence or executed effects."""

from __future__ import annotations
from typing import TYPE_CHECKING

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.config import AgentLimits
from agent_from_scratch.compact import CompactOutcome, Compactor
from agent_from_scratch.context import ContextState, InstructionConfig
from agent_from_scratch.evals.verify import metrics
from agent_from_scratch.llm import LLM, ResponseError, ResponseErrorCode
from agent_from_scratch.session import SessionStore
from agent_from_scratch.tests.test_turn import answer, call
from agent_from_scratch.tools.base import ToolCall, ToolRegistry
from agent_from_scratch.tools.files import WriteFileTool
from agent_from_scratch.trace import ModelRequest

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


def message(role, text) -> ChatCompletionRequestMessage:
    return {"role": role, "content": text}


def exchange(name) -> list[ChatCompletionRequestMessage]:
    return [call(call_id=name).to_message(),
            {"role": "tool", "tool_call_id": name, "content": '{"ok": true}'}]


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=LLM)
        self.model.settings.return_value = {}
        self.model.generate.return_value = answer("Goal: finish. Constraints: use 7. Progress: checked. Next: report.")
        self.pressure = False
        self.model.measure_context.side_effect = self.measure
        self.limits = AgentLimits()
        self.raw: list[ChatCompletionRequestMessage] = [message("system", "rules"), message("user", "old request"),
                    message("assistant", "old response" * 100), message("user", "Use 7, not 6.")]
        self.state = ContextState(self.raw, turn_start=3, last_sent=3)
        self.requests = []

    def measure(self, messages, schemas, **kwargs):
        # Deliberately scripted budget: these tests make no tokenizer claims.
        summarized = any(m.get("content", "").startswith("[Conversation summary:") for m in messages)
        summary_request = messages[0]["content"].startswith("Summarize the supplied")
        count = 100 + len(messages) * 10 if summarized or summary_request else 2000
        remaining = 100 if self.pressure and not (summarized or summary_request) else 4000
        return {"count_method": "exact", "prompt_tokens": count, "response_reserve": 512,
                "remaining_tokens": remaining, "window_tokens": count + 512 + remaining}

    def compact(self):
        outcome = Compactor(self.model, self.limits).attempt(self.state, {}, self.requests, 1)
        return outcome is CompactOutcome.APPLIED

    def test_unusable_summary_retries_once_at_the_same_cut_then_gives_up(self):
        self.model.generate.side_effect = [answer(""), answer("Goal: report 7. Next: answer.")]
        self.assertTrue(self.compact())
        self.assertEqual(len(self.requests), 2)
        self.assertEqual({q.covered_boundary for q in self.requests}, {self.state.covered})
        self.assertEqual(self.state.summary_calls, 2)
        retry = self.requests[1].input_messages
        assert retry is not None
        self.assertIn("previous attempt was rejected", retry[0]["content"])

    def test_attempt_never_exceeds_two_calls_or_the_run_budget(self):
        self.model.generate.side_effect = [answer(""), answer(""), answer("Goal: unused.")]
        self.assertFalse(self.compact())
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.model.generate.call_count, 2)
        self.limits = replace(AgentLimits(), max_compact_calls=1)
        self.state = ContextState(self.raw, turn_start=3, last_sent=3)
        self.requests = []
        self.model.generate.reset_mock()
        self.model.generate.side_effect = [answer(""), answer("Goal: unused.")]
        self.assertFalse(self.compact())
        self.assertEqual(len(self.requests), 1)  # the run ceiling stops the retry
        self.assertEqual(self.model.generate.call_count, 1)

    def test_blocked_summary_input_does_not_retry(self):
        self.model.measure_context.side_effect = lambda messages, schemas, **kwargs: {
            "count_method": "exact", "prompt_tokens": 9000, "response_reserve": 512,
            "remaining_tokens": -100, "window_tokens": 8192}
        self.assertFalse(self.compact())
        self.assertEqual(len(self.requests), 1)
        self.model.generate.assert_not_called()

    def test_compact_publishes_reloaded_rules_without_editing_raw(self):
        with TemporaryDirectory() as directory:
            rules = Path(directory, "AGENTS.md")
            rules.write_text("Original workspace rule.")
            agent = Agent(self.model, None, registry=ToolRegistry([]),
                          instruction_config=InstructionConfig(workspace=Path(directory)))
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            seen_first = False

            def rewrite(messages, schemas, **kwargs):
                nonlocal seen_first
                if not seen_first:
                    seen_first = True
                    rules.write_text("Revised workspace rule.")
                return self.measure(messages, schemas)

            self.pressure = True
            self.model.measure_context.side_effect = rewrite
            result = agent.run_turn("Use 7, not 6.",
                                    [message("user", "old"), message("assistant", "x" * 400)])
            summary, actor = result.model_requests
            assert actor.input_messages is not None
            self.assertIn("Revised workspace rule.", actor.input_messages[0]["content"])
            self.assertIn("Original workspace rule.", result.messages[0]["content"])
            assert summary.instructions is not None
            self.assertEqual(summary.instructions["sources"][-1]["status"], "loaded")

    def test_failed_reload_keeps_the_turn_snapshot_and_records_the_error(self):
        compactor = Compactor(self.model, self.limits,
                              reload_instructions=lambda: (None, {"status": "error",
                                                                  "detail": "unreadable"}))
        self.assertIs(compactor.attempt(self.state, {}, self.requests, 1),
                      CompactOutcome.APPLIED)
        self.assertIsNone(self.state.instructions)
        assert self.requests[0].instructions is not None
        self.assertEqual(self.requests[0].instructions["status"], "error")

    def test_summary_uses_its_own_output_reserve(self):
        self.limits = replace(AgentLimits(), summary_max_tokens=128)
        self.assertTrue(self.compact())
        summary_reserve = [call.kwargs.get("max_tokens")
                           for call in self.model.generate.call_args_list]
        self.assertEqual(summary_reserve, [128])
        measured = [call.kwargs.get("max_tokens")
                    for call in self.model.measure_context.call_args_list
                    if call.args[0][0]["content"].startswith("Summarize the supplied")]
        self.assertEqual(measured, [128])

    def test_manual_and_automatic_paths_share_summary_and_preserve_raw_session(self):
        for manual in (True, False):
            with self.subTest(manual=manual), TemporaryDirectory() as directory:
                self.pressure = not manual
                self.model.generate.reset_mock()
                self.model.generate.side_effect = [answer("Goal: report corrected value 7."), answer("7")]
                agent = Agent(self.model, directory, registry=ToolRegistry([]))
                original = deepcopy(self.raw[1:3])
                result = agent.run_turn("Use 7, not 6.", original, session_id="compact", compact=manual)
                self.assertEqual(result.stop_reason, "final_response")
                self.assertEqual([q.purpose for q in result.model_requests], ["compact", "agent"])
                summary, actor = result.model_requests
                assert actor.input_messages is not None
                self.assertEqual(summary.tools, {})
                self.assertEqual(actor.input_messages[-1], self.raw[-1])
                self.assertEqual(actor.input_messages[0], result.messages[0])
                self.assertEqual(result.messages[1:3], original)
                self.assertEqual(SessionStore(Path(directory, "sessions")).load_history("compact"),
                                 [self.raw[-1], message("assistant", "7")])
                saved = json.loads(next(Path(directory, "runs").glob("*.jsonl")).read_text())
                self.assertEqual(saved["model_requests"][1]["input_messages"], actor.input_messages)
                self.assertEqual(metrics(result)["model_requests"], 2)

    def test_keep_two_recent_batches_and_every_unsent_observation(self):
        self.raw.extend(exchange("old") + exchange("recent") + exchange("fresh"))
        self.state.covered, self.state.summary = 3, "prior summary"
        self.state.last_sent = len(self.raw) - 2
        original = deepcopy(self.raw)
        self.assertTrue(self.compact())
        self.assertEqual(self.state.covered, 6)
        self.assertEqual(self.state.messages()[-4:], self.raw[-4:])
        self.assertEqual(self.state.messages()[2], self.raw[3])
        assert self.requests[0].input_messages is not None
        source = json.loads(self.requests[0].input_messages[1]["content"])
        self.assertEqual(source["previous_summary"], "prior summary")
        self.assertEqual(source["messages"], self.raw[3:6])
        self.assertEqual(self.raw, original)
        self.state.messages()[-3]["content"] = "changed copy"
        self.assertEqual(self.raw, original)

    def test_unsent_parser_feedback_is_not_summarized(self):
        self.raw.append(message("user", "[Runtime feedback] Retry the truncated answer."))
        self.assertTrue(self.compact())
        self.assertEqual(self.state.last_sent, 3)
        self.assertEqual(self.state.messages()[-1], self.raw[-1])
        assert self.requests[0].input_messages is not None
        self.assertNotIn("Runtime feedback", self.requests[0].input_messages[1]["content"])

    def test_second_compaction_merges_summary_without_losing_current_request(self):
        self.assertTrue(self.compact())
        previous = self.state.summary
        self.raw.extend(exchange("old") + exchange("recent") + exchange("fresh"))
        self.state.last_sent = len(self.raw) - 2
        self.assertTrue(self.compact())
        self.assertEqual(self.state.summary_calls, 2)
        self.assertEqual(self.state.covered, 6)
        self.assertEqual(self.state.messages()[2:], [self.raw[3], *self.raw[6:]])
        assert self.requests[-1].input_messages is not None
        self.assertEqual(json.loads(self.requests[-1].input_messages[1]["content"])["previous_summary"],
                         previous)

    def test_compaction_does_not_reset_repeated_failure_counter(self):
        replies = iter([call(left="bad"), answer("summary"), call(left="bad"), call(left="bad")])
        def generate(messages, schemas, **kwargs):
            self.pressure = True
            return next(replies)
        self.model.generate.side_effect = generate
        result = Agent(self.model).run_turn("Calculate", self.raw[1:3])
        self.assertEqual(result.stop_reason, "no_progress")
        self.assertEqual(self.model.generate.call_count, 4)
        self.assertEqual([q.purpose for q in result.model_requests], ["agent", "compact", "agent", "agent"])

    def test_batch_cut_waits_for_all_results_including_skipped(self):
        batch = call(call_id="a").to_message()
        batch.get("tool_calls", []).append(call(call_id="b").to_message().get("tool_calls", [])[0])
        self.raw.extend([batch, {"role": "tool", "tool_call_id": "a", "content": "failed"},
                         {"role": "tool", "tool_call_id": "b", "content": "skipped"},
                         *exchange("recent"), *exchange("fresh")])
        self.state.covered = 3
        self.state.last_sent = 6  # Inside the first batch: no legal cut after its first result.
        self.assertEqual(self.state.compact_boundary(), 3)
        self.state.last_sent = len(self.raw)
        self.assertEqual(self.state.compact_boundary(), 7)

    def test_failures_preserve_view_and_cannot_retry_unchanged_boundary(self):
        for reply in (RuntimeError("offline"), ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE),
                      call(), answer("")):
            with self.subTest(reply=reply):
                self.state = ContextState(self.raw, 3, 3)
                self.requests = []
                self.model.generate.reset_mock()
                self.model.generate.side_effect = [reply, reply]
                before = self.state.messages()
                self.assertFalse(self.compact())
                self.assertEqual(self.state.messages(), before)
                self.assertFalse(self.compact())
                self.assertEqual(self.model.generate.call_count, 2)

    def test_oversized_summary_input_never_generates(self):
        self.model.measure_context.side_effect = None
        self.model.measure_context.return_value = {"count_method": "exact", "remaining_tokens": -1}
        self.assertFalse(self.compact())
        self.model.generate.assert_not_called()
        self.assertEqual(self.requests[0].status, "blocked")
        self.assertEqual(self.state.covered, 1)

    def test_candidate_must_fit_and_shrink_before_publishing(self):
        for after in ({"count_method": "exact", "prompt_tokens": 100, "remaining_tokens": -1},
                      {"count_method": "exact", "prompt_tokens": 3000, "remaining_tokens": 1000}):
            with self.subTest(after=after):
                self.state = ContextState(self.raw, 3, 3)
                self.requests = []
                before = self.measure(self.raw, {})
                self.model.measure_context.side_effect = [before, before, after]
                self.assertFalse(self.compact())
                self.assertEqual(self.state.messages(), self.raw)
                self.assertIn("smaller fitting", self.requests[0].error_message)

    def test_summary_and_actor_share_request_limit(self):
        self.model.generate.side_effect = [answer("summary"), call()]
        agent = Agent(self.model, limits=replace(self.limits, max_iterations=2))
        result = agent.run_turn("Calculate", self.raw[1:3], compact=True)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertEqual(len([m for m in result.messages if m["role"] == "tool"]), 1)
        self.state.summary_calls = 4
        self.assertFalse(self.compact())
        self.state.summary_calls = 0
        self.requests = [ModelRequest(1, 4)]
        self.limits = replace(self.limits, max_iterations=2)
        self.assertFalse(self.compact())

    def test_pressure_after_write_or_parser_error_keeps_fresh_feedback(self):
        for parse_error in (True, False):
            with self.subTest(parse_error=parse_error), TemporaryDirectory() as directory:
                self.pressure = False
                self.model.generate.reset_mock()
                seen = []
                if parse_error:
                    first = ResponseError(ResponseErrorCode.INVALID_RESPONSE)
                else:
                    first = call()
                    first.tool_calls = [ToolCall("write_file", {"path": "done.txt", "content": "once"})]
                replies = iter([first, answer("summary"), answer("Done")])
                def generate(messages, schemas, **kwargs):
                    seen.append(deepcopy(messages))
                    self.pressure = True
                    reply = next(replies)
                    if isinstance(reply, Exception):
                        raise reply
                    return reply
                self.model.generate.side_effect = generate
                agent = Agent(self.model, registry=ToolRegistry([WriteFileTool(Path(directory))]))
                with patch.object(agent, "execute_tool", wraps=agent.execute_tool) as execute:
                    result = agent.run_turn("Finish", self.raw[1:3])
                self.assertEqual(result.stop_reason, "final_response")
                self.assertEqual(execute.call_count, 0 if parse_error else 1)
                self.assertEqual([q.purpose for q in result.model_requests], ["agent", "compact", "agent"])
                self.assertEqual(seen[-1][-1], result.messages[-2])
                self.assertEqual(result.model_requests[1].last_sent_boundary, 4)
                # Compaction changes the view, never the turn's identity or its counters.
                self.assertEqual([q.iteration for q in result.model_requests], [1, 2, 2])
                self.assertIs(agent.registry, agent.registry)
                self.assertTrue(all(i.startswith(result.run_id)
                                    for q in result.model_requests for i in q.call_ids))
                if not parse_error:
                    self.assertEqual(Path(directory, "done.txt").read_text(), "once")

    def test_parser_feedback_survives_compaction_and_stays_unsent(self):
        self.raw.extend(exchange("old") + exchange("recent"))
        self.raw.append(message("user", "[Runtime feedback] The output was cut off."))
        self.state.last_sent = len(self.raw) - 1
        self.assertTrue(self.compact())
        self.assertIn(self.raw[-1], self.state.messages())
        self.assertGreaterEqual(self.state.last_sent, self.state.covered)

    def test_failing_summarizer_preserves_raw_evidence_and_saves_no_session(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            raw_before = deepcopy(self.raw[1:3])
            result = agent.run_turn("Use 7, not 6.", raw_before, session_id="failing")
            self.assertEqual(result.stop_reason, "context_limit")
            self.assertEqual(result.messages[1:3], raw_before)
            self.assertEqual(self.state.summary, "")
            self.assertEqual(SessionStore(Path(directory, "sessions")).load_history("failing"), [])

    def test_truncated_response_under_pressure_executes_nothing(self):
        self.pressure = True
        self.model.generate.side_effect = [
            answer("Goal: continue."),
            ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE),
            answer("Done."),
        ]
        registry = ToolRegistry([])
        result = Agent(self.model, None, registry=registry).run_turn(
            "Use 7, not 6.", deepcopy(self.raw[1:3]))
        feedback = [m for m in result.messages if "[Runtime feedback]" in str(m.get("content"))]
        self.assertEqual(len(feedback), 1)
        self.assertIn("cut off", str(feedback[0]["content"]))
        self.assertFalse(any(m.get("role") == "tool" for m in result.messages))

    def test_completed_compacted_turn_saves_a_checkpoint_after_its_raw_delta(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            # A checkpoint may only cover messages the session itself holds, so replay the
            # REPL's flow: earlier turns are on disk before this one runs.
            store = SessionStore(Path(directory, "sessions"))
            store.append("cp", "earlier", deepcopy(self.raw[1:3]))
            result = agent.run_turn("Use 7, not 6.", store.load_history("cp"), session_id="cp")
            lines = Path(directory, "sessions", "cp.jsonl").read_text().splitlines()
            earlier, turn, checkpoint = (json.loads(line) for line in lines)
            self.assertNotIn("kind", turn)
            self.assertEqual(checkpoint["kind"], "checkpoint")
            self.assertEqual(checkpoint["run_id"], result.run_id)
            # covered counts session messages: one less than ContextState.covered.
            self.assertEqual(checkpoint["covered"], 2)
            self.assertEqual(store.load_checkpoint("cp", store.load_history("cp")), checkpoint)
            self.assertEqual(len(checkpoint["config"]["compact_prompt_sha256"]), 64)

    def test_checkpoint_is_refused_when_history_was_never_saved(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            # Library callers may pass history the store does not hold; a checkpoint over it
            # could never be replayed, so none is written and the raw turn is still saved.
            result = agent.run_turn("Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            self.assertEqual(result.stop_reason, "final_response")
            store = SessionStore(Path(directory, "sessions"))
            self.assertEqual(len(store.load_history("cp")), 2)
            self.assertIsNone(store.load_checkpoint("cp", store.load_history("cp")))

    def test_incomplete_turn_saves_no_checkpoint(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
            Agent(self.model, directory, registry=ToolRegistry([])).run_turn(
                "Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            self.assertIsNone(SessionStore(Path(directory, "sessions")).load_checkpoint("cp", []))

    def test_checkpoint_write_failure_leaves_the_saved_turn_intact(self):
        with TemporaryDirectory() as directory:
            self.pressure = True
            self.model.generate.side_effect = [answer("Goal: report 7."), answer("7")]
            agent = Agent(self.model, directory, registry=ToolRegistry([]))
            with patch("agent_from_scratch.session.SessionStore.append_checkpoint",
                       side_effect=OSError("disk full")):
                result = agent.run_turn("Use 7, not 6.", deepcopy(self.raw[1:3]), session_id="cp")
            self.assertEqual(result.stop_reason, "final_response")
            store = SessionStore(Path(directory, "sessions"))
            self.assertEqual(len(store.load_history("cp")), 2)
            self.assertIsNone(store.load_checkpoint("cp", store.load_history("cp")))

    def test_failed_compaction_with_room_left_does_not_end_the_turn(self):
        """A manual compact must not kill a turn that the budget can still serve."""
        self.model.measure_context.side_effect = None
        self.model.measure_context.return_value = {
            "count_method": "exact", "prompt_tokens": 1000, "response_reserve": 512,
            "remaining_tokens": 4000, "window_tokens": 5512}
        for reply in (ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE), answer("   ")):
            with self.subTest(reply=reply):
                self.model.generate.side_effect = [reply, reply, answer("7")]
                result = Agent(self.model, registry=ToolRegistry([])).run_turn(
                    "Use 7, not 6.", self.raw[1:3], compact=True)
                self.assertEqual(result.stop_reason, "final_response")
                self.assertEqual(result.final_answer, "7")
                self.assertEqual([q.purpose for q in result.model_requests],
                                 ["compact", "compact", "agent"])

    def test_unavailable_measurement_keeps_its_own_error_code(self):
        """A missing tokenizer is not a context limit, whatever compaction did."""
        self.model.measure_context.side_effect = None
        self.model.measure_context.return_value = {"count_method": "unavailable",
                                                   "prompt_tokens": None, "remaining_tokens": None}
        result = Agent(self.model, registry=ToolRegistry([])).run_turn("Finish", self.raw[1:3])
        self.assertEqual(result.stop_reason, "context_limit")
        self.assertEqual(result.model_requests[-1].error_code, "context_unavailable")
        self.assertIn("exact prompt measurement", result.error_message or "")

    def test_repl_compact_command_applies_to_the_next_request_only(self):
        agent = Agent(self.model, registry=ToolRegistry([]))
        self.model.generate.side_effect = [answer("7"), answer("8")]
        seen = []
        original = agent.run_turn

        def run_turn(user_input, history=None, **kwargs):
            seen.append((user_input, kwargs.get("compact", False)))
            return original(user_input, history, **kwargs)

        with patch.object(agent, "run_turn", run_turn), patch("builtins.print"), \
                patch("builtins.input", side_effect=["/compact", "Use 7, not 6.", "again", "/quit"]):
            agent.run_repl()
        self.assertEqual(seen, [("Use 7, not 6.", True), ("again", False)])

    def test_failed_summary_stops_actor_and_counts_reported_usage(self):
        self.pressure = True
        error = ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE, raw_response={
            "usage": {"prompt_tokens": 100, "completion_tokens": 512, "total_tokens": 612}})
        self.model.generate.side_effect = error
        result = Agent(self.model).run_turn("Finish", self.raw[1:3])
        self.assertEqual(result.stop_reason, "context_limit")
        # Both calls of the one attempt are charged, and both report the same usage.
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertEqual(metrics(result)["usage"]["total_tokens"], 1224)
        self.assertEqual(result.messages[1:3], self.raw[1:3])


if __name__ == "__main__":
    unittest.main()
