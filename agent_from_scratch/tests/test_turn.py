import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType, ResponseError, ResponseErrorCode
from agent_from_scratch.session import SessionStore
from agent_from_scratch.trace import RunStopReason, TraceStore
from agent_from_scratch.tools.files import ReadFileTool, WriteFileTool
from agent_from_scratch.tools.register import ToolRegistry


def answer(text="Done"):
    return LLMResponse(role="assistant", content=text, type=ResponseType.direct)


def call(left: Any = 2, right: Any = 2, operation="add", call_id=""):
    payload = {"name": "calculator", "arguments": {
        "operation": operation, "left": left, "right": right,
    }}
    response = LLM.parse_response({"choices": [{"message": {
        "role": "assistant",
        "content": "<tool_call>" + json.dumps(payload) + "</tool_call>",
    }}]})
    response.call_id = call_id
    return response


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=LLM)
        self.model.settings.return_value = {}
        self.model.read_usage.side_effect = LLM.read_usage
        self.agent = Agent(self.model)
        self.seen = []
        prompt = patch("agent_from_scratch.agent.render_prompt", return_value="System")
        prompt.start()
        self.addCleanup(prompt.stop)

    def script(self, *responses):
        responses = iter(responses)

        def generate(messages, tools):
            self.seen.append(deepcopy(messages))
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        self.model.generate.side_effect = generate

    def test_direct_answer(self):
        self.script(answer("Paris"))
        result = self.agent.run_turn("Capital of France?", [])
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "Paris"))
        self.assertEqual(len(self.seen), 1)

    def test_source_examples_never_reach_tool_execution(self):
        with TemporaryDirectory() as directory:
            self.agent.registry = ToolRegistry([WriteFileTool(directory)])
            block = '<tool_call>{"name":"write_file","arguments":{"path":"bad","content":"x"}}</tool_call>'
            for text in (f"```xml\n{block}\n```", f"Example: {block}", f"`{block}`"):
                self.script(LLM.parse_response({"choices": [{"message": {
                    "role": "assistant", "content": text,
                }}]}))
                with patch.object(self.agent, "execute_tool", wraps=self.agent.execute_tool) as execute:
                    result = self.agent.run_turn("Explain this code")
                execute.assert_not_called()
                self.assertEqual(result.final_answer, text)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_injected_registry_drives_schemas_execution_and_saved_observations(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            workspace.mkdir()
            (workspace / "config.txt").write_text("old.txt")
            registry = ToolRegistry([ReadFileTool(workspace), WriteFileTool(workspace)])
            agent = Agent(self.model, directory, registry=registry)
            responses = iter([
                LLMResponse(role="assistant", content="", type=ResponseType.tool_call,
                            tool_name="read_file", tool_params={"path": "config.txt"}),
                LLMResponse(role="assistant", content="", type=ResponseType.tool_call,
                            tool_name="write_file", tool_params={"path": "config.txt", "content": "new.txt"}),
                answer("Updated config.txt"),
            ])

            def generate(messages, tools):
                self.assertEqual(set(tools), {"read_file", "write_file"})
                return next(responses)

            self.model.generate.side_effect = generate
            result = agent.run_turn("Change the output filename", session_id="files")
            self.assertEqual(result.stop_reason, RunStopReason.FINAL_RESPONSE)
            self.assertEqual((workspace / "config.txt").read_text(), "new.txt")
            observations = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
            self.assertEqual(observations[0]["output"]["content"], "old.txt")
            self.assertTrue(all(item["ok"] for item in observations))
            for request, observation in ((result.messages[2], result.messages[3]),
                                         (result.messages[4], result.messages[5])):
                self.assertEqual(request["tool_calls"][0]["id"], observation["tool_call_id"])
            saved = TraceStore(Path(directory, "runs")).load_run(result.run_id)
            self.assertEqual(saved["messages"], result.messages)
            self.assertEqual(SessionStore(Path(directory, "sessions")).load_history("files"),
                             result.messages[1:])
            self.assertFalse(agent.execute_tool("calculator", {}).ok)
            self.assertFalse(self.agent.execute_tool("read_file", {"path": "config.txt"}).ok)

    def test_empty_registry_exposes_and_executes_no_tools(self):
        agent = Agent(self.model, registry=ToolRegistry([]))
        self.model.generate.return_value = answer()
        agent.run_turn("Hello")
        self.assertEqual(self.model.generate.call_args.args[1], {})
        self.assertFalse(agent.execute_tool("calculator", {"operation": "add", "left": 1, "right": 2}).ok)

    def test_read_recovers_from_bad_argument_then_continues_to_eof(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "notes.txt").write_text("abcdefghij")
            self.agent = Agent(self.model, registry=ToolRegistry([ReadFileTool(directory)]))

            def read(**arguments):
                return LLMResponse(role="assistant", content="", type=ResponseType.tool_call,
                                   tool_name="read_file", tool_params={"path": "notes.txt", **arguments})

            self.script(read(limit=4), read(chunk_size=4), read(offset=4, chunk_size=4),
                        read(offset=8, chunk_size=4), answer("Read all notes."))
            result = self.agent.run_turn("Read notes.txt in chunks without asking questions")
            observations = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
            self.assertEqual(observations[0]["error_code"], "invalid_arguments")
            self.assertIn("chunk_size", observations[0]["error_message"])
            chunks = [o["output"] for o in observations[1:]]
            self.assertEqual("".join(c["content"] for c in chunks), "abcdefghij")
            self.assertEqual([c["next_offset"] for c in chunks], [4, 8, None])
            self.assertTrue(chunks[-1]["eof"])
            self.assertEqual(result.stop_reason, RunStopReason.FINAL_RESPONSE)

    def test_dependent_calls_and_matching_ids(self):
        self.script(call(), call(4, 4, "multiply", "native_id"), answer("16"))
        result = self.agent.run_turn("(2+2)*4", [])
        self.assertEqual(result.final_answer, "16")
        self.assertEqual(len(self.seen), 3)
        self.assertEqual(json.loads(self.seen[1][-1]["content"])["output"], 4)
        self.assertEqual(json.loads(self.seen[2][-1]["content"])["output"], 16)
        for index, expected_id in [(2, f"{result.run_id}_call_1"), (4, "native_id")]:
            request, observation = result.messages[index:index + 2]
            self.assertEqual(request["role"], "assistant")
            self.assertEqual(observation["role"], "tool")
            if request["role"] == "assistant" and observation["role"] == "tool":
                assert "tool_calls" in request
                self.assertEqual(request["tool_calls"][0]["id"], expected_id)
                self.assertEqual(observation["tool_call_id"], expected_id)
                self.assertEqual(json.loads(observation["content"])["call_id"], expected_id)
                self.assertNotIn("<tool_call>", request.get("content") or "")

    def test_bad_arguments_are_observed_once_then_corrected(self):
        self.script(call(left="bad"), call(), answer("4"))
        with patch.object(self.agent, "execute_tool", wraps=self.agent.execute_tool) as execute:
            result = self.agent.run_turn("2+2", [])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(len(self.seen[1]), 4)
        failure = json.loads(self.seen[1][-1]["content"])
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["error_code"], "invalid_arguments")
        self.assertIn("arguments.left", failure["error_message"])
        self.assertEqual(failure["tool_name"], "calculator")
        self.assertEqual(result.final_answer, "4")

    def test_repeated_failures_stop_at_iteration_limit(self):
        self.agent.max_iterations = 2
        self.script(call(left="bad"), call(left="bad"), answer("Unused"))
        result = self.agent.run_turn("test", [])
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIsNone(result.final_answer)

    def test_parse_error_is_feedback_then_corrected_without_executing_a_tool(self):
        self.script(ResponseError(ResponseErrorCode.INVALID_RESPONSE), answer("Paris"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("Capital of France?", [])
        execute.assert_not_called()
        self.assertEqual(len(self.seen), 2)
        self.assertIn("could not be processed", self.seen[1][-1]["content"])
        self.assertFalse(any(message["role"] == "tool" for message in result.messages))
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "Paris"))

    def test_repeated_parse_errors_stop_without_a_final_answer(self):
        self.agent.max_iterations = 2
        error = ResponseError(ResponseErrorCode.INVALID_RESPONSE)
        self.script(error, error, answer("Unused"))
        result = self.agent.run_turn("bad", [])
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIsNone(result.final_answer)

    def test_parse_exhaustion_does_not_end_repl_or_leak_failed_turn(self):
        self.agent.max_iterations = 2
        error = ResponseError(ResponseErrorCode.INVALID_RESPONSE)
        self.script(error, error, answer("Paris"))
        with patch("builtins.input", side_effect=["bad", "good", "exit"]), patch("builtins.print") as output:
            self.agent.run_repl()
        self.assertEqual(len(self.seen), 3)
        self.assertIn("maximum iterations", output.call_args_list[0].args[0])
        self.assertTrue(output.call_args_list[1].args[0].endswith(" Paris"))
        self.assertIn("Agent:", output.call_args_list[1].args[0])
        self.assertEqual(self.seen[2], [{"role": "system", "content": "System"},
                                      {"role": "user", "content": "good"}])

    def test_repl_does_not_print_tool_output_as_final_answer(self):
        self.agent.max_iterations = 1
        self.script(call())
        with patch("builtins.input", side_effect=["test", "exit"]), patch("builtins.print") as output:
            self.agent.run_repl()
        self.assertEqual(output.call_args.args[0],
                         "Stopped: reached maximum iterations.")

    def test_parse_events_are_saved_before_retry_and_survive_final_run_save(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.model = LLM.__new__(LLM)
            self.model.llm = Mock()
            self.model.temperature = 0.0
            self.model.max_tokens = 32
            self.model.model_path = "fake-model"
            self.model.n_ctx = 2048
            self.model.n_gpu_layers = 0
            self.agent.llm = self.model
            raw = {"choices": [{"message": {"role": "assistant", "content":
                   '<tool_call>{"name":"calculator","arguments":{"left":2**2}}</tool_call>'}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 12}}
            runs = Path(directory) / "runs"

            def generate(**kwargs):
                attempt = self.model.llm.create_chat_completion.call_count
                # Events must already exist when the NEXT model request begins.
                events = sorted(runs.glob("*.parse-error-*.jsonl"))
                self.assertEqual(len(events), attempt - 1)
                for event_path in events:
                    event = json.loads(event_path.read_text())
                    self.assertEqual(event["raw_response"], raw)
                    self.assertEqual(event["event"], "parse_error")
                    self.assertEqual(event["error_code"], "invalid_tool_call")
                    self.assertIn("Invalid tool-call JSON", event["error"])
                if attempt < 3:
                    return raw
                return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}

            self.model.llm.create_chat_completion.side_effect = generate
            with patch.object(self.agent, "execute_tool") as execute:
                result = self.agent.run_turn("calculate", [])
            execute.assert_not_called()
            self.assertEqual(result.final_answer, "Done")
            self.assertEqual(len(list(runs.glob("*.jsonl"))), 4)
            response_path = runs / result.model_requests[-1].response_file
            self.assertEqual(json.loads(response_path.read_text())["raw_response"]["choices"][0]
                             ["message"]["content"], "Done")
            for iteration in (1, 2):
                event = json.loads((runs / f"{result.run_id}.parse-error-{iteration}.jsonl").read_text())
                self.assertEqual((event["run_id"], event["iteration"]), (result.run_id, iteration))
            self.assertEqual(json.loads((runs / f"{result.run_id}.jsonl").read_text())["final_answer"], "Done")
            self.assertEqual([r.status for r in result.model_requests],
                             ["parse_error", "parse_error", "completed"])
            self.assertEqual(result.model_requests[0].usage,
                             {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": None})
            self.assertIsNone(result.model_requests[-1].usage)
            self.assertFalse(any(m["role"] == "assistant" and "<tool_call>" in (m.get("content") or "")
                                 for m in result.messages))

    def test_parse_event_survives_interruption_before_turn_finishes(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            raw = {"choices": []}
            self.model.generate.side_effect = [
                ResponseError(ResponseErrorCode.INVALID_RESPONSE, raw_response=raw),
                KeyboardInterrupt(),
            ]
            result = self.agent.run_turn("interrupt after malformed output", [])
            runs = Path(directory) / "runs"
            event = runs / f"{result.run_id}.parse-error-1.jsonl"
            self.assertEqual(json.loads(event.read_text())["raw_response"], raw)
            trace = json.loads((runs / f"{result.run_id}.jsonl").read_text())
            self.assertEqual(trace["stop_reason"], "interrupted")
            self.assertEqual([r["status"] for r in trace["model_requests"]], ["parse_error", "interrupted"])
            self.assertIsNone(trace["model_requests"][-1]["usage"])

    def test_interrupted_tool_turn_is_traced_but_not_replayed(self):
        for during_tool in (False, True):
            with self.subTest(during_tool=during_tool), TemporaryDirectory() as directory:
                self.agent.state_dir = directory
                self.seen.clear()
                self.script(call(), answer("Recovered")) if during_tool else self.script(
                    call(), KeyboardInterrupt(), answer("Recovered"))
                with patch.object(self.agent, "execute_tool", wraps=self.agent.execute_tool) as execute:
                    if during_tool:
                        execute.side_effect = KeyboardInterrupt()
                    with patch("builtins.input", side_effect=["calculate", "good", "/quit"]), patch("builtins.print"):
                        self.agent.run_repl()
                execute.assert_called_once()
                traces = [json.loads(p.read_text()) for p in (Path(directory) / "runs").glob("*.jsonl")]
                interrupted = next(t for t in traces if t["stop_reason"] == "interrupted")
                self.assertEqual(interrupted["messages"][-1]["role"], "assistant" if during_tool else "tool")
                if not during_tool:
                    self.assertEqual(json.loads(interrupted["messages"][-1]["content"])["output"], 4)
                self.assertEqual([r["status"] for r in interrupted["model_requests"]],
                                 ["completed"] if during_tool else ["completed", "interrupted"])
                self.assertEqual(self.seen[-1], [{"role": "system", "content": "System"},
                                                {"role": "user", "content": "good"}])
                session = (Path(directory) / "sessions" / "default_session.jsonl").read_text().splitlines()
                self.assertEqual(len(session), 2)
                self.assertEqual(json.loads(session[0])["messages"], [])
                self.assertEqual(json.loads(session[0])["stop_reason"], "interrupted")
                self.assertEqual(json.loads(session[1])["messages"][0]["content"], "good")

    def test_backend_failure_is_saved_without_retry_and_repl_continues(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.script(RuntimeError("backend unavailable"), answer("Recovered"))
            with patch("builtins.input", side_effect=["bad", "good", "/quit"]), patch("builtins.print"):
                self.agent.run_repl()
            self.assertEqual(len(self.seen), 2)
            traces = [json.loads(p.read_text()) for p in (Path(directory) / "runs").glob("*.jsonl")]
            failure = next(t for t in traces if t["stop_reason"] == "model_error")
            self.assertEqual(failure["error_message"], "RuntimeError: backend unavailable")
            self.assertEqual(failure["model_requests"][0]["status"], "model_error")
            self.assertIsNone(failure["model_requests"][0]["usage"])
            self.assertEqual(len(self.seen[-1]), 2)

    def test_request_telemetry_is_saved_and_prefixes_match_actual_requests(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            first, second = call(), answer("4")
            first.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            second.usage = {"prompt_tokens": 40, "completion_tokens": 2, "total_tokens": 42}
            self.script(first, second)
            with patch("builtins.input", side_effect=["calculate", "/quit"]), patch("builtins.print"):
                self.agent.run_repl()
            trace = json.loads(next((Path(directory) / "runs").glob("*.jsonl")).read_text())
            requests = trace["model_requests"]
            self.assertEqual([r["usage"] for r in requests], [first.usage, second.usage])
            self.assertEqual([r["iteration"] for r in requests], [1, 2])
            self.assertEqual([r["call_id"] for r in requests], [first.call_id, None])
            for record, actual in zip(requests, self.seen):
                self.assertEqual(trace["messages"][:record["input_message_count"]], actual)
            session = json.loads((Path(directory) / "sessions" / "default_session.jsonl").read_text())
            self.assertEqual(set(session), {"schema_version", "run_id", "started_at", "stop_reason", "messages"})
            self.assertEqual(session["run_id"], trace["run_id"])
            self.assertEqual(trace["session_id"], "default_session")
            for message in session["messages"]:
                self.assertNotIn("usage", message)
                self.assertNotIn("model_requests", message)

    def test_restart_replays_tool_messages_with_distinct_fallback_ids(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.script(call(), answer("4"), call(), answer("4 again"))
            for agent in (self.agent, Agent(self.model, directory)):
                with patch("builtins.input", side_effect=["calculate", "/quit"]), patch("builtins.print"):
                    agent.run_repl()
            messages = self.seen[-1]
            calls = [m["tool_calls"][0]["id"] for m in messages if m.get("tool_calls")]
            observations = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(set(calls)), 2)
            self.assertEqual(calls, observations)
            self.assertEqual(sum(m["role"] == "system" for m in messages), 1)

    def test_corrupt_session_allows_recovery_commands_without_overwriting_it(self):
        for command in ("/reset", "/new", "/session recovered"):
            with self.subTest(command=command), TemporaryDirectory() as directory:
                self.agent.state_dir = directory
                self.seen.clear()
                path = Path(directory) / "sessions" / "default_session.jsonl"
                path.parent.mkdir()
                original = '{"run_id": "old"}\n'
                path.write_text(original)
                self.script(answer("Recovered"))
                with patch("builtins.input", side_effect=["blocked", command, "good", "/quit"]), patch("builtins.print"):
                    self.agent.run_repl()
                self.assertEqual(len(self.seen), 1)
                self.assertEqual(self.seen[0][-1]["content"], "good")
                if command == "/reset":
                    self.assertEqual(json.loads(path.read_text())["messages"][0]["content"], "good")
                else:
                    self.assertEqual(path.read_text(), original)

    def test_interrupt_or_eof_at_input_exits_without_starting_turn(self):
        for error in (KeyboardInterrupt(), EOFError()):
            with self.subTest(error=type(error).__name__):
                with patch("builtins.input", side_effect=error), patch("builtins.print"):
                    self.agent.run_repl()
                self.model.generate.assert_not_called()

    def test_parse_event_write_failure_does_not_stop_recovery(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.script(ResponseError(ResponseErrorCode.INVALID_RESPONSE), answer("Recovered"))
            with patch("agent_from_scratch.trace.write_jsonl", side_effect=OSError("disk full")), \
                    patch("agent_from_scratch.trace.logger.error") as error_log:
                result = self.agent.run_turn("recover", [])
            self.assertEqual(result.final_answer, "Recovered")
            self.assertTrue(any("parse-error event" in args
                                for args, _ in error_log.call_args_list))

    def test_session_index_links_all_outcomes_to_run_metadata(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.agent.max_iterations = 1
            store = SessionStore(Path(directory) / "sessions")
            traces = TraceStore(Path(directory) / "runs")
            self.script(answer("Blue"), RuntimeError("offline"), call(), KeyboardInterrupt(), answer("Still blue"))
            outcomes = []
            for index in range(5):
                self.model.settings.return_value = {"model_path": f"model-{index}", "temperature": 0.0}
                outcomes.append(self.agent.run_turn(f"request-{index}", store.load_history("demo"), session_id="demo"))
            records = store.load_records("demo")
            self.assertEqual([r["stop_reason"] for r in records],
                             ["final_response", "model_error", "max_iterations", "interrupted", "final_response"])
            for index, record in enumerate(records):
                trace = traces.load_run(record["run_id"])
                self.assertEqual(trace["session_id"], "demo")
                self.assertEqual(trace["settings"]["model_path"], f"model-{index}")
                self.assertEqual(trace["started_at"], record["started_at"])
                self.assertEqual(trace["stop_reason"], record["stop_reason"])
                if record["stop_reason"] != RunStopReason.FINAL_RESPONSE:
                    self.assertEqual(record["messages"], [])
            self.assertEqual([m["content"] for m in store.load_history("demo")],
                             ["request-0", "Blue", "request-4", "Still blue"])
            self.assertEqual([m["content"] for m in self.seen[-1]],
                             ["System", "request-0", "Blue", "request-4"])

    def test_trace_write_failure_still_saves_completed_session_history(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.script(answer("Remembered"))
            with patch("agent_from_scratch.trace.write_jsonl", side_effect=OSError("disk full")), \
                    patch("agent_from_scratch.trace.logger.error") as errors:
                result = self.agent.run_turn("Hi", session_id="demo")
            errors.assert_called_once()
            self.assertEqual(SessionStore(Path(directory) / "sessions").load_history("demo"), result.messages[1:])
            self.assertIsNone(TraceStore(Path(directory) / "runs").load_run(result.run_id))

    def test_session_write_failure_preserves_old_history_and_new_run_trace(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            store = SessionStore(Path(directory) / "sessions")
            old = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Saved"}]
            store.append("demo", "old", old)
            original = (Path(directory) / "sessions" / "demo.jsonl").read_bytes()
            self.script(answer("New"))
            with patch("agent_from_scratch.session.write_jsonl", side_effect=OSError("disk full")), \
                    patch("agent_from_scratch.agent.logger.error") as errors:
                result = self.agent.run_turn("Hi", old, session_id="demo")
            errors.assert_called_once()
            self.assertEqual((Path(directory) / "sessions" / "demo.jsonl").read_bytes(), original)
            self.assertEqual(TraceStore(Path(directory) / "runs").load_run(result.run_id)["session_id"], "demo")


if __name__ == "__main__":
    unittest.main()
