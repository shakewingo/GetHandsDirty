from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import replace
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent, recovery_feedback
from agent_from_scratch.context import InstructionConfig, InstructionLoadError
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType, ResponseError, ResponseErrorCode
from agent_from_scratch.tools.base import ToolCall, ToolRegistry
from agent_from_scratch.session import SessionStore
from agent_from_scratch.trace import RunStopReason, TraceStore
from agent_from_scratch.tools.files import ReadFileTool, WriteFileTool
from agent_from_scratch.tools.shell import ShellTool

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


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
    response.tool_calls[0].call_id = call_id
    return response


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=LLM)
        self.model.measure_context.return_value = {  # Scripted fitting budget; no real tokenizer.
            "count_method": "exact", "prompt_tokens": 100, "window_tokens": 8000,
            "response_reserve": 512, "remaining_tokens": 7388,
        }
        self.model.settings.return_value = {}
        self.model.read_usage.side_effect = LLM.read_usage
        self.agent = Agent(self.model)
        self.seen = []
        self.prompt_dir = Path(self.enterContext(TemporaryDirectory()))
        (self.prompt_dir / "system.md").write_text("System", encoding="utf-8")
        prompt = patch("agent_from_scratch.context.PROMPTS_DIR", self.prompt_dir)
        prompt.start()
        self.addCleanup(prompt.stop)

    def script(self, *responses):
        responses = iter(responses)

        def generate(messages, tools, **kwargs):
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

    def budget(self, remaining):
        # Scripted counts exercise admission policy, not tokenizer accuracy.
        return {"count_method": "exact", "prompt_tokens": 8000 - 512 - remaining,
                "window_tokens": 8000, "response_reserve": 512, "remaining_tokens": remaining}

    def test_context_exact_boundary_allows_final_answer_without_another_measurement(self):
        self.model.measure_context.side_effect = [self.budget(self.agent.limits.context_margin_tokens)]
        self.script(answer("Done"))
        result = self.agent.run_turn("Finish")
        self.assertEqual(result.stop_reason, "final_response")
        self.assertEqual(result.final_answer, "Done")
        self.assertEqual(self.model.generate.call_count, 1)
        self.assertEqual(self.model.measure_context.call_count, 1)

    def test_context_one_token_over_blocks_generation_and_tools(self):
        self.model.measure_context.return_value = self.budget(self.agent.limits.context_margin_tokens - 1)
        self.script(call())
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("Calculate")
        self.model.generate.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(result.stop_reason, "context_limit")
        self.assertEqual(result.model_requests[0].status, "blocked")
        self.assertEqual(result.model_requests[0].error_code, "context_limit")
        self.assertEqual([m["role"] for m in result.messages], ["system", "user"])

    def test_context_unavailable_measurement_or_unbounded_reserve_blocks(self):
        for budget in (None, {**self.budget(1000), "count_method": "unavailable"},
                       {**self.budget(1000), "response_reserve": None, "remaining_tokens": None}):
            with self.subTest(budget=budget):
                self.model.measure_context.return_value = budget
                self.script(answer())
                result = self.agent.run_turn("Finish")
                self.model.generate.assert_not_called()
                self.assertEqual(result.stop_reason, "context_limit")
                self.assertEqual(result.model_requests[0].error_code, "context_unavailable")

    def test_context_overflow_after_tool_preserves_effects_trace_and_session_history(self):
        self.agent.limits = replace(self.agent.limits, max_compact_calls=0)
        with TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            workspace.mkdir()
            self.agent.state_dir = directory
            self.agent.registry = ToolRegistry([WriteFileTool(workspace)])
            store = SessionStore(Path(directory, "sessions"))
            history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Saved"}]
            store.append("budget", "old", history)
            self.model.measure_context.side_effect = [self.budget(1000), self.budget(-100)]
            self.script(LLMResponse("assistant", "Writing", ResponseType.tool_call, tool_calls=[
                ToolCall("write_file", {"path": "evidence.txt", "content": "written"})]))
            with patch.object(self.agent, "execute_tool", wraps=self.agent.execute_tool) as execute:
                result = self.agent.run_turn("Write the file", history, session_id="budget")
            self.assertEqual(result.stop_reason, "context_limit")
            self.assertEqual(self.model.generate.call_count, 1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual((workspace / "evidence.txt").read_text(), "written")
            observation = result.messages[-1]
            assert observation["role"] == "tool"
            self.assertTrue(json.loads(observation.get("content") or "")["ok"])
            self.assertEqual(self.model.measure_context.call_args.args[0][-1], observation)
            self.assertEqual(store.load_history("budget"), history)
            trace = TraceStore(Path(directory, "runs")).load_run(result.run_id)
            assert trace is not None
            self.assertEqual(trace["messages"], result.messages)
            blocked = trace["model_requests"][-1]
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["budget"], self.budget(-100))
            self.assertEqual(blocked["input_message_count"], len(result.messages))
            self.assertEqual(trace["settings"]["context_margin_tokens"], 256)
            for field in ("usage", "raw_response", "finish_reason"):
                self.assertIsNone(blocked[field])

    def test_context_overflow_after_parser_feedback_blocks_retry(self):
        self.model.measure_context.side_effect = [self.budget(1000), self.budget(255)]
        self.script(ResponseError(ResponseErrorCode.INVALID_RESPONSE))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("Calculate")
        self.assertEqual(result.stop_reason, "context_limit")
        self.assertEqual(self.model.generate.call_count, 1)
        execute.assert_not_called()
        self.assertEqual([q.status for q in result.model_requests], ["parse_error", "blocked"])
        self.assertIn("[Runtime feedback]", (result.messages[-1].get("content") or ""))
        self.assertEqual(self.model.measure_context.call_args.args[0][-1], result.messages[-1])

    def test_repl_explains_context_stop_without_claiming_rollback(self):
        self.model.measure_context.return_value = self.budget(-1)
        with patch("builtins.input", side_effect=["Hello", "exit"]), patch("builtins.print") as output:
            self.agent.run_repl()
        printed = "\n".join(str(c.args[0]) for c in output.call_args_list)
        self.assertIn("Request exceeds the context budget", printed)
        self.assertIn("Completed tool actions remain in effect", printed)
        self.model.generate.assert_not_called()

    def test_context_measurements_precede_each_generation_and_survive_backend_failure(self):
        measured_inputs, stats = [], []

        def measure(messages, schemas, **kwargs):
            measured_inputs.append(deepcopy(messages))
            self.assertEqual(schemas, self.agent.registry.schemas())
            count = 100 * len(measured_inputs)
            stats.append({"count_method": "exact", "prompt_tokens": count,
                          "window_tokens": 8000, "response_reserve": 512,
                          "remaining_tokens": 8000 - count - 512})
            return stats[-1]

        replies = iter([ResponseError(ResponseErrorCode.INVALID_RESPONSE), call(), RuntimeError("backend failed")])

        def generate(messages, schemas, **kwargs):
            self.assertEqual(messages, measured_inputs[-1])
            self.assertEqual(len(measured_inputs), self.model.generate.call_count)
            response = next(replies)
            if isinstance(response, BaseException):
                raise response
            return response

        self.model.measure_context.side_effect = measure
        self.model.generate.side_effect = generate
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            result = self.agent.run_turn("2+2", session_id="measurement")
            self.assertEqual(result.stop_reason, "model_error")
            self.assertIn("[Runtime feedback]", measured_inputs[1][-1]["content"])
            self.assertEqual(measured_inputs[2][-1]["role"], "tool")
            trace = TraceStore(Path(directory) / "runs").load_run(result.run_id)
            assert trace is not None
            self.assertEqual([r["budget"] for r in trace["model_requests"]], stats)
            self.assertTrue(all(r["usage"] is None for r in trace["model_requests"]))

    def test_instructions_reload_between_turns_but_not_after_tool_execution(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            rules = workspace / "AGENTS.md"
            rules.write_text("Original rules", encoding="utf-8")
            self.agent.state_dir = str(root / "state")
            self.agent.instruction_config = InstructionConfig(workspace=workspace)
            self.script(call(), answer("4"), answer("Next"))
            execute = self.agent.execute_tool

            def change_rules(*args, **kwargs):
                rules.write_text("Changed rules", encoding="utf-8")
                return execute(*args, **kwargs)

            with patch.object(self.agent, "execute_tool", side_effect=change_rules):
                first = self.agent.run_turn("2+2", session_id="rules")
            self.assertEqual(self.seen[0][0], self.seen[1][0])
            self.assertIn("Original rules", self.seen[1][0]["content"])
            store = SessionStore(root / "state/sessions")
            history = store.load_history("rules")
            self.assertTrue(all(m["role"] != "system" for m in history))
            second = self.agent.run_turn("Next", history, session_id="rules")
            self.assertIn("Changed rules", self.seen[2][0]["content"])
            self.assertEqual(sum(m["role"] == "system" for m in self.seen[2]), 1)
            first_source = first.settings["instructions"]["sources"][-1]
            second_source = second.settings["instructions"]["sources"][-1]
            from hashlib import sha256
            self.assertEqual(first_source["sha256"], sha256(b"Original rules").hexdigest())
            self.assertEqual(second_source["sha256"], sha256(b"Changed rules").hexdigest())
            trace = TraceStore(root / "state/runs").load_run(first.run_id)
            assert trace is not None
            self.assertEqual(trace["settings"]["instructions"], first.settings["instructions"])
            self.assertEqual(trace["messages"][0], self.seen[0][0])

    def test_instruction_failure_preserves_session_and_repl_recovers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "user.md"
            self.agent.state_dir = str(root / "state")
            self.agent.instruction_config = InstructionConfig(user_path=missing)
            store = SessionStore(root / "state/sessions")
            history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Saved"}]
            store.append("rules", "old-run", history)
            before = (root / "state/sessions/rules.jsonl").read_bytes()
            with patch.object(self.agent, "execute_tool") as execute:
                with self.assertRaises(InstructionLoadError):
                    self.agent.run_turn("Rejected", history, session_id="rules")
                self.model.generate.assert_not_called()
                execute.assert_not_called()
            self.assertEqual((root / "state/sessions/rules.jsonl").read_bytes(), before)
            self.assertFalse((root / "state/runs").exists())

            self.script(answer("Recovered"))
            def inputs():
                yield "Still rejected"
                missing.write_text("User defaults", encoding="utf-8")
                yield "Try again"
                yield "exit"

            with patch("builtins.input", side_effect=inputs()), patch("builtins.print") as output:
                self.agent.run_repl(session_id="rules")
            self.model.generate.assert_called_once()
            self.assertIn("Instructions unavailable", output.call_args_list[0].args[0])
            self.assertNotIn("Still rejected", json.dumps(store.load_history("rules")))
            self.assertEqual(store.load_history("rules")[-1].get("content"), "Recovered")

    def test_prepared_inputs_stay_independent_across_feedback_tools_and_replay(self):
        received = []
        responses = iter([ResponseError(ResponseErrorCode.INVALID_RESPONSE), call(), answer("4")])

        def generate(messages, tools, **kwargs):
            received.append(messages)  # Retain actual inputs, without copying them in the mock.
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        self.model.generate.side_effect = generate
        history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Earlier request"},
                   {"role": "assistant", "content": "Earlier answer"}]
        original_history = deepcopy(history)
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            store = SessionStore(Path(directory) / "sessions")
            store.append("context", "earlier-run", history)
            result = self.agent.run_turn("2+2", history, session_id="context")

            self.assertEqual(result.final_answer, "4")
            self.assertEqual([len(messages) for messages in received], [4, 5, 7])
            self.assertIn("[Runtime feedback]", received[1][-1]["content"])
            self.assertEqual([m["role"] for m in received[2][-2:]], ["assistant", "tool"])
            self.assertEqual(json.loads(received[2][-1]["content"])["output"], 4)
            self.assertEqual(len({id(messages) for messages in received}), 3)
            for request, messages in zip(result.model_requests, received):
                self.assertEqual(messages, result.messages[:request.input_message_count])
                self.assertIsNot(messages, result.messages)
            self.assertEqual(store.load_history("context"), result.messages[1:])
            trace = TraceStore(Path(directory) / "runs").load_run(result.run_id)
            assert trace is not None
            self.assertEqual(trace["messages"], result.messages)
            self.assertEqual(history, original_history)

    def test_backend_input_mutation_does_not_change_raw_history_or_next_request(self):
        received = []
        history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Remember this"},
                   {"role": "assistant", "content": "Remembered"}]

        def generate(messages, tools, **kwargs):
            received.append(deepcopy(messages))
            if len(received) == 1:
                messages[1]["content"] = "Changed by backend"
                messages[-1]["content"] = "Changed request"
                return call()
            return answer("4")

        self.model.generate.side_effect = generate
        result = self.agent.run_turn("2+2", history)
        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.messages[1:3], history)
        self.assertEqual(received[1][1:3], history)
        self.assertEqual(result.messages[3].get("content"), "2+2")
        self.assertEqual(received[1][3]["content"], "2+2")
        self.assertEqual(history[0].get("content"), "Remember this")

    def test_source_examples_never_reach_tool_execution(self):
        with TemporaryDirectory() as directory:
            self.agent.registry = ToolRegistry([WriteFileTool(directory)])
            block = '<tool_call>{"name":"write_file","arguments":{"path":"bad","content":"x"}}</tool_call>'
            for text in (f"```xml\n{block}\n```", f"Example: `{block}`", f"`{block}`", f"> {block}"):
                self.script(LLM.parse_response({"choices": [{"message": {
                    "role": "assistant", "content": text,
                }}]}))
                with patch.object(self.agent, "execute_tool", wraps=self.agent.execute_tool) as execute:
                    result = self.agent.run_turn("Explain this code")
                execute.assert_not_called()
                self.assertEqual(result.final_answer, text)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_narrated_calls_execute_and_continue_through_rename_and_read(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            workspace.mkdir()
            model = LLM.__new__(LLM)
            model.measure_context = Mock(return_value={  # Scripted fitting budget.
                "count_method": "exact", "prompt_tokens": 100, "window_tokens": 2048,
                "response_reserve": 512, "remaining_tokens": 1436,
            })
            model.llm = Mock()
            model.model_path, model.temperature, model.max_tokens = "fake-model", 0.0, 512
            model.n_ctx, model.n_gpu_layers = 2048, 0
            registry = ToolRegistry([WriteFileTool(workspace), ReadFileTool(workspace), ShellTool(workspace)])
            agent = Agent(model, directory, registry=registry)

            def raw(content):
                return {"choices": [{"message": {"role": "assistant", "content": content},
                                     "finish_reason": "stop"}]}

            def block(name, arguments):
                return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"

            text = "print('Hello, World!')"
            rename = f"The absolute path is `{workspace / 'test.py'}`.\nNow I will rename it.\n" + block(
                "shell", {"command": "mv test.py test.text"})
            final = str(workspace / "test.text")
            model.llm.create_chat_completion.side_effect = [
                raw(block("write_file", {"path": "test.py", "content": text})),
                raw(rename),
                raw("Verifying.\n" + block("read_file", {"path": "test.text"}) + "\nI will report next."),
                raw(final),
            ]
            result = agent.run_turn("Create test.py, rename it to test.text, and return its path.", session_id="rename")
            self.assertEqual((result.stop_reason, result.final_answer), ("final_response", final))
            self.assertEqual(model.llm.create_chat_completion.call_count, 4)
            self.assertFalse((workspace / "test.py").exists())
            self.assertEqual((workspace / "test.text").read_text(), text)
            observations = [m for m in result.messages if m["role"] == "tool"]
            self.assertEqual([json.loads(m.get("content") or "")["ok"] for m in observations], [True] * 3)
            self.assertIn(text, json.loads(observations[-1].get("content") or "")["output"]["content"])
            calls = [m for m in result.messages if m.get("tool_calls")]
            self.assertEqual([m.get("tool_calls", [])[0]["id"] for m in calls],
                             [m["tool_call_id"] for m in observations])
            self.assertIn("Now I will rename it.", (calls[1].get("content") or ""))
            self.assertNotIn("<tool_call>", (calls[1].get("content") or ""))
            trace = json.loads((Path(directory) / "runs" / f"{result.run_id}.jsonl").read_text())
            self.assertEqual(trace["model_requests"][1]["raw_response"], raw(rename))
            self.assertEqual(SessionStore(Path(directory) / "sessions").load_history("rename"), result.messages[1:])

    def test_injected_registry_drives_schemas_execution_and_saved_observations(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            workspace.mkdir()
            (workspace / "config.txt").write_text("old.txt")
            registry = ToolRegistry([ReadFileTool(workspace), WriteFileTool(workspace)])
            agent = Agent(self.model, directory, registry=registry)
            responses = iter([
                LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('read_file', {'path': 'config.txt'})]),
                LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('write_file', {'path': 'config.txt', 'content': 'new.txt'})]),
                answer("Updated config.txt"),
            ])

            def generate(messages, tools, **kwargs):
                self.assertEqual(set(tools), {"read_file", "write_file"})
                return next(responses)

            self.model.generate.side_effect = generate
            result = agent.run_turn("Change the output filename", session_id="files")
            self.assertEqual(result.stop_reason, RunStopReason.FINAL_RESPONSE)
            self.assertEqual((workspace / "config.txt").read_text(), "new.txt")
            observations = [json.loads(m.get("content") or "") for m in result.messages if m["role"] == "tool"]
            self.assertEqual(observations[0]["output"]["content"], "1| old.txt")
            self.assertTrue(all(item["ok"] for item in observations))
            for request, observation in ((result.messages[2], result.messages[3]),
                                         (result.messages[4], result.messages[5])):
                self.assertEqual(request.get("tool_calls", [])[0]["id"], observation.get("tool_call_id"))
            saved = TraceStore(Path(directory, "runs")).load_run(result.run_id)
            assert saved is not None
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
        from agent_from_scratch.evals.legacy_files import ReadFileTool

        with TemporaryDirectory() as directory:
            (Path(directory) / "notes.txt").write_text("abcdefghij")
            self.agent = Agent(self.model, registry=ToolRegistry([ReadFileTool(directory)]))

            def read(**arguments):
                return LLMResponse('assistant', '', ResponseType.tool_call, tool_calls=[ToolCall('read_file', {'path': 'notes.txt', **arguments})])

            self.script(read(limit=4), read(chunk_size=4), read(offset=4, chunk_size=4),
                        read(offset=8, chunk_size=4), answer("Read all notes."))
            result = self.agent.run_turn("Read notes.txt in chunks without asking questions")
            observations = [json.loads(m.get("content") or "") for m in result.messages if m["role"] == "tool"]
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
                self.assertEqual(json.loads(observation.get("content") or "")["call_id"], expected_id)
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
        self.agent.limits = replace(self.agent.limits, max_iterations=2)
        self.script(call(left="bad"), call(left="bad"), answer("Unused"))
        result = self.agent.run_turn("test", [])
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIsNone(result.final_answer)

    def test_identical_failures_stop_after_three_and_leave_complete_tool_pairs(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.script(call(left="bad"), call(left="bad"), call(left="bad"), answer("Unused"))
            result = self.agent.run_turn("test", session_id="test")
            self.assertEqual((result.stop_reason, len(self.seen)), ("no_progress", 3))
            self.assertEqual(result.messages[-1]["role"], "tool")
            self.assertEqual(SessionStore(Path(directory) / "sessions").load_history("test"), [])
            self.assertIsNone(result.final_answer)

    def test_success_resets_consecutive_failure_limit(self):
        self.script(call(left="bad"), call(left="bad"), call(),
                    call(left="bad"), call(left="bad"), call(), answer("4"))
        self.assertEqual(self.agent.run_turn("test").final_answer, "4")

    def test_identical_parse_errors_stop_at_three(self):
        error = ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)
        self.script(error, error, error, answer("Unused"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("test")
        self.assertEqual((result.stop_reason, len(self.seen)), ("no_progress", 3))
        execute.assert_not_called()

    def test_parse_error_is_feedback_then_corrected_without_executing_a_tool(self):
        self.script(ResponseError(ResponseErrorCode.INVALID_RESPONSE), answer("Paris"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("Capital of France?", [])
        execute.assert_not_called()
        self.assertEqual(len(self.seen), 2)
        self.assertIn("could not be processed", self.seen[1][-1]["content"])
        self.assertFalse(any(message["role"] == "tool" for message in result.messages))
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "Paris"))

    def test_truncated_answer_retries_without_forcing_tools(self):
        self.script(ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE), answer("Concise summary"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("Summarize the supplied text")
        execute.assert_not_called()
        self.assertIn("more concisely", self.seen[1][-1]["content"])
        self.assertNotIn("Use one complete tool-call", self.seen[1][-1]["content"])
        self.assertEqual(result.final_answer, "Concise summary")
        self.assertIn("next necessary", recovery_feedback(ResponseError(ResponseErrorCode.TOO_MANY_TOOL_CALLS)))

    def test_truncated_call_never_executes_and_keeps_finish_evidence(self):
        raw = {"choices": [{"message": {"role": "assistant", "content":
               '<tool_call>{"name":"write_file","arguments":{"path":"a","content":"x"}}'},
                            "finish_reason": "length"}]}
        try:
            LLM.parse_response(raw)
        except ResponseError as error:
            error.raw_response = raw
            self.script(error, answer("Recovered briefly"))
        with patch.object(self.agent, "execute_tool") as execute:
            result = self.agent.run_turn("test")
        execute.assert_not_called()
        self.assertEqual(result.model_requests[0].finish_reason, "length")
        self.assertEqual(result.final_answer, "Recovered briefly")

    def test_unknown_tool_can_recover_using_available_names(self):
        unknown = call()
        unknown.tool_calls[0].name = "missing"
        self.script(unknown, call(), answer("4"))
        result = self.agent.run_turn("2+2")
        failure = json.loads(self.seen[1][-1]["content"])
        self.assertIn("calculator", failure["error_message"])
        self.assertEqual(result.final_answer, "4")

    def test_repeated_parse_errors_stop_without_a_final_answer(self):
        self.agent.limits = replace(self.agent.limits, max_iterations=2)
        error = ResponseError(ResponseErrorCode.INVALID_RESPONSE)
        self.script(error, error, answer("Unused"))
        result = self.agent.run_turn("bad", [])
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIsNone(result.final_answer)

    def test_parse_exhaustion_does_not_end_repl_or_leak_failed_turn(self):
        self.agent.limits = replace(self.agent.limits, max_iterations=2)
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
        self.agent.limits = replace(self.agent.limits, max_iterations=1)
        self.script(call())
        with patch("builtins.input", side_effect=["test", "exit"]), patch("builtins.print") as output:
            self.agent.run_repl()
        self.assertEqual(output.call_args.args[0],
                         "Stopped: reached maximum iterations.")

    def test_success_and_parse_errors_share_one_run_trace_without_entering_history(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            model = LLM.__new__(LLM)
            model.measure_context = Mock(return_value={  # Scripted fitting budget.
                "count_method": "exact", "prompt_tokens": 100, "window_tokens": 2048,
                "response_reserve": 512, "remaining_tokens": 1436,
            })
            backend = Mock()
            model.llm = backend
            model.temperature = 0.0
            model.max_tokens = 32
            model.model_path = "fake-model"
            model.n_ctx = 2048
            model.n_gpu_layers = 0
            self.agent.llm = model
            raw = {"choices": [{"message": {"role": "assistant", "content":
                   '<tool_call>{"name":"calculator","arguments":{"left":2**2}}</tool_call>'}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 12}}
            runs = Path(directory) / "runs"

            def generate(**kwargs):
                attempt = backend.create_chat_completion.call_count
                if attempt < 3:
                    return raw
                return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}

            backend.create_chat_completion.side_effect = generate
            with patch.object(self.agent, "execute_tool") as execute:
                result = self.agent.run_turn("calculate", [], session_id="trace-test")
            execute.assert_not_called()
            self.assertEqual(result.final_answer, "Done")
            self.assertEqual(len(list(runs.glob("*.jsonl"))), 1)
            trace = json.loads((runs / f"{result.run_id}.jsonl").read_text())
            self.assertEqual(trace["schema_version"], 5)
            self.assertEqual(trace["final_answer"], "Done")
            self.assertEqual(trace["model_requests"][-1]["raw_response"]["choices"][0]
                             ["message"]["content"], "Done")
            for request in trace["model_requests"][:2]:
                self.assertEqual(request["raw_response"], raw)
                self.assertEqual(request["error_code"], "invalid_tool_call")
                self.assertIn("Invalid tool-call JSON", request["error_message"])
            history = SessionStore(Path(directory) / "sessions").load_history("trace-test")
            self.assertNotIn("raw_response", json.dumps(history))
            self.assertNotIn("2**2", json.dumps(history))
            self.assertEqual([r.status for r in result.model_requests],
                             ["parse_error", "parse_error", "completed"])
            self.assertEqual(result.model_requests[0].usage,
                             {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": None})
            self.assertIsNone(result.model_requests[-1].usage)
            self.assertFalse(any(m["role"] == "assistant" and "<tool_call>" in (m.get("content") or "")
                                 for m in result.messages))

    def test_unified_trace_keeps_parse_evidence_on_handled_interruption(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            raw = {"choices": []}
            self.model.generate.side_effect = [
                ResponseError(ResponseErrorCode.INVALID_RESPONSE, raw_response=raw),
                KeyboardInterrupt(),
            ]
            result = self.agent.run_turn("interrupt after malformed output", [])
            runs = Path(directory) / "runs"
            trace = json.loads((runs / f"{result.run_id}.jsonl").read_text())
            self.assertEqual(trace["model_requests"][0]["raw_response"], raw)
            self.assertEqual(len(list(runs.glob("*.jsonl"))), 1)
            self.assertEqual(trace["stop_reason"], "interrupted")
            self.assertEqual([r["status"] for r in trace["model_requests"]], ["parse_error", "interrupted"])
            self.assertIsNone(trace["model_requests"][-1]["usage"])

    def test_disabled_persistence_keeps_malformed_envelope_only_in_run_evidence(self):
        raw = ["invalid response envelope"]
        self.script(ResponseError(ResponseErrorCode.INVALID_RESPONSE, raw_response=raw), answer())
        with patch("agent_from_scratch.trace.write_jsonl") as write:
            result = self.agent.run_turn("recover")
        write.assert_not_called()
        self.assertEqual(result.model_requests[0].raw_response, raw)
        self.assertNotIn("invalid response envelope", json.dumps(result.messages))

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
                self.assertEqual(interrupted["messages"][-1]["role"], "tool")
                if during_tool:
                    observation = json.loads(interrupted["messages"][-1]["content"])
                    self.assertFalse(observation["ok"])
                    self.assertEqual(observation["error_code"], "interrupted")
                    self.assertIsNone(observation["output"])
                    self.assertEqual(interrupted["messages"][-2]["tool_calls"][0]["id"],
                                     interrupted["messages"][-1]["tool_call_id"])
                else:
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
            self.assertEqual([r["call_ids"] for r in requests], [[first.tool_calls[0].call_id], []])
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

    def test_run_trace_write_failure_does_not_undo_recovery(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.script(ResponseError(ResponseErrorCode.INVALID_RESPONSE), answer("Recovered"))
            with patch("agent_from_scratch.trace.write_jsonl", side_effect=OSError("disk full")), \
                    patch("agent_from_scratch.trace.logger.error") as error_log:
                result = self.agent.run_turn("recover", [])
            self.assertEqual(result.final_answer, "Recovered")
            self.assertTrue(any("run trace" in args
                                for args, _ in error_log.call_args_list))

    def test_session_index_links_all_outcomes_to_run_metadata(self):
        with TemporaryDirectory() as directory:
            self.agent.state_dir = directory
            self.agent.limits = replace(self.agent.limits, max_iterations=1)
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
                assert trace is not None
                self.assertEqual(trace["session_id"], "demo")
                self.assertEqual(trace["settings"]["model_path"], f"model-{index}")
                self.assertEqual(trace["started_at"], record["started_at"])
                self.assertEqual(trace["stop_reason"], record["stop_reason"])
                if record["stop_reason"] != RunStopReason.FINAL_RESPONSE:
                    self.assertEqual(record["messages"], [])
            self.assertEqual([m.get("content") for m in store.load_history("demo")],
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
            old: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Saved"}]
            store.append("demo", "old", old)
            original = (Path(directory) / "sessions" / "demo.jsonl").read_bytes()
            self.script(answer("New"))
            with patch("agent_from_scratch.session.write_jsonl", side_effect=OSError("disk full")), \
                    patch("agent_from_scratch.agent.logger.error") as errors:
                result = self.agent.run_turn("Hi", old, session_id="demo")
            errors.assert_called_once()
            self.assertEqual((Path(directory) / "sessions" / "demo.jsonl").read_bytes(), original)
            trace = TraceStore(Path(directory) / "runs").load_run(result.run_id)
            assert trace is not None
            self.assertEqual(trace["session_id"], "demo")


class CheckpointResetTests(unittest.TestCase):
    def test_reset_clears_the_session_checkpoint(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(Path(directory, "sessions"))
            history: list[ChatCompletionRequestMessage] = [{"role": "user", "content": "a"},
                                                           {"role": "assistant", "content": "b"}]
            store.append("s", "run1", history)
            store.append_checkpoint("s", "run1", covered=2, summary="Goal: x", history=history,
                                    config={"compact_prompt_sha256": "a" * 64,
                                            "summary_max_tokens": 512, "model": {}})
            agent = Agent(Mock(spec=LLM), directory)
            agent._session_command("/reset", "s", store)
            self.assertEqual(store.load_history("s"), [])
            self.assertIsNone(store.load_checkpoint("s", history))


if __name__ == "__main__":
    unittest.main()
