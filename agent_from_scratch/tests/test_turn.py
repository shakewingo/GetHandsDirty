import json
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import Mock, patch

from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM, LLMResponse, ResponseType


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
        self.agent = Agent(self.model)
        self.seen = []
        prompt = patch("agent_from_scratch.agent.render_prompt", return_value="System")
        prompt.start()
        self.addCleanup(prompt.stop)

    def script(self, *responses):
        responses = iter(responses)

        def generate(messages, tools):
            self.seen.append(deepcopy(messages))
            return next(responses)

        self.model.generate.side_effect = generate

    def test_direct_answer(self):
        self.script(answer("Paris"))
        result = self.agent.run_turn("Capital of France?")
        self.assertEqual((result.stop_reason, result.final_answer), ("final_response", "Paris"))
        self.assertEqual(len(self.seen), 1)

    def test_dependent_calls_and_matching_ids(self):
        self.script(call(), call(4, 4, "multiply", "native_id"), answer("16"))
        result = self.agent.run_turn("(2+2)*4")
        self.assertEqual(result.final_answer, "16")
        self.assertEqual(len(self.seen), 3)
        self.assertEqual(self.seen[1][-1]["content"], "Tool output: 4")
        self.assertEqual(self.seen[2][-1]["content"], "Tool output: 16")
        for index, expected_id in [(2, "call_1"), (4, "native_id")]:
            request, observation = result.messages[index:index + 2]
            self.assertEqual(request["role"], "assistant")
            self.assertEqual(observation["role"], "tool")
            if request["role"] == "assistant" and observation["role"] == "tool":
                assert "tool_calls" in request
                self.assertEqual(request["tool_calls"][0]["id"], expected_id)
                self.assertEqual(observation["tool_call_id"], expected_id)
                self.assertNotIn("<tool_call>", request.get("content") or "")

    def test_bad_arguments_are_observed_once_then_corrected(self):
        self.script(call(left="bad"), call(), answer("4"))
        with patch.object(self.agent, "execute_tool", wraps=self.agent.execute_tool) as execute:
            result = self.agent.run_turn("2+2")
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(len(self.seen[1]), 4)
        self.assertIn("Tool calling failed:", self.seen[1][-1]["content"])
        self.assertEqual(result.final_answer, "4")

    def test_repeated_failures_stop_at_iteration_limit(self):
        self.agent.max_iterations = 2
        self.script(call(left="bad"), call(left="bad"), answer("Unused"))
        result = self.agent.run_turn("test")
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertIsNone(result.final_answer)

    def test_parse_error_ends_turn_and_repl_accepts_next_request(self):
        def generate(messages, tools):
            if messages[-1]["content"] == "bad":
                return LLM.parse_response({"choices": []})
            return answer("Paris")

        self.model.generate.side_effect = generate
        with patch("builtins.input", side_effect=["bad", "good", "exit"]), patch("builtins.print") as output:
            self.agent.run_repl()
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertIn("Could not process the model response:", output.call_args_list[0].args[0])
        self.assertEqual(output.call_args_list[1].args[0], "Paris")

    def test_repl_does_not_print_tool_output_as_final_answer(self):
        self.agent.max_iterations = 1
        self.script(call())
        with patch("builtins.input", side_effect=["test", "exit"]), patch("builtins.print") as output:
            self.agent.run_repl()
        self.assertEqual(output.call_args.args[0],
                         "Stopped: maximum iterations reached without a final answer.")


if __name__ == "__main__":
    unittest.main()
