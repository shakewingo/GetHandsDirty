"""The external scorers reproduce their benchmarks' own rules; checked with fakes, no model."""

import ast
import json
import unittest

from agent_from_scratch.evals.external import bfcl, when2call
from agent_from_scratch.tools.base import ToolCall

ROW = {"uuid": "u1", "question": "Weather in Oslo?", "correct_answer": "cannot_answer",
       "tools": ['{"name": "get.weather", "description": "d", "parameters": {"type": "dict", '
                 '"properties": {"city": {"type": "str", "description": "c"}, "days": {"type": "integer"}}, '
                 '"required": ["city"]}}'],
       "answers": {"direct": "It is sunny.", "tool_call": '{"name": "get.weather", "arguments": {"city": "Oslo"}}',
                   "request_for_info": "Which day?", "cannot_answer": "I cannot check the weather."}}


class MCQTests(unittest.TestCase):
    def test_the_prompt_and_choices_are_the_qwen_variant(self):
        prompt, choices, target = when2call.mcq_item(ROW)
        self.assertTrue(prompt.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud."))
        self.assertIn("<tools>\n" + ROW["tools"][0] + "\n</tools>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>user\nWeather in Oslo?<|im_end|>\n<|im_start|>assistant\n"))
        # The tool call is offered as bare JSON, choice 1, as process_docs_qwen2_5 does.
        self.assertEqual(choices, list(ROW["answers"].values()))
        self.assertEqual(target, 3)

    def test_trailing_whitespace_moves_into_the_scored_continuation(self):
        encode = lambda text: [ord(c) for c in text]  # one token per character
        seen = []

        def score(ids):
            seen.append("".join(map(chr, ids)))
            return [None] + [-1.0] * (len(ids) - 1)
        value = when2call.loglikelihood("ab\n", "xyz", encode, score)
        self.assertEqual(value, -4.0)  # "\n" + "xyz", as lm-eval-harness's _encode_pair scores it
        self.assertEqual(seen, ["ab\nxyz"])

    def test_macro_f1_matches_scikit_learn_on_a_known_case(self):
        # Per label: 0 has tp 1, fn 1 -> 2/3; 1 has tp 1, fp 2 -> 1/2; 2 has tp 0 -> 0.
        # f1_score([0, 0, 1, 2], [0, 1, 1, 1], average="macro") = (2/3 + 1/2 + 0) / 3 = 7/18.
        self.assertAlmostEqual(when2call.f1_macro([0, 0, 1, 2], [0, 1, 1, 1]), 7 / 18)

    def test_metrics_follow_the_repository(self):
        rows = [
            # Raw likelihood prefers choice 1; normalised by length, choice 3 wins.
            {"target": 3, "correct": "cannot_answer", "no_tools": True, "choice_lengths": [10, 10, 10, 40],
             "logliks": [-9.0, -5.0, -9.0, -8.0]},
            {"target": 1, "correct": "tool_call", "no_tools": False, "choice_lengths": [10, 10, 10, 10],
             "logliks": [-9.0, -1.0, -9.0, -9.0]},
        ]
        metrics = when2call.mcq_metrics(rows)
        self.assertEqual((metrics["acc"], metrics["acc_norm"]), (0.5, 1.0))
        self.assertEqual(metrics["hallucination_rate"], 1.0)  # no tools, cannot_answer, chose the call
        self.assertEqual(metrics["confusion"]["cannot_answer"], {"tool_call": 1})


class GenerateAndJudgeTests(unittest.TestCase):
    def test_tools_are_converted_as_the_inference_script_does(self):
        tool = when2call.inference_tools(ROW)["get_weather"]["function"]
        self.assertEqual(tool["name"], "get_weather")
        self.assertEqual(tool["parameters"]["type"], "object")
        self.assertEqual({p["type"] for p in tool["parameters"]["properties"].values()}, {"string"})

    def test_a_reply_is_recorded_as_its_first_call_or_its_text(self):
        call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>'
        self.assertEqual(json.loads(when2call.as_response(call)), {"name": "get_weather", "arguments": {"city": "Oslo"}})
        self.assertEqual(when2call.as_response("  Which day?  "), "Which day?")

    def test_judge_scores_are_the_share_of_the_correct_kind(self):
        judged = [{"correct_answer": "tool_call", "classification": "tool_call"},
                  {"correct_answer": "tool_call", "classification": "cannot_answer"},
                  {"correct_answer": "request_for_info", "classification": "request_for_info"},
                  {"correct_answer": "cannot_answer", "classification": "cannot_answer"}]
        scores = when2call.judge_scores(judged)
        self.assertEqual(scores["tool_call"]["score"], 0.5)
        self.assertEqual(scores["mean"], round((0.5 + 1 + 1) / 3, 4))


DOCS = {"MathAPI": [{"name": "add", "description": "a + b", "parameters": {
            "type": "dict", "properties": {"a": {"type": "float"}, "b": {"type": "float"}}, "required": ["a", "b"]}},
        {"name": "sqrt", "description": "root", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}}}}]}
ENTRY = {"id": "multi_turn_base_0", "involved_classes": ["MathAPI"], "excluded_function": ["sqrt"],
         "initial_config": {}, "question": [[{"role": "user", "content": "Add 1 and 2."}],
                                            [{"role": "user", "content": "Thanks."}]]}


def reply(*calls, text=""):
    blocks = "".join(f'<tool_call>\n{json.dumps({"name": n, "arguments": a})}\n</tool_call>' for n, a in calls)
    return text + blocks


class BFCLTests(unittest.TestCase):
    def test_tools_drop_excluded_functions_and_use_json_schema_types(self):
        tools = bfcl.tools_for(ENTRY, DOCS)
        self.assertEqual(list(tools), ["add"])
        self.assertEqual(tools["add"]["function"]["parameters"]["properties"]["a"]["type"], "number")

    def test_a_call_string_holds_only_literals(self):
        text = bfcl.call_string(ToolCall("echo", {"content": "x'); __import__('os')#", "n": [1, {"k": None}]}))
        tree = ast.parse(text, mode="eval").body
        self.assertEqual(tree.func.id, "echo")
        self.assertEqual(ast.literal_eval(tree.keywords[0].value), "x'); __import__('os')#")
        self.assertTrue(all(isinstance(k.value, (ast.Constant, ast.List)) for k in tree.keywords))

    def test_each_turn_runs_until_the_model_stops_calling(self):
        replies = iter([reply(("add", {"a": 1, "b": 2})), "3.", "You're welcome."])
        executed = []

        def execute(strings):
            executed.append(strings)
            return ["{'result': 3.0}"] * len(strings)
        decoded, messages, stop = bfcl.run_entry(ENTRY, bfcl.tools_for(ENTRY, DOCS), lambda m: next(replies), execute)
        self.assertEqual(decoded, [[["add(a=1, b=2)"]], []])
        self.assertEqual(executed, [["add(a=1, b=2)"]])
        self.assertIsNone(stop)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "assistant", "user", "assistant"])

    def test_a_function_that_was_not_offered_is_never_executed(self):
        replies = iter([reply(("sqrt", {"x": 4}), ("add", {"a": 1, "b": 1})), "done", "ok"])
        executed = []
        decoded, messages, _ = bfcl.run_entry(ENTRY, bfcl.tools_for(ENTRY, DOCS), lambda m: next(replies),
                                              lambda s: executed.append(s) or ["2"] * len(s))
        self.assertEqual(executed, [["add(a=1, b=1)"]])
        self.assertEqual(decoded[0], [["add(a=1, b=1)"]])
        self.assertIn("not an available function", messages[2]["content"])

    def test_a_turn_stops_after_twenty_steps_and_a_server_error_ends_the_entry(self):
        calls = lambda m: reply(("add", {"a": 1, "b": 1}))
        decoded, _, stop = bfcl.run_entry(ENTRY, bfcl.tools_for(ENTRY, DOCS), calls, lambda s: ["2"] * len(s))
        self.assertEqual((len(decoded[0]), stop), (bfcl.MAX_STEPS, "max_steps"))

        def broken(messages):
            raise RuntimeError("400: prompt too long")
        decoded, _, stop = bfcl.run_entry(ENTRY, bfcl.tools_for(ENTRY, DOCS), broken, lambda s: [])
        self.assertEqual(decoded, [[], []])
        self.assertTrue(stop.startswith("generate_failed"))


class CommandTests(unittest.TestCase):
    def test_both_commands_are_registered(self):
        from agent_from_scratch.evals.__main__ import COMMANDS
        self.assertEqual((COMMANDS["when2call"], COMMANDS["bfcl"]), ("external.when2call", "external.bfcl"))


if __name__ == "__main__":
    unittest.main()
