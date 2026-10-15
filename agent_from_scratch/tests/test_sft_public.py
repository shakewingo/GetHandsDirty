"""Public rows become exact (prompt, completion) examples in our template, or are dropped and counted."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.config import BENCH_N_CTX, PROMPTS_DIR
from agent_from_scratch.evals.sft.public import collect, json_schema, Dropped
from agent_from_scratch.llm import LLM, QWEN_END

SYSTEM = (PROMPTS_DIR / "system.md").read_text(encoding="utf-8").strip()
WEATHER = json.dumps({"name": "get_weather", "description": "Weather for a city.",
                      "parameters": {"type": "dict", "properties": {
                          "city": {"type": "str", "description": "City name", "default": ""},
                          "days": {"type": "int", "description": "Days ahead", "default": 1}},
                          "required": ["city"]}})
TOUCAN_TOOLS = json.dumps([{"type": "function", "function": {
    "name": "reverse", "description": "Reverse text.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}])


def short(text):
    return len(text) // 4


def w2c_sft(reply, tools=(WEATHER,)):
    return {"tools": list(tools), "messages": [{"role": "user", "content": "Weather in Oslo?"},
                                               {"role": "assistant", "content": reply}]}


def w2c_pref(chosen):
    return {"tools": [WEATHER], "messages": [{"role": "user", "content": "Weather in Oslo?"}],
            "chosen_response": {"role": "assistant", "content": chosen},
            "rejected_response": {"role": "assistant", "content": "No."}}


def toucan(messages, uuid="u1", subset="multi-turn"):
    return {"uuid": uuid, "subset_name": subset, "tools": TOUCAN_TOOLS, "messages": json.dumps(messages)}


def call(name, arguments):
    return {"role": "tool_call", "content": repr({"name": name, "arguments": json.dumps(arguments)})}


def parse(completion):
    return LLM.parse_response({"choices": [{"message": {"role": "assistant",
                                                        "content": completion.removesuffix(QWEN_END)}}]})


class ConvertTests(unittest.TestCase):
    def test_a_when2call_reply_is_one_example_in_our_template(self):
        examples, funnel = collect([w2c_sft("Which city do you mean?")], "when2call-sft", count=short)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertTrue(example["prompt"].startswith("<|im_start|>system\n" + SYSTEM))
        self.assertTrue(example["prompt"].endswith("<|im_start|>assistant\n"))
        self.assertEqual(example["completion"], "Which city do you mean?" + QWEN_END)
        tool = json.loads(next(line for line in example["prompt"].splitlines() if "get_weather" in line))
        parameters = tool["function"]["parameters"]
        self.assertEqual((parameters["type"], parameters["properties"]["city"]["type"],
                          parameters["properties"]["days"]["type"]), ("object", "string", "integer"))
        self.assertEqual(example["provenance"]["skeleton"], "when2call-sft:question")
        self.assertEqual(funnel["kept_rows_by_category"], {"question": 1})

    def test_a_when2call_tool_call_becomes_our_tool_call_block(self):
        chosen = '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}]</TOOLCALL>'
        examples, funnel = collect([w2c_pref(chosen)], "when2call-pref", count=short)
        completion = examples[0]["completion"]
        self.assertIn("<tool_call>", completion)
        self.assertNotIn("TOOLCALL", completion)
        calls = parse(completion).tool_calls
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Oslo", "days": 2})])
        self.assertEqual(funnel["kept_rows_by_category"], {"call": 1})

    def test_toucan_narration_calls_and_results_become_turns_and_context(self):
        row = toucan([{"role": "user", "content": "Reverse abc and xyz."},
                      {"role": "assistant", "content": "I'll reverse both."},
                      call("reverse", {"text": "abc"}), call("reverse", {"text": "xyz"}),
                      {"role": "tool_response", "content": "cba"}, {"role": "tool_response", "content": "zyx"},
                      {"role": "assistant", "content": "cba and zyx."}])
        examples, _ = collect([row], "toucan", count=short)
        self.assertEqual(len(examples), 2)
        first, second = examples
        parsed = parse(first["completion"])
        self.assertEqual(parsed.content, "I'll reverse both.")
        self.assertEqual([c.arguments["text"] for c in parsed.tool_calls], ["abc", "xyz"])
        self.assertEqual(second["completion"], "cba and zyx." + QWEN_END)
        # Tool results are context in the next prompt, in our envelope, and never a target.
        envelopes = [json.loads(block.split("\n")[0]) for block in
                     second["prompt"].split("<tool_response>\n")[1:]]
        self.assertEqual([(e["tool_name"], e["ok"], e["output"]) for e in envelopes],
                         [("reverse", True, "cba"), ("reverse", True, "zyx")])
        for example in examples:
            self.assertNotIn("<tool_response>", example["completion"])
        self.assertEqual(second["provenance"]["skeleton"], "toucan:multi-turn")
        self.assertEqual(first["provenance"]["task_id"], second["provenance"]["task_id"])

    def test_every_completion_parses_back_and_ends_with_one_stop_token(self):
        rows = [w2c_sft("I can't do that."), w2c_sft("Which city?", tools=())]
        examples, _ = collect(rows, "when2call-sft", count=short)
        for example in examples:
            self.assertTrue(example["completion"].endswith(QWEN_END))
            self.assertEqual(example["completion"].count(QWEN_END), 1)
            self.assertEqual(parse(example["completion"]).content + QWEN_END, example["completion"])


class DropTests(unittest.TestCase):
    def dropped(self, rows, source, count=short):
        examples, funnel = collect(rows, source, count=count)
        self.assertEqual(examples, [])
        return funnel["dropped"]

    def test_thinking_text_is_dropped(self):
        self.assertEqual(self.dropped([w2c_sft("<think>hmm</think>Which city?")], "when2call-sft"), {"thinking": 1})

    def test_an_unparseable_call_is_dropped(self):
        self.assertEqual(self.dropped([w2c_pref("<TOOLCALL>[{oops</TOOLCALL>")], "when2call-pref"), {"bad_call": 1})
        row = toucan([{"role": "user", "content": "x"}, {"role": "tool_call", "content": "not a dict"},
                      {"role": "tool_response", "content": "y"}, {"role": "assistant", "content": "z"}])
        self.assertEqual(self.dropped([row], "toucan"), {"bad_call": 1})

    def test_more_than_eight_calls_in_one_turn_is_dropped(self):
        calls = [call("reverse", {"text": str(n)}) for n in range(9)]
        row = toucan([{"role": "user", "content": "x"}, *calls,
                      *[{"role": "tool_response", "content": "r"}] * 9, {"role": "assistant", "content": "done"}])
        self.assertEqual(self.dropped([row], "toucan"), {"too_many_calls": 1})

    def test_an_unknown_parameter_type_is_dropped(self):
        tool = json.dumps({"name": "f", "parameters": {"type": "dict", "properties": {"x": {"type": "Any"}}}})
        self.assertEqual(self.dropped([w2c_sft("Which?", tools=(tool,))], "when2call-sft"), {"unknown_type": 1})

    def test_a_row_longer_than_the_window_is_dropped_not_truncated(self):
        self.assertEqual(self.dropped([w2c_sft("Which city?")], "when2call-sft", count=lambda text: BENCH_N_CTX + 1),
                         {"too_long": 1})

    def test_a_result_without_a_call_or_a_call_without_a_result_is_dropped(self):
        orphan = toucan([{"role": "user", "content": "x"}, {"role": "tool_response", "content": "y"},
                         {"role": "assistant", "content": "z"}])
        unanswered = toucan([{"role": "user", "content": "x"}, call("reverse", {"text": "a"}),
                             {"role": "assistant", "content": "z"}])
        self.assertEqual(self.dropped([orphan, unanswered], "toucan"), {"unpaired": 2})

    def test_json_schema_maps_nested_types(self):
        schema = {"type": "dict", "properties": {"tags": {"type": "list", "items": {"type": "str"}}}}
        self.assertEqual(json_schema(schema), {"type": "object", "properties": {
            "tags": {"type": "array", "items": {"type": "string"}}}})
        with self.assertRaises(Dropped):
            json_schema({"type": "callable"})

    def test_python_type_hints_map_without_inventing_item_types(self):
        for name, expected in (("str, optional", "string"), ("int, optional, default=100", "integer"),
                               ("List[int]", "array"), ("Tuple[float, float]", "array"),
                               ("List[List[str]]", "array")):
            with self.subTest(name=name):
                self.assertEqual(json_schema({"type": name}), {"type": expected})
        for name in ("List[Union[int, float]]", "Callable[[float], float]"):
            with self.subTest(name=name), self.assertRaises(Dropped):
                json_schema({"type": name})


class SamplingTests(unittest.TestCase):
    def test_whole_rows_in_a_seeded_order_until_the_target(self):
        rows = [w2c_sft(f"Which city, {n}?") for n in range(10)]
        first, funnel = collect(rows, "when2call-sft", count=short, max_examples=4, seed=3)
        again, _ = collect(rows, "when2call-sft", count=short, max_examples=4, seed=3)
        other, _ = collect(rows, "when2call-sft", count=short, max_examples=4, seed=4)
        self.assertEqual(len(first), 4)
        self.assertEqual(first, again)
        self.assertNotEqual([e["id"] for e in first], [e["id"] for e in other])
        self.assertEqual(funnel["rows_visited"], 4)

    def test_the_output_trains_with_the_unchanged_train_command(self):
        from agent_from_scratch.evals.sft.train import load_dataset
        from agent_from_scratch.evals.sft.tensorize import select_examples

        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "rows.jsonl"
            source.write_text("".join(json.dumps(w2c_sft(f"Which city, {n}?")) + "\n" for n in range(3)))
            gguf = Path(temporary) / "student.gguf"
            gguf.write_bytes(b"")
            out = Path(temporary) / "public"
            # The command needs a real GGUF for the token count; check its refusals, then write
            # the dataset through the same code path with a stub counter.
            result = subprocess.run([sys.executable, "-m", "agent_from_scratch.evals", "public",
                                     "--source", "when2call-sft", "--input", str(source), "--revision", "r",
                                     "--student-gguf", str(Path(temporary) / "missing.gguf"),
                                     "--output", str(out)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("student-gguf", result.stderr)
            self.assertFalse(out.exists())
            from agent_from_scratch.evals.sft import public
            from unittest.mock import patch

            argv = ["public", "--source", "when2call-sft", "--input", str(source), "--revision", "r",
                    "--student-gguf", str(gguf), "--output", str(out)]
            with patch.object(sys, "argv", argv), \
                    patch("agent_from_scratch.evals.sft.teacher.student_token_counter", return_value=(short, "stub")):
                public.main()
            examples, metadata = load_dataset(out)
            self.assertEqual(len(examples), 3)
            self.assertEqual(metadata["source"]["revision"], "r")
            self.assertEqual(len(select_examples(examples, max_per_skeleton=10)), 3)


if __name__ == "__main__":
    unittest.main()
