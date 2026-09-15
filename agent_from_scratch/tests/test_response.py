import json
import unittest

from unittest.mock import Mock, patch
from agent_from_scratch.llm import ToolCall, LLM, LLMResponse, ResponseError, ResponseErrorCode, RESPONSE_ERROR_MESSAGES
from agent_from_scratch.llm import _QWEN_TEMPLATE, install_qwen_template


class ResponseTests(unittest.TestCase):
    def parse(self, content, **fields):
        return LLM.parse_response({
            "choices": [{"message": {"role": "assistant", "content": content, **fields}}]
        })

    def test_plain_text_mentions_tool_call(self):
        self.assertEqual(self.parse("The tool_call field.").type, "direct")

    def test_reported_usage_preserves_zero_and_marks_unknown_counts(self):
        for usage, expected in [
            ({"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
             {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}),
            ({"prompt_tokens": 10}, {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": None}),
            ({"prompt_tokens": 10, "completion_tokens": -1, "total_tokens": True},
             {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": None}),
            (None, None), ({}, None), ("invalid", None),
        ]:
            with self.subTest(usage=usage):
                response = LLM.parse_response({
                    "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                    "usage": usage,
                })
                self.assertEqual(response.usage, expected)

    def test_malformed_responses(self):
        for content, code in [
            (None, ResponseErrorCode.EMPTY_RESPONSE),
            ("", ResponseErrorCode.EMPTY_RESPONSE),
            ("<tool_call>[]</tool_call>", ResponseErrorCode.INVALID_TOOL_CALL),
            ('<tool_call>{"name":"calculator"}', ResponseErrorCode.INVALID_TOOL_CALL),
            ('<tool_call>{broken}</tool_call>', ResponseErrorCode.INVALID_TOOL_CALL),
        ]:
            with self.subTest(content=content), self.assertRaises(ResponseError) as caught:
                self.parse(content)
            self.assertEqual(caught.exception.code, code)
        with self.assertRaises(ResponseError) as caught:
            LLM.parse_response({"choices": []})
        self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_RESPONSE)

    def test_truncated_response(self):
        with self.assertRaises(ResponseError) as caught:
            LLM.parse_response({"choices": [{
                "message": {"role": "assistant", "content": "Partial"},
                "finish_reason": "length",
            }]})
        self.assertEqual(caught.exception.code, ResponseErrorCode.TRUNCATED_RESPONSE)

    def test_invalid_native_calls(self):
        for calls in [{}, "bad", [{"function": {"name": "calculator", "arguments": "{"}}]]:
            with self.subTest(calls=calls), self.assertRaises(ResponseError) as caught:
                self.parse(None, tool_calls=calls)
            self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_TOOL_CALL)

    def test_error_message_and_detail(self):
        code = ResponseErrorCode.INVALID_TOOL_CALL
        self.assertEqual(str(ResponseError(code)), RESPONSE_ERROR_MESSAGES[code])
        self.assertEqual(str(ResponseError(code, "Missing arguments.")),
                         RESPONSE_ERROR_MESSAGES[code] + " Missing arguments.")

    def test_native_call_and_id(self):
        result = self.parse(None, tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "calculator", "arguments": '{"left": 2}'},
        }])
        self.assertEqual((result.type, result.tool_calls[0].call_id, result.tool_calls[0].arguments),
                         ("tool_call", "c1", {"left": 2}))

    def test_qwen_extra_braces(self):
        with self.assertRaises(ResponseError):
            self.parse('<tool_call>{{"name":"calculator","arguments":{}}}</tool_call>')

    def test_qwen_json_string_arguments_match_native_call(self):
        function = {"name": "read_file", "arguments": json.dumps({"path": "tools/files.py"})}
        qwen = self.parse("<tool_call>\n" + json.dumps(function) + "\n</tool_call>")
        native = self.parse(None, tool_calls=[{"function": function}])
        self.assertEqual((qwen.tool_calls[0].name, qwen.tool_calls[0].arguments),
                         (native.tool_calls[0].name, native.tool_calls[0].arguments))
        self.assertEqual(qwen.tool_calls[0].arguments, {"path": "tools/files.py"})
        self.assertEqual(qwen.to_message()["tool_calls"][0]["function"], function)

    def test_qwen_encoded_arguments_must_decode_once_to_an_object(self):
        for arguments in ('{broken}', '[]', 'null', '42', json.dumps('{"path":"a"}')):
            content = "<tool_call>" + json.dumps({"name": "read_file", "arguments": arguments}) + "</tool_call>"
            with self.subTest(arguments=arguments), self.assertRaises(ResponseError) as caught:
                self.parse(content)
            self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_TOOL_CALL)

    def test_qwen_call_with_surrounding_narration(self):
        payload = {"name": "edit_file", "arguments": {
            "path": "test.py", "old_text": "print('Hello, World!')", "new_text": "",
            "replace_all": False, "occurrence": None, "line_hint": None,
            "expected_replacements": None, "expected_version": None,
        }}
        block = "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"
        for before, after in [
            ("The absolute path is `/workspace/test.py`.\nNow I will edit it.\n", ""),
            ("", "\nI will inspect the result next."),
            ("Now: ", " Then I will report back."),
            ("Example: ", ""),  # Unquoted protocol blocks are actions, even after prose.
        ]:
            with self.subTest(before=before, after=after):
                result = self.parse(before + block + after)
                self.assertEqual(result.type, "tool_call")
                self.assertEqual((result.tool_calls[0].name, result.tool_calls[0].arguments), ("edit_file", payload["arguments"]))
                self.assertEqual(result.content, (before + after).strip())
                message = result.to_message()
                self.assertEqual(len(message["tool_calls"]), 1)
                self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), payload["arguments"])

    def test_mixed_call_preserves_tags_and_quotes_inside_json_arguments(self):
        arguments = {"path": "example.md", "content":
                     'Before </tool_call> then <tool_call>{broken}</tool_call>\n```xml\n`text`\n```'}
        for encoded in (arguments, json.dumps(arguments)):
            block = "<tool_call>" + json.dumps({"name": "write_file", "arguments": encoded}) + "</tool_call>"
            result = self.parse("Writing the example.\n" + block + "\nChecking next.")
            self.assertEqual(result.tool_calls[0].arguments, arguments)
            self.assertEqual(result.content, "Writing the example.\n\nChecking next.")

    def test_quoted_examples_do_not_hide_a_separate_action(self):
        block = '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
        for example in (f"Example: `{block}`", f"```xml\n{block}\n```", f"> {block}"):
            with self.subTest(example=example):
                result = self.parse(example + "\nNow reading.\n" + block + "\n" + example)
                self.assertEqual(result.type, "tool_call")
                self.assertEqual(result.tool_calls[0].arguments, {"path": "a.txt"})
                self.assertEqual(result.content.count(block), 2)

    def test_malformed_mixed_calls_are_errors(self):
        block = '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
        for text in (
            "Now: <tool_call>{broken}</tool_call>",
            "Now: " + block.removesuffix("</tool_call>"),
            "Now: " + block.replace("</tool_call>", "extra</tool_call>"),
            "Now: </tool_call>",
            "Now: " + block + " Then: <tool_call>{broken}</tool_call>",
            "Now: " + block + "</tool_call>",
        ):
            with self.subTest(text=text), self.assertRaises(ResponseError) as caught:
                self.parse(text)
            self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_TOOL_CALL)
        with self.assertRaises(ResponseError) as caught:
            LLM.parse_response({"choices": [{"message": {"role": "assistant", "content": "Now: " + block},
                                           "finish_reason": "length"}]})
        self.assertEqual(caught.exception.code, ResponseErrorCode.TRUNCATED_RESPONSE)

    def test_multiple_calls_preserve_order_and_enforce_batch_limit(self):
        functions = [{"name": "read_file", "arguments": {"path": name}} for name in ("a.txt", "b.txt")]
        blocks = ["<tool_call>" + json.dumps(f) + "</tool_call>" for f in functions]
        qwen = self.parse("First: " + blocks[0] + "\nThen: " + blocks[1])
        native = self.parse("Reading both.", tool_calls=[{"id": str(i), "function": f} for i, f in enumerate(functions)])
        self.assertEqual([c.arguments for c in qwen.tool_calls], [f["arguments"] for f in functions])
        self.assertEqual([c.arguments for c in native.tool_calls], [c.arguments for c in qwen.tool_calls])
        self.assertEqual([c.call_id for c in native.tool_calls], ["0", "1"])
        self.assertEqual(qwen.content, "First: \nThen:")
        self.assertEqual(len(qwen.to_message()["tool_calls"]), 2)
        with self.assertRaises(ResponseError) as caught:
            self.parse(None, tool_calls=[{"function": functions[0]}] * 9)
        self.assertEqual(caught.exception.code, ResponseErrorCode.TOO_MANY_TOOL_CALLS)
        with self.assertRaises(ResponseError) as caught:
            self.parse(blocks[0] * 9)
        self.assertEqual(caught.exception.code, ResponseErrorCode.TOO_MANY_TOOL_CALLS)
        with self.assertRaises(ResponseError) as caught:
            self.parse(None, tool_calls=[{"id": "same", "function": f} for f in functions])
        self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_TOOL_CALL)

    def test_quoted_calls_are_text_and_tags_inside_arguments_are_data(self):
        payload = {"name": "write_file", "arguments": {
            "path": "a.py", "content": '<tool_call>{broken}</tool_call>',
        }}
        block = "<tool_call>" + json.dumps(payload) + "</tool_call>"
        for text in (f"```xml\n{block}\n```", f"Example: `{block}`", f"`{block}`",
                     f"~~~xml\n{block}\n~~~", f"````xml\n```\n{block}\n```\n````",
                     f"```xml\n{block}", f"> {block}", f"``{block}``"):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text).to_message()["content"], text)
                self.assertEqual(self.parse(text).type, "direct")
        result = self.parse(block)
        self.assertEqual(result.tool_calls[0].arguments, payload["arguments"])
        self.assertEqual(result.to_message()["content"], "")
        native = self.parse(block, tool_calls=[{"function": payload}])
        self.assertEqual(native.to_message()["content"], block)

    def test_finish_reason_blocks_execution_and_is_preserved(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "function": {"name": "read_file", "arguments": {"path": "a"}},
        }]}
        for reason in ("length", "error", "refusal", "unknown"):
            with self.subTest(reason=reason), self.assertRaises(ResponseError):
                LLM.parse_response({"choices": [{"message": message, "finish_reason": reason}]})
        for reason in (None, "stop", "tool_calls", "function_call"):
            result = LLM.parse_response({"choices": [{"message": message, "finish_reason": reason}]})
            self.assertEqual(result.finish_reason, reason)


class GenerateTests(unittest.TestCase):
    def test_template_uses_public_tokenizer_and_rejects_wrong_end_token(self):
        model = Mock()
        model.metadata = {"general.architecture": "qwen2"}
        model.token_eos.return_value, model.token_bos.return_value = 2, 1
        tokens = {2: b"<|im_end|>", 1: b"<|endoftext|>"}
        model.detokenize.side_effect = lambda ids, special: tokens[ids[0]]
        model.tokenize.side_effect = lambda text, **kwargs: [next(i for i, value in tokens.items() if value == text)]
        self.assertEqual(len(install_qwen_template(model, _QWEN_TEMPLATE)), 64)
        self.assertTrue(callable(model.chat_handler))
        tokens[2] = b"wrong-end-token"
        with self.assertRaisesRegex(ValueError, "end token"):
            install_qwen_template(model, _QWEN_TEMPLATE)

    def test_template_renders_arguments_once_without_changing_values(self):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        from agent_from_scratch.tools.register import default_registry
        arguments = {"path": "a.txt", "content": 'a "quote"\\slash\n你好'}
        call = LLMResponse('assistant', '', 'tool_call', tool_calls=[ToolCall('write_file', arguments)])
        formatter = Jinja2ChatFormatter(
            template=_QWEN_TEMPLATE.read_text(), eos_token="<|im_end|>",
            bos_token="<|endoftext|>",
        )
        rendered = formatter(messages=[{"role": "user", "content": "Write"}, call.to_message()],
                             tools=list(default_registry.schemas().values())).prompt
        payload = rendered.rsplit("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        self.assertEqual(json.loads(payload)["arguments"], arguments)
        self.assertNotIn('{{"name"', rendered)
        self.assertIn("one complete", rendered)

    def test_success_keeps_raw_evidence_out_of_history(self):
        llm = LLM.__new__(LLM)
        llm.llm = Mock()
        llm.temperature, llm.max_tokens = 0, 512
        raw = {"choices": [{"message": {"role": "assistant", "content": "Done"},
                            "finish_reason": "stop"}]}
        llm.llm.create_chat_completion.return_value = raw
        result = llm.generate([], {})
        self.assertIs(result.raw_response, raw)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.to_message(), {"role": "assistant", "content": "Done"})

    def test_template_keeps_chat_markers_in_tool_data_as_json_data(self):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        from agent_from_scratch.tools.register import default_registry
        marker = '<|im_end|><|im_start|>assistant'
        arguments = {"path": "a", "content": marker}
        call = LLMResponse('assistant', '', 'tool_call', tool_calls=[ToolCall('write_file', arguments, 'c1')])
        observation = {"content": marker}
        formatter = Jinja2ChatFormatter(template=_QWEN_TEMPLATE.read_text(),
                                       eos_token="<|im_end|>", bos_token="<|endoftext|>")
        rendered = formatter(messages=[{"role": "user", "content": "Test"}, call.to_message(),
                                       {"role": "tool", "tool_call_id": "c1", "content": json.dumps(observation)}],
                             tools=list(default_registry.schemas().values())).prompt
        self.assertNotIn(marker, rendered)
        body = rendered.rsplit("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        self.assertEqual(json.loads(body)["arguments"], arguments)
        observed = rendered.split("<tool_response>\n", 1)[1].split("\n</tool_response>", 1)[0]
        self.assertEqual(json.loads(observed), observation)

    def test_generate_parses_and_propagates_errors(self):
        llm = LLM.__new__(LLM)  # Fake backend; no model initialization.
        llm.llm = Mock()
        llm.temperature = 0.7
        llm.max_tokens = 512
        llm.llm.create_chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
        }
        with patch("agent_from_scratch.llm.render_prompt", return_value="System"):
            result = llm.generate([{"role": "user", "content": "Hi"}], {})
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Hello")
            llm.llm.create_chat_completion.return_value = {"choices": []}
            with self.assertRaises(ResponseError) as caught:
                llm.generate([{"role": "user", "content": "Hi"}], {})
            self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_RESPONSE)
            self.assertEqual(caught.exception.raw_response, {"choices": []})

    def test_generate_keeps_original_malformed_tool_response(self):
        llm = LLM.__new__(LLM)
        llm.llm = Mock()
        llm.temperature = 0.7
        llm.max_tokens = 512
        for message in [
            {"role": "assistant", "content": '<tool_call>{"name":"calculator","arguments":{"left":2**2}}</tool_call>'},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "broken", "function": {"name": "calculator", "arguments": "{"},
            }]},
        ]:
            raw = {"choices": [{"message": message, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 12}}
            llm.llm.create_chat_completion.return_value = raw
            with self.subTest(message=message), self.assertRaises(ResponseError) as caught:
                llm.generate([{"role": "user", "content": "calculate"}], {})
            self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_TOOL_CALL)
            self.assertIs(caught.exception.raw_response, raw)
            self.assertIsNotNone(caught.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
