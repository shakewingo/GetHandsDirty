import json
import unittest

from unittest.mock import Mock, patch
from agent_from_scratch.llm import LLM, LLMResponse, ResponseError, ResponseErrorCode, RESPONSE_ERROR_MESSAGES
from agent_from_scratch.llm import _QWEN_TEMPLATE


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
        self.assertEqual((result.type, result.call_id, result.tool_params),
                         ("tool_call", "c1", {"left": 2}))

    def test_qwen_extra_braces(self):
        result = self.parse('<tool_call>{{"name":"calculator","arguments":{}}}</tool_call>')
        self.assertEqual(result.tool_name, "calculator")

    def test_qwen_json_string_arguments_match_native_call(self):
        function = {"name": "read_file", "arguments": json.dumps({"path": "tools/files.py"})}
        qwen = self.parse("<tool_call>\n" + json.dumps(function) + "\n</tool_call>")
        native = self.parse(None, tool_calls=[{"function": function}])
        self.assertEqual((qwen.tool_name, qwen.tool_params),
                         (native.tool_name, native.tool_params))
        self.assertEqual(qwen.tool_params, {"path": "tools/files.py"})
        self.assertEqual(qwen.to_message()["tool_calls"][0]["function"], function)

    def test_qwen_encoded_arguments_must_decode_once_to_an_object(self):
        for arguments in ('{broken}', '[]', 'null', '42', json.dumps('{"path":"a"}')):
            content = "<tool_call>" + json.dumps({"name": "read_file", "arguments": arguments}) + "</tool_call>"
            with self.subTest(arguments=arguments), self.assertRaises(ResponseError) as caught:
                self.parse(content)
            self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_TOOL_CALL)

    def test_multiple_calls_are_rejected(self):
        with self.assertRaises(ResponseError) as caught:
            self.parse(None, tool_calls=[{}, {}])
        self.assertEqual(caught.exception.code, ResponseErrorCode.MULTIPLE_TOOL_CALLS)
        with self.assertRaises(ResponseError) as caught:
            self.parse("<tool_call>{}</tool_call>" * 2)
        self.assertEqual(caught.exception.code, ResponseErrorCode.MULTIPLE_TOOL_CALLS)


class GenerateTests(unittest.TestCase):
    def test_template_renders_arguments_once_without_changing_values(self):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        from agent_from_scratch.tools.register import default_registry
        arguments = {"path": "a.txt", "content": 'a "quote"\\slash\n你好'}
        call = LLMResponse("assistant", "", "tool_call", "write_file", arguments)
        formatter = Jinja2ChatFormatter(
            template=_QWEN_TEMPLATE.read_text(), eos_token="<|im_end|>",
            bos_token="<|endoftext|>",
        )
        rendered = formatter(messages=[{"role": "user", "content": "Write"}, call.to_message()],
                             tools=list(default_registry.schemas().values())).prompt
        payload = rendered.rsplit("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        self.assertEqual(json.loads(payload)["arguments"], arguments)
        self.assertNotIn('{{"name"', rendered)
        self.assertIn("only one complete", rendered)

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
