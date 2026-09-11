import unittest

from unittest.mock import Mock, patch
from agent_from_scratch.llm import LLM, LLMResponse, parse_response, ResponseError, ResponseErrorCode, RESPONSE_ERROR_MESSAGES


class ResponseTests(unittest.TestCase):
    def parse(self, content, **fields):
        return parse_response({
            "choices": [{"message": {"role": "assistant", "content": content, **fields}}]
        })

    def test_plain_text_mentions_tool_call(self):
        self.assertEqual(self.parse("The tool_call field.").type, "direct")

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
            parse_response({"choices": []})
        self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_RESPONSE)

    def test_truncated_response(self):
        with self.assertRaises(ResponseError) as caught:
            parse_response({"choices": [{
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

    def test_multiple_calls_are_rejected(self):
        with self.assertRaises(ResponseError) as caught:
            self.parse(None, tool_calls=[{}, {}])
        self.assertEqual(caught.exception.code, ResponseErrorCode.MULTIPLE_TOOL_CALLS)
        with self.assertRaises(ResponseError) as caught:
            self.parse("<tool_call>{}</tool_call>" * 2)
        self.assertEqual(caught.exception.code, ResponseErrorCode.MULTIPLE_TOOL_CALLS)


class GenerateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
