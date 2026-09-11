from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Dict, Any, TYPE_CHECKING, List
from .utils import render_prompt, decode_qwen_tool_call

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionTool, ChatCompletionRequestMessage


class ResponseType(StrEnum):
    tool_call = "tool_call"
    direct = "direct"


@dataclass
class LLMResponse:
    role: str
    content: str
    type: ResponseType
    tool_name: str | None = None
    tool_params: dict | None = None
    usage: dict | None = None
    call_id: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ResponseErrorCode(StrEnum):
    INVALID_RESPONSE = "invalid_response"
    INVALID_TOOL_CALL = "invalid_tool_call"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED_RESPONSE = "truncated_response"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"


RESPONSE_ERROR_MESSAGES = {
    ResponseErrorCode.INVALID_RESPONSE: "Invalid model response structure.",
    ResponseErrorCode.INVALID_TOOL_CALL: "Invalid tool call.",
    ResponseErrorCode.EMPTY_RESPONSE: "Model response has no text or tool call.",
    ResponseErrorCode.TRUNCATED_RESPONSE: "Model response was truncated.",
    ResponseErrorCode.MULTIPLE_TOOL_CALLS: "Only one tool call per response is supported.",
}


class ResponseError(ValueError):
    def __init__(self, code: ResponseErrorCode, detail: str = ""):
        self.code = code
        message = RESPONSE_ERROR_MESSAGES[code]
        super().__init__(f"{message} {detail}" if detail else message)


def parse_response(response) -> LLMResponse:
    """
    Example 1 w tool_calls: {'id': 'chatcmpl-xxx', 'object': 'chat.completion', 'created': 1789008759, 'model': 'qwen2.5.gguf', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '<tool_call>\n{{"name": "calculator", "arguments": {"operation": "add", "left": 2, "right": 2}}}\n</tool_call>'}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 284, 'completion_tokens': 32, 'total_tokens': 316}}
    Example 2 wo tool_calls: {'choices':[{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of China is Beijing.'}, 'logprobs': None, 'finish_reason': 'stop'}]}
    """
    try:
        choice = response["choices"][0]
        message = choice["message"]
        role = message["role"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ResponseError(ResponseErrorCode.INVALID_RESPONSE) from error
    if content is not None and not isinstance(content, str):
        raise ResponseError(ResponseErrorCode.INVALID_RESPONSE, "Content must be text or null.")
    if choice.get("finish_reason") == "length":
        raise ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE)
    usage = response.get("usage", {})
    calls = message.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ResponseError(
            ResponseErrorCode.INVALID_TOOL_CALL, "tool_calls must be a list."
        )
    if calls:
        if len(calls) > 1:
            raise ResponseError(ResponseErrorCode.MULTIPLE_TOOL_CALLS)
        try:
            call = calls[0]
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            name = function["name"]
            call_id = call.get("id", "")
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ResponseError(
                ResponseErrorCode.INVALID_TOOL_CALL, "Cannot decode native tool call."
            ) from error
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ResponseError(
                ResponseErrorCode.INVALID_TOOL_CALL,
                "Expected a string name and object arguments.",
            )
        return LLMResponse(
            role=role,
            content=content or "",
            type=ResponseType.tool_call,
            tool_name=name,
            tool_params=arguments,
            usage=usage,
            call_id=call_id,
        )
    if not isinstance(content, str) or not content.strip():
        raise ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
    if "<tool_call>" in content or "</tool_call>" in content:
        # qwen sepcific tool_call format processing
        if content.count("<tool_call>") > 1:
            raise ResponseError(ResponseErrorCode.MULTIPLE_TOOL_CALLS)
        try:
            tool_content = decode_qwen_tool_call(content)
        except ValueError as error:
            raise ResponseError(
                ResponseErrorCode.INVALID_TOOL_CALL, str(error)
            ) from error
        tool_name = tool_content.get("name")
        tool_params = tool_content.get("arguments")
        return LLMResponse(
            role=role,
            content=content,
            type=ResponseType.tool_call,
            tool_name=tool_name,
            tool_params=tool_params,
            usage=usage,
        )
    return LLMResponse(
        role=role,
        content=content,
        type=ResponseType.direct,
        tool_name=None,
        tool_params=None,
        usage=usage,
    )


_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "gz-data"
    / "hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f"
    / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
)


class LLM:
    def __init__(
        self,
        model_path: str = str(_MODEL_PATH),
        temperature: float = 0.7,
        max_tokens: int = 512,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        verbose=False,  # turn off tensor / metadata loading, prefix-match, timing info from llama-cpp-python
    ):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        tools: Dict[str, ChatCompletionTool],
    ) -> LLMResponse:
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=list(tools.values()),
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return parse_response(response)


if __name__ == "__main__":
    from .tools.register import tool_schemas

    user_input = "2+2 is what?"
    llm = LLM()
    messages: list[ChatCompletionRequestMessage] = [
        {"role": "system", "content": render_prompt("system.md")},
        {"role": "user", "content": user_input},
    ]
    print(llm.generate(messages, tools=tool_schemas))
