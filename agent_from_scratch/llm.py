from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Dict, Any, TYPE_CHECKING, List
from .utils import render_prompt, decode_qwen_tool_call

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionTool, ChatCompletionRequestMessage, ChatCompletionRequestAssistantMessage


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
    raw_response: dict | None = None

    def to_message(self) -> ChatCompletionRequestAssistantMessage:
        if self.type == ResponseType.direct:
            return {"role": "assistant", "content": self.content}

        # parse_response() already validates the name and arguments.
        assert self.tool_name is not None
        assert self.tool_params is not None

        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [{
                "id": self.call_id,
                "type": "function",
                "function": {
                    "name": self.tool_name,
                    "arguments": json.dumps(self.tool_params),
                },
            }],
        }


class ResponseErrorCode(StrEnum):
    INVALID_RESPONSE = "invalid_response"
    INVALID_TOOL_CALL = "invalid_tool_call"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED_RESPONSE = "truncated_response"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    UNSUPPORTED_FINISH_REASON = "unsupported_finish_reason"


RESPONSE_ERROR_MESSAGES = {
    ResponseErrorCode.INVALID_RESPONSE: "Invalid model response structure.",
    ResponseErrorCode.INVALID_TOOL_CALL: "Invalid tool call.",
    ResponseErrorCode.EMPTY_RESPONSE: "Model response has no text or tool call.",
    ResponseErrorCode.TRUNCATED_RESPONSE: "Model response was truncated.",
    ResponseErrorCode.MULTIPLE_TOOL_CALLS: "Only one tool call per response is supported.",
    ResponseErrorCode.UNSUPPORTED_FINISH_REASON: "Unsupported model finish reason.",
}


class ResponseError(ValueError):
    def __init__(self, code: ResponseErrorCode, detail: str = "", *, raw_response: Any = None):
        self.code = code
        self.raw_response = raw_response
        message = RESPONSE_ERROR_MESSAGES[code]
        super().__init__(f"{message} {detail}" if detail else message)


_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "gz-data"
    / "hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f"
    / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
)
_QWEN_TEMPLATE = Path(__file__).parent / "prompts" / "qwen_chat.jinja"


def install_qwen_template(model, path: Path) -> str:
    """Install the project's checked Qwen2.5 format on this instance only."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    if model.metadata.get("general.architecture") != "qwen2":
        raise ValueError("The project chat template requires Qwen2.")
    for token, token_id in (("<|im_end|>", model.token_eos()),
                            ("<|endoftext|>", model.token_bos())):
        if model.tokenize(token.encode(), add_bos=False, special=True) != [token_id]:
            raise ValueError(f"Unexpected Qwen special token: {token}")
    template = path.read_text(encoding="utf-8")
    model.chat_handler = Jinja2ChatFormatter(
        template=template, eos_token="<|im_end|>", bos_token="<|endoftext|>",
        stop_token_ids=[model.token_eos()],
    ).to_chat_handler()
    return sha256(template.encode("utf-8")).hexdigest()


class LLM:
    def __init__(
        self,
        model_path: str = str(_MODEL_PATH),
        temperature: float = 0.7,
        max_tokens: int = 512,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        verbose=False,  # turn off tensor / metadata loading, prefix-match, timing info from llama-cpp-python
        chat_template_path: str | Path | None = None,
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
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.chat_template_sha256 = (install_qwen_template(self.llm, Path(chat_template_path))
                                     if chat_template_path is not None else None)

    def settings(self) -> dict[str, Any]:
        """Snapshot the actual configuration used by this model instance."""
        return {"model_path": self.model_path, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "chat_template_sha256": getattr(self, "chat_template_sha256", None),
                "chat_handler": "project_qwen_jinja" if getattr(self, "chat_template_sha256", None)
                else str(self.llm.chat_format)}

    @staticmethod
    def read_usage(usage: Any) -> dict[str, int | None] | None:
        """Keep reported counts; missing or invalid counts stay unknown, never zero."""
        if not isinstance(usage, dict):
            return None
        counts = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            counts[name] = value if type(value) is int and value >= 0 else None
        return counts if any(value is not None for value in counts.values()) else None

    @staticmethod
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
        if choice.get("finish_reason") not in (None, "stop", "tool_calls", "function_call"):
            raise ResponseError(ResponseErrorCode.UNSUPPORTED_FINISH_REASON,
                                str(choice.get("finish_reason")))
        usage = LLM.read_usage(response.get("usage"))
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
            if (not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict)
                    or not isinstance(call_id, str)):
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
                finish_reason=choice.get("finish_reason"),
            )
        if not isinstance(content, str) or not content.strip():
            raise ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
        if content.strip().startswith("<tool_call>"):
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
                content="",
                type=ResponseType.tool_call,
                tool_name=tool_name,
                tool_params=tool_params,
                usage=usage,
                finish_reason=choice.get("finish_reason"),
            )
        return LLMResponse(
            role=role,
            content=content,
            type=ResponseType.direct,
            tool_name=None,
            tool_params=None,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
        )

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
        try:
            parsed = LLM.parse_response(response)
            parsed.raw_response = response
            return parsed
        except ResponseError as error:
            error.raw_response = response
            raise


if __name__ == "__main__":
    from .tools.register import default_tool_schemas

    user_input = "What is 2*2?"
    llm = LLM()
    messages: list[ChatCompletionRequestMessage] = [
        {"role": "system", "content": render_prompt("system.md")},
        {"role": "user", "content": user_input},
    ]
    print(llm.generate(messages, tools=default_tool_schemas))
