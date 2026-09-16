from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from hashlib import sha256
from dataclasses import dataclass, field
from enum import StrEnum
import json
from typing import Dict, Any, TYPE_CHECKING, List

from .config import (MAX_TOKENS, MAX_TOOL_CALLS_PER_RESPONSE, MODEL_PATH, N_CTX,
                     N_GPU_LAYERS, QWEN_TEMPLATE, TEMPERATURE)
from .tools.base import ToolCall
from .utils import render_prompt, extract_qwen_tool_calls

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
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None
    finish_reason: str | None = None
    raw_response: Mapping[str, Any] | None = None

    def to_message(self) -> ChatCompletionRequestAssistantMessage:
        if self.type == ResponseType.direct:
            return {"role": "assistant", "content": self.content}

        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


class ResponseErrorCode(StrEnum):
    INVALID_RESPONSE = "invalid_response"
    INVALID_TOOL_CALL = "invalid_tool_call"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED_RESPONSE = "truncated_response"
    TOO_MANY_TOOL_CALLS = "too_many_tool_calls"
    UNSUPPORTED_FINISH_REASON = "unsupported_finish_reason"


RESPONSE_ERROR_MESSAGES = {
    ResponseErrorCode.INVALID_RESPONSE: "Invalid model response structure.",
    ResponseErrorCode.INVALID_TOOL_CALL: "Invalid tool call.",
    ResponseErrorCode.EMPTY_RESPONSE: "Model response has no text or tool call.",
    ResponseErrorCode.TRUNCATED_RESPONSE: "Model response was truncated.",
    ResponseErrorCode.TOO_MANY_TOOL_CALLS: f"At most {MAX_TOOL_CALLS_PER_RESPONSE} tool calls per response are supported.",
    ResponseErrorCode.UNSUPPORTED_FINISH_REASON: "Unsupported model finish reason.",
}


class ResponseError(ValueError):
    def __init__(self, code: ResponseErrorCode, detail: str = "", *, raw_response: Any = None):
        self.code = code
        self.raw_response = raw_response
        message = RESPONSE_ERROR_MESSAGES[code]
        super().__init__(f"{message} {detail}" if detail else message)


def install_qwen_template(model, path: Path) -> str:
    """Install the project's checked Qwen2.5 format on this instance only."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    if model.metadata.get("general.architecture") != "qwen2":
        raise ValueError("The project chat template requires Qwen2.")
    eos_id, bos_id = model.token_eos(), model.token_bos()
    if min(eos_id, bos_id) < 0:
        raise ValueError("Missing Qwen special token IDs.")
    eos, bos = (model.detokenize([token_id], special=True).decode("utf-8")
                for token_id in (eos_id, bos_id))
    for token, token_id in ((eos, eos_id), (bos, bos_id)):
        if model.tokenize(token.encode(), add_bos=False, special=True) != [token_id]:
            raise ValueError(f"Unexpected Qwen special token: {token}")
    template = path.read_text(encoding="utf-8")
    if not eos or eos not in template:
        raise ValueError("The chat template does not use this model's end token.")
    model.chat_handler = Jinja2ChatFormatter(
        template=template, eos_token=eos, bos_token=bos, stop_token_ids=[eos_id],
    ).to_chat_handler()
    return sha256(template.encode("utf-8")).hexdigest()


class LLM:
    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        n_gpu_layers: int = N_GPU_LAYERS,
        n_ctx: int = N_CTX,
        verbose=False,  # turn off tensor / metadata loading, prefix-match, timing info from llama-cpp-python
        chat_template_path: str | Path | None = QWEN_TEMPLATE,
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
        """Validate the entire native/Qwen batch before allowing any execution."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
            role = message["role"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise ResponseError(ResponseErrorCode.INVALID_RESPONSE) from error
        if role != "assistant":
            raise ResponseError(ResponseErrorCode.INVALID_RESPONSE, "Expected an assistant message.")
        if content is not None and not isinstance(content, str):
            raise ResponseError(ResponseErrorCode.INVALID_RESPONSE, "Content must be text or null.")
        if choice.get("finish_reason") == "length":
            raise ResponseError(ResponseErrorCode.TRUNCATED_RESPONSE)
        if choice.get("finish_reason") not in (None, "stop", "tool_calls", "function_call"):
            raise ResponseError(ResponseErrorCode.UNSUPPORTED_FINISH_REASON,
                                str(choice.get("finish_reason")))
        usage = LLM.read_usage(response.get("usage"))
        # Generate tool content extraction
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ResponseError(
                ResponseErrorCode.INVALID_TOOL_CALL, "tool_calls must be a list."
            )
        if calls:
            if len(calls) > MAX_TOOL_CALLS_PER_RESPONSE:
                raise ResponseError(ResponseErrorCode.TOO_MANY_TOOL_CALLS)
            parsed_calls = []
            for call in calls:
                try:
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
                        "Expected a string name, object arguments and string call ID.",
                    )
                if call_id and any(previous.call_id == call_id for previous in parsed_calls):
                    raise ResponseError(ResponseErrorCode.INVALID_TOOL_CALL, "Duplicate tool-call ID.")
                parsed_calls.append(ToolCall(name, arguments, call_id))
            return LLMResponse(
                role=role,
                content=content or "",
                type=ResponseType.tool_call,
                tool_calls=parsed_calls,
                usage=usage,
                finish_reason=choice.get("finish_reason"),
            )
        if not isinstance(content, str) or not content.strip():
            raise ResponseError(ResponseErrorCode.EMPTY_RESPONSE)
        try:
            # Specific tool extraction case for qwen model
            extracted, narration = extract_qwen_tool_calls(content)
        except ValueError as error:
            raise ResponseError(
                ResponseErrorCode.INVALID_TOOL_CALL, str(error)
            ) from error
        if len(extracted) > MAX_TOOL_CALLS_PER_RESPONSE:
            raise ResponseError(ResponseErrorCode.TOO_MANY_TOOL_CALLS)
        if extracted:
            return LLMResponse(
                role=role,
                content=narration,
                type=ResponseType.tool_call,
                tool_calls=[ToolCall(call["name"], call["arguments"]) for call in extracted],
                usage=usage,
                finish_reason=choice.get("finish_reason"),
            )
        return LLMResponse(
            role=role,
            content=content,
            type=ResponseType.direct,
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
            stream=False,
        )
        try:
            if not isinstance(response, dict):
                raise ResponseError(ResponseErrorCode.INVALID_RESPONSE,
                                    "Expected a non-streaming response object.")
            parsed = LLM.parse_response(response)
            parsed.raw_response = response
            return parsed
        except ResponseError as error:
            error.raw_response = response
            raise

    def close(self) -> None:
        """Release native model resources before interpreter shutdown."""
        self.llm.close()


if __name__ == "__main__":
    from .tools.register import default_tool_schemas

    user_input = "What is 2*2?"
    llm = LLM()
    messages: list[ChatCompletionRequestMessage] = [
        {"role": "system", "content": render_prompt("system.md")},
        {"role": "user", "content": user_input},
    ]
    print(llm.generate(messages, tools=default_tool_schemas))
