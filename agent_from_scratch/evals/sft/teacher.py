"""A stronger API model behind the same interface as `VLLMClient`, for Stage 9 teacher rollouts.

The teacher is DeepSeek's OpenAI-compatible chat API. The harness's message list is already in
that shape, so it is sent nearly as it is and the reply goes straight through
`LLM.parse_response`, the one parser. The API has no token-count endpoint, so the context
budget is the student's: the request rendered with the project template and counted with the
student's own vocabulary. The harness triggers then fire exactly where the student's would.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Callable

from ...config import BENCH_DECODING, BENCH_N_CTX, CHAT_TEMPLATE_PATH, MAX_TOOL_CALLS_PER_RESPONSE
from ...llm import (LLM, QWEN_END, LLMResponse, ResponseError, ResponseErrorCode,
                    compile_chat_template, render_chat)

BASE_URL = "https://api.deepseek.com"
API_KEY_VARIABLE = "DEEPSEEK_API_KEY"


def student_token_counter(gguf: Path) -> tuple[Callable[[str], int], str]:
    """(count, label) from a GGUF's vocabulary alone; no weights are loaded.

    Args:
        gguf: a GGUF of the student model. Only its tokenizer is read.
    """
    from llama_cpp import Llama

    vocabulary = Llama(model_path=str(gguf), vocab_only=True, verbose=False)
    label = f"{vocabulary.metadata.get('general.name')} ({Path(gguf).name})"
    return (lambda text: len(vocabulary.tokenize(text.encode("utf-8"), add_bos=False, special=True)),
            label)


class TeacherClient:
    """DeepSeek's chat API as the model backend, with the same interface as `LLM`.

    Args:
        model: API model ID. An alias such as `deepseek-flash` moves between releases, so the
            release name the server reports is recorded in `settings()`.
        count: student tokenizer, text -> token count (see `student_token_counter`).
        budget_tokenizer: label for `count`, recorded in `settings()`.
        effort: `reasoning_effort` (low, high, max); "none" turns thinking off.
        sample: bookkeeping label for this pass. The API takes no seed.
        max_output_tokens: the teacher's own reply limit. Thinking counts toward it.
        n_ctx, reserve: the student's window and reply reserve, used for the budget only.
    """

    def __init__(self, model: str = "deepseek-flash", *, count: Callable[[str], int],
                 budget_tokenizer: str, effort: str = "low", sample: int = 0,
                 max_output_tokens: int = 8192, n_ctx: int = BENCH_N_CTX,
                 reserve: int = BENCH_DECODING["max_tokens"], base_url: str = BASE_URL,
                 timeout: float = 600.0, retries: int = 3, backoff: float = 2.0, client=None):
        import httpx

        if client is None:
            key = os.environ.get(API_KEY_VARIABLE)
            if not key:
                raise ValueError(f"Set {API_KEY_VARIABLE} to run the teacher.")
            client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  headers={"Authorization": f"Bearer {key}"})
        self.http, self.base_url, self.retries, self.backoff = client, base_url, retries, backoff
        self.model, self.effort, self.sample = model, effort, sample
        self.max_output_tokens, self.n_ctx, self.reserve = max_output_tokens, n_ctx, reserve
        self._count, self.budget_tokenizer = count, budget_tokenizer
        self._template = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
        served = {entry["id"]: entry for entry in self._request("GET", "/models")["data"]}
        if model not in served:
            raise ValueError(f"The API does not serve {model!r}; it serves {sorted(served)}.")
        self.model_name = served[model].get("name")
        if count("<|im_start|>" + QWEN_END) != 2:
            raise ValueError("The chat template's <|im_start|>/<|im_end|> are not single tokens "
                             "in the student tokenizer.")
        # First tool-call ID of a turn -> the reasoning that came with it. The API asks for a
        # turn's reasoning_content back whenever tools are in play; the harness does not keep it.
        self._reasoning: dict[str, str] = {}

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        import httpx

        for attempt in range(self.retries):
            try:
                response = self.http.request(method, path, json=body)
                if response.status_code < 500 and response.status_code != 429:
                    response.raise_for_status()
                    return response.json()
                error: Exception = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            except (httpx.TransportError, httpx.TimeoutException) as transport:
                error = transport
            if attempt == self.retries - 1:
                raise error
            time.sleep(self.backoff * 2 ** attempt)
        raise AssertionError("unreachable")

    def _messages(self, messages: list) -> list[dict]:
        """The harness's messages with only the fields the API defines."""
        converted = []
        for message in messages:
            entry = {"role": message["role"], "content": message.get("content") or ""}
            if message["role"] == "tool":
                entry["tool_call_id"] = message["tool_call_id"]
            elif message["role"] == "assistant" and message.get("tool_calls"):
                entry["tool_calls"] = message["tool_calls"]
                reasoning = self._reasoning.get(message["tool_calls"][0]["id"])
                if reasoning is not None:
                    entry["reasoning_content"] = reasoning
            converted.append(entry)
        return converted

    def measure_context(self, messages, tools, *, max_tokens: int | None = None) -> dict[str, Any]:
        """Same contract as `LLM.measure_context`, in student tokens against the student's window."""
        configured = self.reserve if max_tokens is None else max_tokens
        reserve = configured if configured is not None and configured > 0 else None
        count = self._count(render_chat(self._template, messages, tools))
        return {"count_method": "exact", "prompt_tokens": count, "window_tokens": self.n_ctx,
                "response_reserve": reserve,
                "remaining_tokens": self.n_ctx - count - reserve if reserve is not None else None}

    def generate(self, messages, tools, *, max_tokens: int | None = None,
                 max_tool_calls: int | None = None) -> LLMResponse:
        # Thinking counts toward max_tokens, so the caller's reserve (sized for the student) is
        # not used here. Thinking mode ignores sampling parameters and the API takes no seed.
        body: dict[str, Any] = {"model": self.model, "messages": self._messages(messages),
                                "max_tokens": self.max_output_tokens, "stream": False}
        if self.effort == "none":
            body["thinking"] = {"type": "disabled"}
        else:
            body.update(thinking={"type": "enabled"}, reasoning_effort=self.effort)
        if tools:
            body["tools"] = list(tools.values())
        raw = self._request("POST", "/chat/completions", body)
        try:
            parsed = LLM.parse_response(
                raw, MAX_TOOL_CALLS_PER_RESPONSE if max_tool_calls is None else max_tool_calls)
        except ResponseError as error:
            error.raw_response = raw
            raise
        reasoning = raw["choices"][0]["message"].get("reasoning_content")
        if parsed.tool_calls and isinstance(reasoning, str):
            self._reasoning[parsed.tool_calls[0].call_id] = reasoning
        parsed.raw_response = raw
        return parsed

    def settings(self) -> dict[str, Any]:
        return {"backend": "deepseek", "base_url": self.base_url, "model": self.model,
                "model_name": self.model_name, "effort": self.effort, "sample": self.sample,
                "max_tokens": self.max_output_tokens, "n_ctx": self.n_ctx,
                "response_reserve": self.reserve, "budget_tokenizer": self.budget_tokenizer,
                # The template is used for the budget only; the teacher never sampled under it.
                "chat_template_sha256": None}

    def close(self) -> None:
        self.http.close()
