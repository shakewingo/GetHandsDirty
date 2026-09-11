from __future__ import annotations

import json
from typing import Any
from .llm import LLM, ResponseError, ResponseType
from typing import TYPE_CHECKING, Literal
from .tools.register import registry, tool_schemas
from .tools.base import ToolResult
from .utils import render_prompt
from loguru import logger
from dataclasses import dataclass

if TYPE_CHECKING:
    from llama_cpp import ChatCompletionRequestMessage


@dataclass
class TurnResult:
    messages: list[ChatCompletionRequestMessage]
    final_answer: str | None = None
    stop_reason: Literal["final_response", "max_iterations", "response_error"] = (
        "max_iterations"
    )
    error_message: str | None = None


class Agent:
    def __init__(self, llm: "LLM"):
        self.llm = llm
        self.max_iterations = 20

    def execute_tool(
        self, tool_name: Any, tool_params: Any, call_id: str = ""
    ) -> ToolResult:
        return registry.invoke(tool_name, tool_params, call_id=call_id)

    def run_turn(self, user_input: str) -> TurnResult:
        messages: list[ChatCompletionRequestMessage] = [
            {"role": "system", "content": render_prompt("system.md")},
            {"role": "user", "content": user_input},
        ]
        iter_ = 0
        while iter_ < self.max_iterations:
            iter_ += 1
            # Use messages to generate response, inject response and do max_iteration or meet exit conditions for final output
            try:
                response = self.llm.generate(messages, tool_schemas)
            except ResponseError as error:
                logger.error(f"LLM response error: {error}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be processed: {error}. "
                            "Please correct its format and try again."
                        ),
                    }
                )
                continue
            if response.type == ResponseType.tool_call and not response.call_id:
                response.call_id = f"call_{iter_}"
            messages.append(response.to_message())
            if response.type == ResponseType.tool_call:
                logger.debug(
                    "Tool call detected: {} {}",
                    response.tool_name,
                    response.tool_params,
                )
                tool_result = self.execute_tool(
                    response.tool_name, response.tool_params, response.call_id
                )
                if tool_result.ok:
                    logger.debug("Tool executed successfully: {}", tool_result)
                    messages.append(
                        {
                            "role": "tool",
                            "content": f"Tool output: {json.dumps(tool_result.output)}",
                            "tool_call_id": tool_result.call_id,
                        }
                    )
                else:
                    logger.error(
                        "Tool execution failed due to: {}", tool_result.error_message
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": f"Tool calling failed: {tool_result.error_message}",
                            "tool_call_id": tool_result.call_id,
                        }
                    )
            elif response.type == ResponseType.direct:
                return TurnResult(
                    messages=messages,
                    final_answer=response.content,
                    stop_reason="final_response",
                )
        logger.debug(
            "Final messages:\n{}",
            json.dumps(messages, indent=2, ensure_ascii=False),
        )
        return TurnResult(messages=messages)

    def run_repl(
        self,
    ):
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            turn_result = self.run_turn(user_input)
            if turn_result.final_answer is not None:
                print(turn_result.final_answer)
            elif turn_result.stop_reason == "response_error":
                print(
                    f"Could not process the model response: {turn_result.error_message}"
                )
            else:
                print("Stopped: maximum iterations reached without a final answer.")


if __name__ == "__main__":
    llm = LLM()
    agent = Agent(llm)
    agent.run_repl()
