from __future__ import annotations

import json
from typing import Any
from .llm import LLM, ResponseError, ResponseType
from typing import TYPE_CHECKING
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


class Agent:
    def __init__(self, llm: "LLM"):
        self.llm = llm
        self.max_iterations = 10
        self.max_tool_calls = 2

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
            response = self.llm.generate(messages, tool_schemas)
            messages.append({"role": "assistant", "content": response.content})
            if response.type == ResponseType.tool_call:
                logger.debug(
                    "Tool call detected: {} {}",
                    response.tool_name,
                    response.tool_params,
                )
                tool_iter_ = 0
                while tool_iter_ < self.max_tool_calls:
                    tool_iter_ += 1
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
                        break
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
                break
        logger.debug(f"Final messages: {messages}")
        return TurnResult(messages=messages)

    def run_repl(
        self,
    ):
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            turn_result = self.run_turn(user_input)
            print(turn_result.messages[-1].get("content", "Sorry I don't know."))


if __name__ == "__main__":
    llm = LLM()
    agent = Agent(llm)
    agent.run_repl()
