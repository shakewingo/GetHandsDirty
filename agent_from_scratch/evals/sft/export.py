# collect, examples_from_trace, rejection, select
import json
from pathlib import Path

from ...llm import LLM, QWEN_END, ResponseError, render_chat


def examples_from_trace(trace: dict, template) -> list[dict]:
    """One example per completed actor request, in request order.

    Args:
        trace: the JSON object in `state/runs/<run_id>.jsonl` (schema 5).
        template: `compile_chat_template(...)` of prompts/qwen_chat.jinja.

    Returns:
        list of {"request_index", "prompt", "completion", "raw_matches_canonical"}.
    """
    messages, examples = trace["messages"], []
    for index, request in enumerate(trace["model_requests"]):
        if request["purpose"] != "agent" or request["status"] != "completed":
            continue
        count = request["input_message_count"]
        turn = messages[count] if count < len(messages) else {}
        calls = turn.get("tool_calls") or []
        if turn.get("role") != "assistant" or [call["id"] for call in calls] != request["call_ids"]:
            raise ValueError(f"Request {index} does not line up with messages[{count}].")
        prompt = render_chat(template, request["input_messages"], request["tools"])
        # The target is what the template adds when the turn joins the conversation. The server
        # stops on <|im_end|>, so the newline the template writes after it is never generated.
        full = render_chat(template, [*request["input_messages"], turn], request["tools"],
                           add_generation_prompt=False)
        if not full.startswith(prompt) or not full.endswith(QWEN_END + "\n"):
            raise ValueError(f"Request {index}: the rendered turn does not extend its prompt.")
        completion = full[len(prompt):].removesuffix("\n")
        # Round trip: the text the model is trained to write must parse to the action it took.
        try:
            parsed = LLM.parse_response({"choices": [{"message": {
                "role": "assistant", "content": completion.removesuffix(QWEN_END)}}]}, len(calls))
        except ResponseError as error:
            raise ValueError(f"Request {index}: the target does not parse: {error}") from error
        taken = [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in calls]
        if ([(call.name, call.arguments) for call in parsed.tool_calls] != taken
                or parsed.content != (turn.get("content") or "")):
            raise ValueError(f"Request {index}: the target does not parse back to the recorded turn.")
        try:
            sampled = request["raw_response"]["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            sampled = None  # Not the vLLM completion shape: nothing was sampled as Qwen text.
        examples.append({"request_index": index, "prompt": prompt, "completion": completion,
                         "raw_matches_canonical": None if sampled is None
                         else sampled + QWEN_END == completion})
    return examples


def rejection(record: dict, task: dict, trace: dict) -> str | None:
    """Why a rollout may not become training data; None when it may.

    Args:
        record: the task's entry in results.json (verifier outcome plus metrics()).
        task: the task's task.json, read for `fault_signal` only.
        trace: the run trace.
    """

def select(trajectories: list[dict], max_per_task: int) -> list[dict]:
    """Distinct trajectories per task, shortest first, at most `max_per_task` each.

    Each trajectory has at least `task_id`, `seed`, `run_dir` and `examples`.
    """

def collect(run_dirs: list[Path], track: str, *, max_per_task: int = 2,
            pilot: bool = False) -> tuple[list[dict], dict]:
    """Read bench run directories and return (examples, metadata) for one track."""