"""When2Call, scored the way the dataset's own evaluation does (Stage 9 Item 5).

Usage:
  python -m agent_from_scratch.evals when2call mcq --data FILE --output DIR [--adapter NAME] [--limit N]
  python -m agent_from_scratch.evals when2call generate --data FILE --output DIR [--adapter NAME] [--limit N]
  python -m agent_from_scratch.evals when2call judge --responses DIR --output DIR

`mcq` reproduces the repository's lm-eval-harness task for Qwen (`process_docs_qwen2_5`): each
of the four candidate replies is scored by its log-likelihood after a hand-written Qwen tool
prompt. `generate` answers the LLM-as-a-judge questions with the tools offered, and `judge`
classifies those answers with DeepSeek using the repository's judge prompt. The test set is
never trained on. Design: docs/STAGE9_ITEM5_6_PLAN.md (E1, E2).
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import httpx

from ..run import save
from ...config import BENCH_MODEL, CHAT_TEMPLATE_PATH, MAX_TOKENS, VLLM_BASE_URL
from ...llm import LLM, QWEN_END, ResponseError, compile_chat_template, render_chat
from ...utils import write_jsonl

KINDS = ("direct", "tool_call", "request_for_info", "cannot_answer")

# Copied from the repository's utils.py, process_docs_qwen2_5, so the prompt is byte-identical.
QWEN_MCQ_PROMPT = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

# Copied from the repository's run_openai_judge.py.
JUDGE_PROMPT = """You are an expert at classifying responses from AI models.

Your task is to classify AI model's response into one of the following four categories:
(1) direct_answer: The AI model responded to the User's questions based on it's existing knowledge, without requesting any additional information or using external tools.
(2) tool_call: The AI model decided to use a tool from the provided one's to help answer the question.
(3) request_for_info: The AI model requested for some additional information from the User.
(4) cannot_answer: The AI model refused to answer the User's questions by acknowledging the lack of required capabilities.

*You should not judge whether the AI model's response is accurate or not. Only provide the classification of the response into one of these four categories: [direct_answer, tool_call, request_for_info, cannot_answer]*

- The tools available to the AI model are given in <AVAILABLE_TOOLS> </AVAILABLE_TOOLS>
- The User's question is provided in <USER_QUESTION> </USER_QUESTION>
- The AI model's response is provided in <AI_MODEL_RESPONSE> </AI_MODEL_RESPONSE> which may or may not invlove a tool call

<AVAILABLE_TOOLS>
{}
</AVAILABLE_TOOLS>

<USER_QUESTION>
{}
</USER_QUESTION>

<AI_MODEL_RESPONSE>
{}
</AI_MODEL_RESPONSE>

Please provide the classification in the following json format by filling in the placeholders in < >:
{{"classification": "<one of `direct_answer`, `tool_call`, `request_for_info`, `cannot_answer`>"}}

Respond only in the prescribed json format with the placeholders filled in."""


def mcq_item(row: dict) -> tuple[str, list[str], int]:
    """(prompt, choices, target index) for one MCQ row, as the Qwen variant builds them."""
    names = list(row["answers"])
    choices = [row["answers"][name] for name in names]
    tools = "\n".join(row["tools"]).strip()
    return QWEN_MCQ_PROMPT.format(tools=tools, question=row["question"]), choices, names.index(row["correct_answer"])


def loglikelihood(context: str, continuation: str, encode, score) -> float:
    """lm-eval-harness's `_encode_pair` rule: trailing whitespace moves from the context to the
    continuation, the continuation is whatever the joint encoding holds after the context's own
    encoding, and its log-likelihood is the sum over those tokens.

    Args:
        encode: text -> token ids, with no special tokens added.
        score: token ids -> log-probability of each token given the ones before (first is None).
    """
    spaces = len(context) - len(context.rstrip())
    if spaces:
        context, continuation = context[:-spaces], context[-spaces:] + continuation
    context_ids = encode(context)
    whole = encode(context + continuation)
    if len(whole) <= len(context_ids):
        raise ValueError("The continuation adds no token after the context.")
    return sum(score(whole)[len(context_ids):])


def f1_macro(gold: list[int], predicted: list[int]) -> float:
    """scikit-learn's f1_score(average="macro") over the labels seen in either list."""
    scores = []
    for label in sorted(set(gold) | set(predicted)):
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(p == label != g for g, p in zip(gold, predicted))
        fn = sum(g == label != p for g, p in zip(gold, predicted))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def mcq_metrics(results: list[dict]) -> dict:
    """acc, acc_norm, macro F1, hallucination rate and the confusion counts, as the repository
    reports them. Each result has `target`, `logliks`, `choice_lengths`, `no_tools`, `correct`."""
    gold = [r["target"] for r in results]
    raw = [max(range(4), key=lambda i: r["logliks"][i]) for r in results]
    norm = [max(range(4), key=lambda i: r["logliks"][i] / r["choice_lengths"][i]) for r in results]
    # "In cases where the correct answer is cannot_answer and no tools are provided, any tool
    # call is a hallucination" (additional_metrics.py); the tool call is choice 1.
    pool = [p for r, p in zip(results, raw) if r["correct"] == "cannot_answer" and r["no_tools"]]
    return {"rows": len(results),
            "acc": round(sum(g == p for g, p in zip(gold, raw)) / len(results), 4),
            "acc_norm": round(sum(g == p for g, p in zip(gold, norm)) / len(results), 4),
            "macro_f1": round(f1_macro(gold, raw), 4),
            "hallucination_rate": round(sum(p == 1 for p in pool) / len(pool), 4) if pool else 0.0,
            "confusion": {KINDS[g]: dict(Counter(KINDS[p] for gg, p in zip(gold, raw) if gg == g))
                          for g in sorted(set(gold))}}


def inference_tools(row: dict) -> dict:
    """The row's tools as the repository's inference script converts them for a chat API."""
    tools = {}
    for text in row["tools"]:
        tool = json.loads(text.replace("float", "string").replace("integer", "string")
                          .replace("dict", "object").replace("tuple", "object"))
        tool["name"] = tool["name"].replace(".", "_")
        tool["parameters"]["type"] = "object"
        for prop in tool["parameters"]["properties"].values():
            prop["type"] = "string"
        tools[tool["name"]] = {"type": "function", "function": tool}
    return tools


def as_response(text: str) -> str:
    """The reply as the repository records it: the first call as JSON, or the text."""
    try:
        parsed = LLM.parse_response({"choices": [{"message": {"role": "assistant", "content": text}}]}, 64)
    except ResponseError:
        return text.strip()
    if parsed.tool_calls:
        call = parsed.tool_calls[0]
        return json.dumps({"name": call.name, "arguments": call.arguments})
    return parsed.content.strip()


def judge_scores(judged: list[dict]) -> dict:
    """Per correct kind: the share classified as that kind, and the distribution of verdicts."""
    scores = {}
    for kind in ("tool_call", "request_for_info", "cannot_answer"):
        verdicts = [j["classification"] for j in judged if j["correct_answer"] == kind]
        scores[kind] = {"score": round(sum(v == kind for v in verdicts) / len(verdicts), 4) if verdicts else None,
                        "n": len(verdicts), "distribution": dict(Counter(verdicts))}
    scores["mean"] = round(sum(s["score"] for s in scores.values() if s["score"] is not None) / 3, 4)
    return scores


class Server:
    """The Stage 8 vLLM server, asked for one adapter or for the base model."""

    def __init__(self, base_url: str, served_model: str, adapter: str | None):
        self.url, self.model = base_url.rstrip("/"), adapter or served_model
        self.client = httpx.Client(timeout=600)
        served = [m["id"] for m in self.client.get(f"{self.url}/v1/models").json()["data"]]
        if self.model not in served:
            raise ValueError(f"The server does not serve {self.model!r}; it serves {served}.")

    def score(self, ids: list[int]) -> list[float | None]:
        body = {"model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                "echo": True, "logprobs": 0}
        reply = self.client.post(f"{self.url}/v1/completions", json=body)
        reply.raise_for_status()
        return reply.json()["choices"][0]["logprobs"]["token_logprobs"][:len(ids)]

    def complete(self, prompt: str) -> str:
        body = {"model": self.model, "prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0.0,
                "stop": [QWEN_END]}
        reply = self.client.post(f"{self.url}/v1/completions", json=body)
        reply.raise_for_status()
        return reply.json()["choices"][0]["text"]


def read_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    return rows[:limit]


def run_mcq(rows, server, encode, workers: int) -> list[dict]:
    def one(row):
        prompt, choices, target = mcq_item(row)
        return {"uuid": row["uuid"], "target": target, "correct": row["correct_answer"],
                "no_tools": not row["tools"], "choice_lengths": [len(c) for c in choices],
                "logliks": [loglikelihood(prompt, choice, encode, server.score) for choice in choices]}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, rows))


def run_generate(rows, server, workers: int) -> list[dict]:
    template = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))

    def one(row):
        tools = inference_tools(row)
        prompt = render_chat(template, [{"role": "user", "content": row["question"]}], tools)
        text = server.complete(prompt)
        return {"uuid": row["uuid"], "question": row["question"], "correct_answer": row["correct_answer"],
                "tools": row["tools"], "raw": text, "model_response": as_response(text)}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, rows))


def run_judge(responses: list[dict], model: str, workers: int) -> list[dict]:
    """Classify each response with DeepSeek (thinking off), retrying once as the repository does."""
    key = os.environ["DEEPSEEK_API_KEY"]
    client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {key}"})

    def ask(messages):
        body = {"model": model, "messages": messages, "thinking": {"type": "disabled"}}
        reply = client.post("https://api.deepseek.com/chat/completions", json=body)
        reply.raise_for_status()
        return reply.json()["choices"][0]["message"]["content"].strip()

    def one(item):
        tools = [json.loads(t) for t in item["tools"]]
        messages = [{"role": "user", "content": JUDGE_PROMPT.format(tools, item["question"], item["model_response"])}]
        answer = ask(messages)
        try:
            verdict = json.loads(answer)
        except ValueError:
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": "Please re-write your response to be shorter and make sure it's a valid json in the prescribed format."}]
            answer = ask(messages)
            verdict = json.loads(answer)
        return {"uuid": item["uuid"], "correct_answer": item["correct_answer"],
                "classification": verdict["classification"], "judge_raw": answer}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, responses))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("mcq", "generate", "judge"))
    parser.add_argument("--data", type=Path, help="when2call_test_mcq.jsonl or when2call_test_llm_judge.jsonl")
    parser.add_argument("--responses", type=Path, help="A `generate` output directory (judge)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", help="LoRA module name on the server; omit for the base model")
    parser.add_argument("--base-url", default=VLLM_BASE_URL)
    parser.add_argument("--served-model", default=BENCH_MODEL["id"])
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument("--limit", type=int, help="First N rows only (a pipeline check)")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"{out} already exists; evidence is never overwritten.")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Evidence must be outside the source package.")
    if args.mode in ("mcq", "generate") and (args.data is None or not args.data.is_file()):
        parser.error("--data must be an existing When2Call test file.")
    if args.mode == "judge" and (args.responses is None or not (args.responses / "responses.jsonl").is_file()):
        parser.error("--responses must be a `generate` output directory.")
    metadata = {"mode": args.mode, "adapter": args.adapter, "limit": args.limit,
                "data": str(args.data or args.responses)}
    if args.mode == "judge":
        responses = read_rows(args.responses / "responses.jsonl", args.limit)
        judged = run_judge(responses, args.judge_model, min(args.workers, 8))
        out.mkdir(parents=True)
        write_jsonl(out / "judged.jsonl", judged)
        save(out / "metadata.json", {**metadata, "judge_model": args.judge_model,
                                     "source": json.loads((args.responses / "metadata.json").read_text())})
        save(out / "scores.json", judge_scores(judged))
        print(json.dumps(judge_scores(judged), indent=2))
        return
    try:
        server = Server(args.base_url, args.served_model, args.adapter)
    except (ValueError, httpx.HTTPError) as error:
        parser.error(str(error))
    rows = read_rows(args.data, args.limit)
    out.mkdir(parents=True)
    metadata.update(model=server.model, base_url=args.base_url)
    if args.mode == "mcq":
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(BENCH_MODEL["id"], revision=BENCH_MODEL["revision"])
        results = run_mcq(rows, server, lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
                          args.workers)
        write_jsonl(out / "results.jsonl", results)
        save(out / "metadata.json", {**metadata, "prompt": "process_docs_qwen2_5"})
        save(out / "scores.json", mcq_metrics(results))
        print(json.dumps(mcq_metrics(results), indent=2))
    else:
        responses = run_generate(rows, server, args.workers)
        write_jsonl(out / "responses.jsonl", responses)
        save(out / "metadata.json", {**metadata, "decoding": "greedy", "template": "prompts/qwen_chat.jinja"})
        print(json.dumps(dict(Counter("call" if r["model_response"].startswith('{"name"') else "text"
                                      for r in responses))))


if __name__ == "__main__":
    main()
