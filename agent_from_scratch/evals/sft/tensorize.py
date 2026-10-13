"""Turn exported SFT examples into token rows whose loss falls on the completion only."""

from ...llm import QWEN_END

IGNORE = -100  # The label a causal LM's loss skips.


def encode_example(example: dict, tokenize, *, max_tokens: int | None = None) -> dict:
    """One training row: the whole text as tokens, with labels on the completion only.

    Args:
        example: one line of examples.jsonl, with `id`, `prompt` and `completion`.
        tokenize: text -> (ids, offsets); offsets[i] is the (start, end) character span of token i.
        max_tokens: refuse a longer row. Nothing is ever truncated.

    Returns:
        {"id", "input_ids", "labels"}; labels[i] is IGNORE for a prompt token, else input_ids[i].
    """
    name, prompt = example["id"], example["prompt"]
    # One joint tokenization: the halves tokenized apart could merge differently at the seam.
    text = prompt + example["completion"]
    ids, offsets = tokenize(text)
    if not ids or len(ids) != len(offsets):
        raise ValueError(f"Example {name}: the tokenizer returned {len(ids)} ids and "
                         f"{len(offsets)} offsets.")
    if max_tokens is not None and len(ids) > max_tokens:
        raise ValueError(f"Example {name}: {len(ids)} tokens exceed max_tokens={max_tokens}; "
                         "a row is never truncated.")
    boundary, labels = len(prompt), []
    for token, (start, end) in zip(ids, offsets):
        if start < boundary < end:
            raise ValueError(f"Example {name}: a token spans the prompt/completion boundary, "
                             "so it is neither context nor target.")
        labels.append(IGNORE if start < boundary else token)
    # The stop token must be a target, or the adapter never learns to stop.
    start, end = offsets[-1]
    if start < boundary or text[start:end] != QWEN_END:
        raise ValueError(f"Example {name}: the completion must end with one {QWEN_END} token.")
    return {"id": name, "input_ids": list(ids), "labels": labels}


def select_examples(examples: list[dict], *, max_per_skeleton: int | None = None,
                    limit: int | None = None) -> list[dict]:
    """A subset in file order: whole trajectories up to `max_per_skeleton` examples per skeleton,
    then the first `limit` examples."""
    def trajectory(example: dict) -> tuple:
        provenance = example["provenance"]
        return provenance["skeleton"], provenance["run_dir"], provenance["task_id"]

    kept = list(examples)
    if max_per_skeleton is not None:
        sizes: dict[tuple, int] = {}
        for example in examples:
            sizes[trajectory(example)] = sizes.get(trajectory(example), 0) + 1
        used: dict[str, int] = {}
        chosen = set()
        # A trajectory that does not fit is skipped; a later, shorter one may still fit.
        for key, size in sizes.items():
            if used.get(key[0], 0) + size <= max_per_skeleton:
                used[key[0]] = used.get(key[0], 0) + size
                chosen.add(key)
        kept = [example for example in examples if trajectory(example) in chosen]
    return kept[:limit]
