"""Keep the prompt inside the model's context window by dropping the oldest turns (sliding window)."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb for English; real tokenizers differ per model


def trim_history(messages: list[dict], max_tokens: int) -> list[dict]:
    kept, total = [], 0
    for m in reversed(messages):
        t = estimate_tokens(m["content"])
        if kept and total + t > max_tokens:
            break
        kept.append(m)
        total += t
    kept.reverse()
    while kept and kept[0]["role"] != "user":  # chat APIs expect the first turn to be from the user
        kept.pop(0)
    return kept
