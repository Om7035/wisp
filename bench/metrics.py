def compute_metrics(events: list[tuple[float, str]]) -> dict:
    """events: list of (timestamp_seconds_since_request_start, token_text),
    ordered by arrival. Must be non-empty."""
    if not events:
        raise ValueError("compute_metrics requires at least one event")

    ttft_seconds = events[0][0]
    total_tokens = len(events)
    total_seconds = events[-1][0]

    tokens_per_second = total_tokens / total_seconds

    return {
        "ttft_seconds": ttft_seconds,
        "total_tokens": total_tokens,
        "total_seconds": total_seconds,
        "tokens_per_second": tokens_per_second,
    }
