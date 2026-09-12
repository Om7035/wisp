from metrics import compute_metrics

def test_compute_metrics_basic():
    # (timestamp_seconds, token_text) — first event is time-to-first-token
    events = [
        (0.20, "Hello"),
        (0.25, " world"),
        (0.30, "!"),
    ]
    result = compute_metrics(events)
    assert result["ttft_seconds"] == 0.20
    assert result["total_tokens"] == 3
    assert result["total_seconds"] == 0.30
    assert abs(result["tokens_per_second"] - (3 / 0.30)) < 1e-9

def test_compute_metrics_single_token():
    events = [(0.15, "Hi")]
    result = compute_metrics(events)
    assert result["ttft_seconds"] == 0.15
    assert result["total_tokens"] == 1
    assert result["tokens_per_second"] == 1 / 0.15
