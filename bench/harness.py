import argparse
import json
import time
import sys

import requests

from metrics import compute_metrics


def stream_completion(base_url: str, prompt: str, n_predict: int) -> list[tuple[float, str]]:
    start = time.monotonic()
    events: list[tuple[float, str]] = []
    resp = requests.post(
        f"{base_url}/completion",
        json={"prompt": prompt, "n_predict": n_predict, "stream": True},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = json.loads(line[len("data: "):])
        token = payload.get("content", "")
        if token == "":
            continue
        events.append((time.monotonic() - start, token))
        if payload.get("stop"):
            break
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt", default="Explain how a diesel engine works in three sentences.")
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--label", required=True, help="e.g. 'milestone-a-single-device'")
    args = parser.parse_args()

    events = stream_completion(args.base_url, args.prompt, args.n_predict)
    if not events:
        print("No tokens received — is llama-server running?", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(events)
    metrics["label"] = args.label
    metrics["n_predict"] = args.n_predict

    print(json.dumps(metrics, indent=2))
    with open(args.out, "a") as f:
        f.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()
