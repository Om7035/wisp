# Wisp

Adaptive, failure-tolerant LLM inference orchestration across heterogeneous,
unreliable consumer devices. See ARCHITECTURE.md for the full design.

## Status
Milestone A (single-device baseline) complete.

## Building
See ARCHITECTURE.md §7 for the milestone ladder. To reproduce the baseline:
    cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DGGML_RPC=ON
    cmake --build vendor/llama.cpp/build -j
    pip install -r bench/requirements.txt
    python bench/harness.py --label milestone-a-single-device
