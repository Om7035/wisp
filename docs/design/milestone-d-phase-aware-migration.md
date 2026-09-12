# Milestone D — Low-Level Design: Phase-Aware Placement + KV Migration

Parent doc: [`ARCHITECTURE.md`](../../ARCHITECTURE.md). Depends on Milestone C's gossip
layer, `ClusterView`, and reschedule trigger — this milestone changes what happens
*inside* `Reconfiguring` (see Milestone C's state machine, §5), it doesn't replace it.

## 1. Scope, stated honestly

**What Milestone D claims:** a single generation request survives a device failing,
degrading, or being reassigned mid-generation — via a fast checkpoint/restore cycle, not
literal zero-downtime compute-graph migration (ADR-2 in `ARCHITECTURE.md` already rules
that out as out-of-scope). We also split prefill and decode across separate device
pools, since they have different resource profiles (prefill: compute-bound; decode:
memory-bandwidth-bound).

**What Milestone D does NOT claim:** we are not modifying ggml internals. Every
mechanism below is built from llama.cpp's **existing, documented server API** — this
milestone's job is orchestrating that API well under failure, not extending llama.cpp
itself.

## 2. The leverage point: llama.cpp slot save/restore

llama-server, run with `--slot-save-path <dir>` and `--slots`, already exposes:

- `POST /slots/{id}?action=save` with `{"filename": "..."}` — dumps that slot's full KV
  cache + sampling/recurrent state to a binary file. Confirmed near-instant to produce
  relative to re-running the prefill that generated that state.
- `POST /slots/{id}?action=restore` with `{"filename": "..."}` — loads it back. Restore
  is confirmed dramatically faster than re-prefilling the same context from scratch.

File size: hundreds of MB to multiple GB depending on context length — this is *why*
compression on the transfer path (§5) is not optional the way it might look on paper;
moving a multi-GB file over home WiFi is the dominant cost in this whole pipeline.

**The open question this design must resolve:** whether a slot saved under one
`--tensor-split` / device topology can be restored under a *different* one. This is
explicitly unconfirmed in the llama.cpp community as of this research (see
`ARCHITECTURE.md` §2 sources) — nobody has published a clear yes/no. We do not assume
either answer; §4 below gives a design that works either way, and the first coding task
under this milestone (before anything else) is a small, isolated experiment that settles
this empirically for our target model/quantization.

## 3. Phase-aware placement as two llama-server instances

llama-server does not natively let you use a different device topology for prefill vs.
decode within one running instance — topology is fixed at process launch. Rather than
patch llama.cpp to add this (out of scope per ADR-3), Milestone D runs **two separate
llama-server processes**, each with its own `--tensor-split` over its own device subset:

```text
   Request arrives
        │
        ▼
┌─────────────────────┐   slot save    ┌─────────────────────┐
│  Instance P          │───────────────▶│  Instance D          │
│  (prefill pool:       │   + restore    │  (decode pool:        │
│   compute-heavy       │                │   memory-bandwidth-   │
│   devices)             │                │   heavy devices)       │
│  Ingests prompt,       │                │  Runs the actual       │
│  produces first token, │                │  autoregressive decode │
│  saves slot            │                │  loop from the restored│
└─────────────────────┘                │  slot                  │
                                          └─────────────────────┘
```

This is a direct, deliberate reuse of the exact same save/restore primitive for two
different purposes: moving prefill→decode is the *normal* path (happens on every
request), and moving decode→decode-on-a-different-device-set after a mid-generation
failure (§4) is the *degraded* path. One mechanism, two call sites — not two separate
systems to build and test.

## 4. Migration on mid-generation failure

Extends Milestone C's `Reconfiguring` state (§5 of the Milestone C doc) for the case
where `generation_active == true` when a trigger fires — instead of aborting the
request (Milestone C's behavior), Milestone D migrates it:

```text
Reconfiguring (generation_active == true)
  1. Save current decode instance's slot to disk
     POST /slots/{id}?action=save
  2. Compress the slot file (zstd, see §5)
  3. Transfer compressed file to the new decode device set's coordinator-visible path
  4. Decompress
  5. Launch new decode instance with new --tensor-split over the surviving/new devices
  6. Restore slot
     POST /slots/{id}?action=restore
  7. IF restore fails (cross-topology incompatibility, see §2's open question):
       Fallback: restore the slot onto a SINGLE-DEVICE staging instance first
       (same-topology restore is not in question — only cross-topology is).
       Treat the restored context as a fresh prompt; re-run prefill against it
       on the new target topology via Instance P. This costs a re-prefill of the
       accumulated context (slower than a true restore) but never requires solving
       cross-topology binary compatibility at all, and is strictly cheaper than the
       Milestone C fallback of losing the request entirely.
  8. Resume decode loop, continue streaming tokens to the original client connection
```

Step 7's fallback is the single most important honesty mechanism in this whole
milestone: it means **the demo's "graceful degradation" claim holds even in the worst
case where cross-topology restore turns out not to work.** Never present the happy path
(direct cross-topology restore) as guaranteed in any README, blog post, or demo caption
— present it as "fast path when available, always-correct fallback otherwise."

## 5. Compression pipeline

**Baseline (build this first):** wrap the slot file in `zstd` (crate:
[`zstd`](https://crates.io/crates/zstd), level 3 — favor speed over ratio, since the
whole point is reducing wall-clock migration time, not disk usage). Measure actual
compression ratio and throughput on a real saved slot file before deciding anything
further.

**Upgrade path (only if the baseline's numbers justify it):** port the exponent-codebook
+ sparse-escape-stream approach from
[SplitZip](https://github.com/Intelligent-Microsystems-Lab/SplitZip) — it targets
exactly this data shape (BF16/FP16 KV activation tensors) and reports ~600 GB/s
compression throughput on GPU, dramatically faster than general-purpose `zstd`. Do not
start here; `zstd` is almost certainly "good enough" for a first working migration, and
SplitZip's GPU-kernel-level implementation is a substantial engineering investment of
its own that should only happen once the pipeline works end-to-end and profiling shows
compression time, not network time, is the bottleneck.

## 6. Fault injection (how the demo actually gets built)

The benchmark harness (`bench/harness.py`, extended) drives three concrete injected
failures, matching the "headline feature" framing in `ARCHITECTURE.md`:

1. **Kill a decode-pool device's `device-agent` process mid-generation** — simulates a
   hard crash. Expect: gossip failure detector fires, migration cycle runs, generation
   resumes on the surviving devices.
2. **Throttle a device's network link mid-generation** (Linux: `tc qdisc add dev <iface>
   root netem rate 1mbit`; document the equivalent for macOS/Windows test devices, or
   restrict the fault-injection device set to Linux boxes if cross-platform `tc` parity
   isn't worth building). Expect: compute-score/bandwidth drift trigger fires,
   proactive rebalancing runs *before* the link fully collapses.
3. **Mock a battery drain** — since real battery drain is slow, `device-agent` supports
   a `--fake-battery-percent <n>` override flag for demo/test purposes, decrementing on a
   timer instead of reading real hardware. Expect: battery-threshold trigger fires,
   that device is proactively evicted from the decode pool before it dies.

Each of these three becomes one recorded demo clip. This is the actual "watch it keep
generating while I kill the WiFi on the phone mid-sentence" moment from earlier
discussion — plan to record it once #2 above is working, since it's the most visually
convincing of the three.

## 7. Metrics

- **Recovery time** — wall-clock from fault injection to the first post-migration token
  being emitted.
- **Network bytes transferred per migration** — raw slot file size vs. compressed size
  actually sent.
- **Token-level correctness** — the migrated generation's output, given a fixed seed,
  compared token-for-token against a non-migrated control run of the same prompt on a
  single stable topology. This is the check that migration didn't silently corrupt
  state — treat any mismatch as a release-blocking bug, not a rounding-error footnote.
- **Fallback-path trigger rate** — how often step 7's single-device-staging fallback is
  actually needed vs. the direct cross-topology restore path, once you have enough real
  runs to say.

## 8. What ships as the actual deliverable

A short technical writeup (`docs/milestone-d-results.md`, written after building this,
not before) reporting the four metrics in §7 across the three fault-injection scenarios
in §6, plus the recorded demo clips. This is the artifact that goes on Twitter/Reddit and
in front of a GNC/ML-systems recruiter — not the code alone.
