# Wisp — Architecture

*(Working name. Rename before going public — check GitHub/crates.io/PyPI availability first.)*

## Documentation map

| Doc | Covers |
|---|---|
| `ARCHITECTURE.md` (this file) | Thesis, prior-art gaps, scope limits, components, failure domains, ADRs, milestone ladder, tech stack, risks |
| [`docs/design/milestone-c-dynamic-scheduler.md`](docs/design/milestone-c-dynamic-scheduler.md) | Low-level design for Milestone C: gossip schema, `ClusterView`, reschedule trigger conditions, reconfigure state machine, metrics |
| [`docs/design/milestone-d-phase-aware-migration.md`](docs/design/milestone-d-phase-aware-migration.md) | Low-level design for Milestone D: llama.cpp slot save/restore as the migration primitive, phase-aware two-instance design, cross-topology-restore open question + fallback, fault-injection plan, metrics |
| [`docs/superpowers/plans/2026-09-12-milestone-a-b-baseline-cluster.md`](docs/superpowers/plans/2026-09-12-milestone-a-b-baseline-cluster.md) | Bite-sized, TDD, zero-placeholder executable plan for Milestones A+B. C and D get their own bite-sized plans after A/B produce real hardware numbers — see that plan's own note on why. |
| `docs/milestone-b-results.md`, `docs/milestone-d-results.md` | Written *after* building each milestone — real measured numbers, not projections. Don't create these ahead of time. |

**Process convention:** design and review (this doc, the design docs, plan review) is
done by a large reasoning model (Sonnet/Opus). Actual code-writing, once a plan is
approved and we move to subagent-driven execution, is delegated to Haiku subagents per
task — the plans in this repo are written detailed enough (exact file paths, exact
code, exact expected test output) that a smaller model can execute them faithfully
without needing to make design judgment calls itself. If a Haiku subagent hits a
decision the plan didn't already make for it, that's a signal the plan was under-specified
— fix the plan, don't let the subagent improvise.

## 1. Thesis

> Can an LLM inference cluster made from unreliable, heterogeneous consumer devices
> continuously re-optimize itself as compute, memory, battery, and network conditions
> change — without restarting generation?

Wisp is an orchestration layer over existing distributed-inference execution engines
(starting with llama.cpp's RPC backend) that adds three things none of the current
tools do together:

1. **Online, not one-shot, scheduling** — re-decides device/layer placement while a
   cluster is running, not just at startup.
2. **Phase-aware placement** — prefill (compute-bound) and decode (memory-bandwidth-bound)
   can be assigned to different device subsets, not treated as one workload.
3. **Graceful degradation as a first-class feature** — a device going flaky (battery
   drain, WiFi collapse, thermal throttle, outright disconnect) triggers a live
   reconfiguration, not a crash or a full restart of the request.

## 2. Prior art — what we build on, and the specific gap in each

| Project | What it gives us | What it doesn't do (our gap) |
|---|---|---|
| [ggml-org/llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | Production-grade tensor-op transport over TCP; pools GPU/CPU/mem across machines; `--tensor-split` to control per-device share | Split is fixed at model-load time. No runtime reconfiguration API. Confirmed via upstream docs/issues (e.g. [#13083](https://github.com/ggml-org/llama.cpp/issues/13083), [#21006](https://github.com/ggml-org/llama.cpp/issues/21006) — RPC multi-machine balancing is itself an open pain point upstream). |
| [Lizonghang/prima.cpp](https://github.com/Lizonghang/prima.cpp) (ICLR 2026) | Halda: a heterogeneity-aware *static* scheduler (co-optimizes CPU/GPU/RAM/disk assignment); pipelined-ring parallelism | One-shot profiling + placement at startup. No online re-scheduling, no phase-aware (prefill/decode) split, no live migration. |
| [Intelligent-Microsystems-Lab/SplitZip](https://github.com/Intelligent-Microsystems-Lab/SplitZip) | GPU-friendly *lossless* KV-cache compressor (~600GB/s compress, exponent-codebook + sparse escape stream) | Designed for datacenter disaggregated serving (RDMA-class links, prefill/decode already separated by the caller). We adapt the codec idea for high-latency, lossy home WiFi and for our own migration trigger logic. |
| [hpdps-group/KVServe](https://github.com/hpdps-group/KVServe) | Service-aware, *adaptive* compression that reacts to bandwidth/SLO at runtime | Same datacenter-disaggregation assumption; no notion of a device disappearing mid-transfer or of consumer-grade unreliability. |
| NVIDIA PAIR (Sept 2026) | Proves the *market* believes in "pool idle home machines" | Operates at the task/agent level — routes whole independent subtasks to whichever idle full machine can run them. Never splits a single model forward-pass across devices that individually can't run it, and does no mid-generation reconfiguration. This is the difference we lead with in the README. |
| [Microsoft: Online Scheduling for LLM Inference w/ KV Cache Constraints](https://arxiv.org/abs/2502.07115) | Proves "online KV-constrained scheduling" is a real, studied problem class | Single reliable datacenter cluster, no heterogeneity or unreliability dimension. |

**Our claim, stated defensibly:** we investigate an underexplored combination —
heterogeneous consumer-device inference + online phase-aware scheduling +
network-constrained KV-state migration under real device unreliability — not any one
of those three in isolation.

## 3. What we are explicitly NOT building

- **No custom tensor kernels / GEMM.** llama.cpp's ggml backend does this; we never
  touch compute internals.
- **No consensus protocol (Raft/Paxos).** This is a single-owner personal cluster, not
  a multi-tenant HA system. One designated **coordinator device is authoritative** for
  scheduling decisions. Gossip is used only for failure *detection* and telemetry
  dissemination, never for agreeing on cluster state. (ADR-1, below.)
- **No true zero-downtime hot-swap of an in-flight ggml compute graph.** ggml has no
  API for this and building one is its own multi-month project. MVP uses
  **checkpoint → reconfigure → resume**, and the research contribution is making that
  cycle fast (sub-second target), not claiming literal zero downtime. (ADR-2, below.)
- **No custom gossip/SWIM implementation.** Use an existing crate (`chitchat`, or `foca`
  if we need transport-agnostic control).
- **No cloud/datacenter story.** Home network only; no autoscaling, no multi-region.
- **No concurrent multi-request scheduling in the MVP.** One active generation stream
  at a time. Request queuing is a stretch goal after Milestone D, not before.
- **No from-scratch KV compression codec on day one.** Start with `zstd` on raw KV
  tensors as the boring baseline; only port/reimplement the SplitZip exponent-codebook
  approach once the migration pipeline works end-to-end and we can measure whether it
  matters.

## 4. Components

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Coordinator (Rust binary, runs on one designated device)            │
│                                                                       │
│  ┌────────────────┐   ┌───────────────┐   ┌─────────────────────┐   │
│  │ Device Registry │──▶│  Scheduler     │──▶│  Cluster Launcher    │   │
│  │ (gossip-fed,    │   │ (placement +   │   │ (builds/executes     │   │
│  │  telemetry view)│   │  reschedule    │   │  llama-server/rpc-   │   │
│  │                 │   │  trigger logic)│   │  server invocations) │   │
│  └────────────────┘   └───────────────┘   └─────────────────────┘   │
│          ▲                                          │                │
│          │ gossip (chitchat)                        │ spawns/kills   │
└──────────┼──────────────────────────────────────────┼────────────────┘
           │                                          ▼
   ┌───────┴────────┐                        ┌──────────────────┐
   │ Device Agent    │  (one per worker node) │  rpc-server        │
   │ (Rust binary)   │◀──────supervises───────│  (llama.cpp, RPC)  │
   │ - telemetry     │                        └──────────────────┘
   │ - gossip peer   │
   │ - subprocess    │
   │   supervisor    │
   └────────────────┘
```

**Coordinator** — owns the scheduling decision. Watches the device registry (fed by
gossip), runs the placement algorithm, and when a reschedule is warranted, drives the
checkpoint → relaunch → resume cycle via the Cluster Launcher.

**Device Agent** — one per worker machine (including, optionally, the coordinator's own
machine). Reports telemetry (CPU/mem/disk/battery/network), participates in gossip for
liveness, and supervises the local `rpc-server` process (start/stop/restart on command).

**Cluster Launcher** — pure translation layer: `PlacementPlan → argv` for
`rpc-server`/`llama-server`, plus the checkpoint/resume choreography.

**KV Migration** (Milestone D) — a separate service that serializes KV cache state,
compresses it (zstd baseline → SplitZip-style codec later), and streams it to the new
device set as part of a reschedule.

## 5. Failure domains

| Domain | Detection | Response |
|---|---|---|
| Hard crash / process death of a worker | gossip phi-accrual failure detector (bounded suspicion window) marks node dead | Scheduler excludes it, recomputes placement, triggers checkpoint/reload excluding the dead node |
| Graceful degradation (battery <15%, bandwidth dropping, thermal throttle) | **Self-reported** by the Device Agent proactively, not just inferred from timeout — this is the actual novel bit, not just failure detection | Scheduler proactively rebalances *before* hard failure — reduce that device's share or evict it cleanly |
| Network partition between coordinator and a subset of devices | Suspicion timeout, same as crash | Treated identically to a crash: coordinator is authoritative, so we accept losing a partitioned device's work rather than attempting split-brain reconciliation (ADR-1) |
| Coordinator itself dies | Out of scope for MVP — single point of failure by design (ADR-1); documented, not solved |

## 6. Key architecture decisions (ADRs, abbreviated)

**ADR-1: Single authoritative coordinator, no consensus protocol.**
Alternative considered: leaderless gossip-based agreement (like a mini Raft) so any
device could become coordinator. Rejected — this is a personal, single-owner cluster;
consensus overhead buys availability guarantees nobody needs here and multiplies the
project's real complexity for zero demo value. Revisit only if this becomes a
multi-tenant product.

**ADR-2: Checkpoint-reload, not live compute-graph migration.**
Alternative considered: patching ggml to support hot tensor-placement swap mid-graph.
Rejected for MVP — that's a separate, deep C++ systems project with its own multi-month
timeline and uncertain payoff. Our contribution is making the checkpoint-pause-relaunch-
resume cycle fast and automatic; we report the actual pause duration honestly as a
metric rather than claiming zero-downtime.

**ADR-3: Build on ggml RPC transport, don't reinvent it.**
Alternative considered: custom tensor-op transport protocol. Rejected — the transport
already exists, is maintained, and reinventing it would be a fourth project bolted onto
this one (echoing the earlier warning against "Rust coordinator + Python inference +
C++ + Go agents + custom protocol" — pick one hard problem, not five).

**ADR-4: Rust for control plane, unmodified llama.cpp (C/C++) for execution, Python for
evaluation only — no Go.**
Keeps the language surface to three, each doing the thing it's actually good at:
control-plane correctness and concurrency (Rust), tensor execution (existing C++, don't
touch it), measurement/plotting/fault-injection scripting (Python).

## 7. Milestone ladder

Each milestone is a complete, demoable, independently-testable system — not a partial
slice of a bigger thing.

| Milestone | Deliverable | New components | Key metric(s) |
|---|---|---|---|
| **A — Single-device baseline** | llama.cpp running locally, benchmark harness exists | `bench/harness.py` | tokens/sec, TTFT (baseline numbers everything else is compared against) |
| **B — Static heterogeneous cluster** | 2-3 real owned devices, one-time profiling → computed `--tensor-split`, beats naive memory-proportional default | `telemetry` crate, `device-agent`, `coordinator` (static registry + placement + launcher) | tokens/sec vs. naive split, on real hardware |
| **C — Dynamic cluster** ([low-level design](docs/design/milestone-c-dynamic-scheduler.md)) | Same cluster, but gossip membership + live telemetry + automatic reschedule on threshold (no manual restart needed) | gossip integration, reschedule trigger logic, checkpoint/relaunch cycle | reschedule decision latency, reconfiguration wall-clock time, aborted-request rate |
| **D — Phase-aware + KV migration** ([low-level design](docs/design/milestone-d-phase-aware-migration.md)) | Prefill and decode assigned to *different* device subsets; injected failure (kill WiFi/battery/process) triggers live migration; generation continues | KV migration service (built on llama.cpp's existing slot save/restore API), phase-split scheduler | recovery time after injected failure, network bytes/token during migration, token-level output correctness vs. control run |

Low-level design for C and D is done (linked above) — algorithms, protocols, state
machines, and the concrete llama.cpp APIs each leans on are all specified. What's
deliberately *not* done yet is the **bite-sized TDD task breakdown** for C and D (the
thing Task-1-through-8 is for A/B) — that still waits on real numbers from A/B (e.g.,
actual home-WiFi RTT variance for tuning the phi-accrual failure detector, actual
reconfiguration wall-clock time for deciding how urgent D's investment is). Turning a
finished low-level design into bite-sized steps is quick once those numbers exist;
guessing the numbers to write the steps early would just mean rewriting them anyway.

## 8. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Control plane (coordinator, device-agent) | Rust | Correctness + concurrency for the scheduler; matches the ecosystem of the crates we're leaning on (chitchat, foca, systemstat) |
| Gossip / failure detection | [`chitchat`](https://quickwit.io/blog/chitchat) (scuttlebutt + phi-accrual) | Phi-accrual handles flaky, variable-latency home WiFi better than fixed SWIM timeouts; battle-tested in Quickwit |
| Telemetry | `systemstat` + `battery` crates | Cross-platform (Linux/macOS/Windows) CPU/mem/disk/network/battery in one place |
| Execution substrate | llama.cpp (`GGML_RPC=ON`), unmodified where possible | Don't reinvent tensor transport or kernels |
| Control-plane HTTP | `axum` | Device Agent telemetry endpoint, Coordinator's OpenAI-compatible-ish API |
| KV compression | `zstd` (baseline) → SplitZip-style codec (later, if justified by measurement) | Boring baseline first, fancy only if data says it matters |
| Evaluation harness | Python (`requests`, `pandas`, `matplotlib`) | Fault injection scripting + plots, not a systems-programming job |

## 9. Biggest risks

1. **Checkpoint/reload latency might be too slow to demo well.** Mitigation: measure
   honestly starting at Milestone C; if it's seconds instead of sub-second, that's still
   a publishable finding, just reframe the pitch around "automatic, no manual
   intervention" rather than "seamless."
2. **You may not own 3+ genuinely heterogeneous devices.** Mitigation: a cheap SBC
   (Raspberry Pi) plus your own laptop/desktop is enough heterogeneity to be real; VMs
   with artificially capped resources are an acceptable fallback for extra "devices."
3. **llama.cpp RPC version lockstep requirement** (every node needs the same build) means
   your fault-injection story must not include "device runs old software" — document
   this as an assumed invariant, not a bug to fix.
