# Milestone C — Low-Level Design: Dynamic Cluster

Parent doc: [`ARCHITECTURE.md`](../../ARCHITECTURE.md). Builds directly on Milestone B's
`telemetry` / `device-agent` / `coordinator` crates — this is additive, not a rewrite.

## 1. Scope, stated honestly

**What Milestone C claims:** the cluster reconfigures itself automatically — no human
runs `plan_placement` and restarts `llama-server` by hand — when a device joins,
leaves, or its resources change enough to matter.

**What Milestone C does NOT claim:** that an in-flight generation survives a
reconfiguration. Without KV migration (Milestone D), a placement change still requires
killing and relaunching `llama-server`, which means any in-flight generation is lost.
C's reconfiguration is **generation-atomic**: it only ever happens between requests,
never mid-generation. Making a reconfiguration mid-generation non-destructive is
specifically what Milestone D adds. Conflating the two claims is the single easiest way
to overstate this project's results — don't.

## 2. Gossip layer

**Crate:** [`chitchat`](https://github.com/quickwit-oss/chitchat) (scuttlebutt gossip +
phi-accrual failure detector). Chosen over `foca`/raw SWIM because phi-accrual adapts
its suspicion threshold to observed heartbeat jitter — exactly the behavior you want on
a home WiFi link where RTT variance, not just packet loss, is the normal case.

**Node identity:** each `device-agent` instance runs a chitchat node keyed by
`device_id` (same string used everywhere else in the system — `DeviceProfile`,
`DeviceConfig`, `LayerAssignment` all share this key, so no translation layer is needed
between gossip and the existing Milestone B types).

**Gossip KV schema** (chitchat propagates arbitrary key-value pairs per node):

| Key | Value | Update cadence |
|---|---|---|
| `profile` | JSON-encoded `telemetry::DeviceProfile` | every 2s |
| `generation_active` | `"true"` / `"false"` | on state change (event-driven, not polled) |

The `generation_active` flag is what makes the "generation-atomic" guarantee in §1
possible: the Coordinator can see, cluster-wide, whether it's safe to reconfigure right
now without a side-channel RPC call.

**Phi-accrual threshold:** start at `phi = 8` (the Cassandra/Akka default, well-tested
in production gossip systems). Do not hand-tune this before Milestone B has produced
real home-WiFi RTT samples — tuning a failure detector against imagined network
conditions is exactly the kind of premature precision this project should avoid.
Record actual RTT variance during Milestone B benchmarking specifically so this number
has real data behind it before C is implemented.

## 3. `ClusterView` (Coordinator-side)

```rust
pub struct ClusterView {
    // device_id -> (last known profile, last-used-for-placement profile, last_seen)
    nodes: HashMap<String, NodeState>,
}

pub struct NodeState {
    pub current_profile: DeviceProfile,
    pub profile_at_last_placement: DeviceProfile,
    pub last_seen: Instant,
    pub suspected_dead: bool, // driven by chitchat's failure detector callback
}
```

This replaces Milestone B's one-shot `fetch_all_profiles` call with a live,
continuously-updated view fed by gossip callbacks instead of polling.

## 4. Reschedule trigger — decision function

A pure function, unit-testable exactly like `plan_placement` in Milestone B:

```rust
pub enum RescheduleReason {
    MembershipChanged { joined: Vec<String>, left: Vec<String> },
    ComputeScoreDrift { device_id: String, old: f64, new: f64 },
    BatteryCrossedThreshold { device_id: String, percent: f64 },
}

pub fn should_reschedule(
    view: &ClusterView,
    last_reschedule: Instant,
    debounce: Duration,
) -> Option<RescheduleReason> { /* ... */ }
```

**Concrete trigger conditions** (each independently testable with synthetic
`ClusterView` fixtures, following the same pattern as Milestone B's
`plan_placement` tests):

1. **Membership change** — any node joins or is marked `suspected_dead` since the last
   placement. Always reschedules; no debounce (a dead node's work needs reassigning
   immediately, not after a cooldown).
2. **Compute-score drift** — `|new_score - profile_at_last_placement.compute_score| / profile_at_last_placement.compute_score > 0.20`.
   Debounced (see below) — a device's score jittering ±5% every telemetry tick must not
   cause constant reconfiguration.
3. **Battery threshold crossing** — `battery_percent` crosses *below* 15%, edge-triggered
   (fires once on the transition, not every tick while below 15% — check
   `old_percent >= 15.0 && new_percent < 15.0`, not just `new_percent < 15.0`, or a
   flapping battery reading at 14.9%/15.1% would trigger repeatedly).

**Debounce:** minimum 30 seconds between score-drift-triggered reschedules (membership
changes bypass debounce, per #1). This number is a starting default to be revised once
real telemetry noise is measured — flag it, don't treat it as final.

## 5. Reconfigure cycle — state machine

```text
                    ┌──────────────────────────────────────────┐
                    │                                            │
                    ▼                                            │
   ┌─────────┐  reschedule   ┌───────────────────┐  generation  │
   │ Serving │──trigger────▶│ ReconfigurePending  │──active?─────┘
   └─────────┘   fires       └───────────────────┘  (wait)
        ▲                              │
        │                    generation_active == false
        │                              ▼
        │                    ┌───────────────────┐
        │                    │  Reconfiguring     │
        │                    │  1. kill old        │
        │                    │  llama-server        │
        │                    │  2. plan_placement() │
        │                    │  3. build args      │
        │                    │  4. launch new       │
        │                    └───────────────────┘
        └──────────────────────────────┘
                relaunched, serving
```

`ReconfigurePending` is the state that enforces generation-atomicity: it polls the
gossip-fed `generation_active` flag and only proceeds to `Reconfiguring` once it reads
`false`. If a request is actively generating when a MembershipChanged trigger fires
(e.g. the device serving decode just died), the Coordinator does not wait indefinitely —
it aborts the in-flight request with a clear error to the caller after a
configurable timeout (default 5s), then proceeds to reconfigure. This is the honest
Milestone C behavior: a dead node mid-generation costs you that one request. Milestone D
exists specifically to remove this cost.

## 6. Metrics to collect (feeds the eventual paper/writeup)

- **Reschedule decision latency** — time from trigger condition becoming true to
  `plan_placement()` returning (should be microseconds; this isn't the expensive part).
- **Reconfiguration wall-clock time** — time from killing the old `llama-server` to the
  new one accepting requests. This is the number that determines whether Milestone D's
  investment is worth it (if this is already sub-second, the case for D is about
  request-continuity, not speed; if it's many seconds, D's value proposition sharpens).
- **Aborted-request rate** — how often a MembershipChanged trigger fires while
  `generation_active == true`, i.e., how often Milestone C's "cost you that one
  request" case actually happens under real conditions (run with realistic request
  rates, not just one request at a time, to get a meaningful number here).

## 7. What this milestone deliberately leaves open for D

- Reconfiguration always costs a full `llama-server` restart (no partial/incremental
  topology change).
- Any in-flight generation during a MembershipChanged trigger is aborted, not migrated.
- No phase-aware (prefill vs. decode) placement — one topology serves both phases.

These three gaps, in order, are exactly what Milestone D's design
(`milestone-d-phase-aware-migration.md`) closes.
