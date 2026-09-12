# Milestone A+B: Baseline & Static Heterogeneous Cluster — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a single-device llama.cpp baseline running with a real benchmark harness
(Milestone A), then build the telemetry + placement + launcher pipeline that runs a
model across 2-3 real heterogeneous devices with a computed (not naive) tensor-split
(Milestone B).

**Architecture:** Rust workspace with three library/binary crates (`telemetry`,
`device-agent`, `coordinator`) sitting on top of an unmodified llama.cpp (built with
`GGML_RPC=ON`) as a git submodule. Python-only for the benchmark harness. Full design
rationale lives in `ARCHITECTURE.md` at the repo root — read that first if anything
here seems under-justified.

**Tech Stack:** Rust 1.80+ (workspace), `axum` 0.7, `tokio` 1.x, `reqwest` 0.12, `serde`
+ `serde_json` + `toml`, `systemstat` 0.2, `battery` 0.7, Python 3.11+ (`requests`,
`pytest`, `pandas`).

## Global Constraints

- Every machine in the cluster must run the **exact same llama.cpp build/commit** —
  the GGML RPC wire format breaks across versions (confirmed upstream behavior).
  Pin the submodule commit; never `git pull` it casually.
- No unit test may require a live network call, a live `rpc-server` process, or real
  hardware sensors. Anything that needs those is a documented manual verification step,
  not a test.
- `bytes_per_layer` and `total_layers` for a given model are constants you look up once
  (from the GGUF metadata / model card) and pass in — this plan does not parse GGUF
  headers; that's out of scope for A/B.

---

### Task 1: llama.cpp baseline + benchmark harness core (Milestone A)

**Files:**
- Create: `vendor/llama.cpp` (git submodule, pinned commit)
- Create: `bench/metrics.py`
- Create: `bench/test_metrics.py`
- Create: `bench/harness.py`
- Create: `bench/requirements.txt`
- Create: `README.md`

**Interfaces:**
- Produces: `compute_metrics(events: list[tuple[float, str]]) -> dict` with keys
  `ttft_seconds` (float), `tokens_per_second` (float), `total_tokens` (int),
  `total_seconds` (float) — this exact function/shape is reused by every later
  benchmark comparison (Task 8, and Milestones C/D later).

- [ ] **Step 1: Add llama.cpp as a pinned submodule**

```bash
git init
git submodule add https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp
git log -1 --format=%H > ../../docs/llama.cpp.pinned-commit.txt
cd ../..
git add .gitmodules vendor/llama.cpp docs/llama.cpp.pinned-commit.txt
```

- [ ] **Step 2: Build llama.cpp with RPC support**

```bash
cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build --config Release -j
```

Expected: `vendor/llama.cpp/build/bin/llama-server` and `.../bin/rpc-server` exist
(`.exe` suffix on Windows).

- [ ] **Step 3: Write the failing test for metrics computation**

```python
# bench/test_metrics.py
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd bench && python -m pytest test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 5: Implement metrics.py**

```python
# bench/metrics.py
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd bench && python -m pytest test_metrics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Write the harness that drives a real llama-server**

```python
# bench/harness.py
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
```

```
# bench/requirements.txt
requests>=2.31
pytest>=8.0
pandas>=2.2
matplotlib>=3.8
```

- [ ] **Step 8: Manually run the baseline (Milestone A deliverable)**

```bash
./vendor/llama.cpp/build/bin/llama-server -m /path/to/your-model.gguf --port 8080 &
pip install -r bench/requirements.txt
python bench/harness.py --label milestone-a-single-device
```

Expected: JSON printed with `ttft_seconds`, `tokens_per_second` for your own hardware.
This is your baseline number — write it down, every later milestone is compared
against it.

- [ ] **Step 9: Write README and commit**

```markdown
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
```

```bash
git add README.md bench/ vendor/llama.cpp .gitmodules docs/
git commit -m "Milestone A: llama.cpp baseline + benchmark harness"
```

---

### Task 2: `telemetry` crate — DeviceProfile + deterministic compute score

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/telemetry/Cargo.toml`
- Create: `crates/telemetry/src/lib.rs`
- Create: `crates/telemetry/src/types.rs`
- Create: `crates/telemetry/src/score.rs`

**Interfaces:**
- Produces: `pub struct RawMetrics { cpu_ghz: f64, cpu_cores: u32, available_mem_bytes: u64, disk_read_mbps: f64, battery_percent: Option<f64>, on_ac_power: bool }` (all fields `pub`)
- Produces: `pub struct DeviceProfile { pub device_id: String, pub compute_score: f64, pub available_mem_bytes: u64, pub battery_percent: Option<f64> }` with `#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]`
- Produces: `pub fn compute_score(raw: &RawMetrics) -> f64` — pure function, used by Task 6's placement algorithm as the ranking weight.
- Produces: `pub fn collect_raw_metrics() -> RawMetrics` — NOT unit tested (hits real hardware); Task 3 calls it.

- [ ] **Step 1: Workspace + crate scaffolding**

```toml
# Cargo.toml (workspace root)
[workspace]
resolver = "2"
members = ["crates/telemetry", "crates/device-agent", "crates/coordinator"]
```

```toml
# crates/telemetry/Cargo.toml
[package]
name = "telemetry"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
systemstat = "0.2"
battery = "0.7"
```

- [ ] **Step 2: Write the failing test for `compute_score`**

```rust
// crates/telemetry/src/score.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawMetrics;

    #[test]
    fn faster_cpu_scores_higher_at_equal_cores() {
        let fast = RawMetrics { cpu_ghz: 4.0, cpu_cores: 8, available_mem_bytes: 16_000_000_000, disk_read_mbps: 500.0, battery_percent: None, on_ac_power: true };
        let slow = RawMetrics { cpu_ghz: 2.0, cpu_cores: 8, available_mem_bytes: 16_000_000_000, disk_read_mbps: 500.0, battery_percent: None, on_ac_power: true };
        assert!(compute_score(&fast) > compute_score(&slow));
    }

    #[test]
    fn low_battery_and_no_ac_penalizes_score() {
        let plugged_in = RawMetrics { cpu_ghz: 3.0, cpu_cores: 4, available_mem_bytes: 8_000_000_000, disk_read_mbps: 300.0, battery_percent: Some(90.0), on_ac_power: true };
        let draining = RawMetrics { cpu_ghz: 3.0, cpu_cores: 4, available_mem_bytes: 8_000_000_000, disk_read_mbps: 300.0, battery_percent: Some(8.0), on_ac_power: false };
        assert!(compute_score(&plugged_in) > compute_score(&draining));
    }

    #[test]
    fn score_is_never_negative() {
        let dying = RawMetrics { cpu_ghz: 1.0, cpu_cores: 1, available_mem_bytes: 1_000_000, disk_read_mbps: 5.0, battery_percent: Some(1.0), on_ac_power: false };
        assert!(compute_score(&dying) >= 0.0);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p telemetry score::tests`
Expected: FAIL with compile error — `compute_score` and `RawMetrics` don't exist yet.

- [ ] **Step 4: Implement `types.rs` and `score.rs`**

```rust
// crates/telemetry/src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct RawMetrics {
    pub cpu_ghz: f64,
    pub cpu_cores: u32,
    pub available_mem_bytes: u64,
    pub disk_read_mbps: f64,
    pub battery_percent: Option<f64>,
    pub on_ac_power: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub device_id: String,
    pub compute_score: f64,
    pub available_mem_bytes: u64,
    pub battery_percent: Option<f64>,
}
```

```rust
// crates/telemetry/src/score.rs
use crate::types::RawMetrics;

/// Deterministic, hardware-agnostic ranking weight for the placement algorithm.
/// Higher is better. Never negative.
///
/// Formula: raw compute capacity (cpu_ghz * cpu_cores) scaled down by a
/// [0.1, 1.0] power-health multiplier so a draining, unplugged laptop ranks
/// below an otherwise-identical plugged-in one without ever hitting zero
/// (a dying device should still get *some* work if it's the only device left).
pub fn compute_score(raw: &RawMetrics) -> f64 {
    let raw_compute = raw.cpu_ghz * raw.cpu_cores as f64;

    let power_health: f64 = match (raw.battery_percent, raw.on_ac_power) {
        (_, true) => 1.0,                              // plugged in: full weight
        (Some(pct), false) => (pct / 100.0).clamp(0.1, 1.0),
        (None, false) => 1.0,                           // desktop with no battery sensor
    };

    (raw_compute * power_health).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawMetrics;

    #[test]
    fn faster_cpu_scores_higher_at_equal_cores() {
        let fast = RawMetrics { cpu_ghz: 4.0, cpu_cores: 8, available_mem_bytes: 16_000_000_000, disk_read_mbps: 500.0, battery_percent: None, on_ac_power: true };
        let slow = RawMetrics { cpu_ghz: 2.0, cpu_cores: 8, available_mem_bytes: 16_000_000_000, disk_read_mbps: 500.0, battery_percent: None, on_ac_power: true };
        assert!(compute_score(&fast) > compute_score(&slow));
    }

    #[test]
    fn low_battery_and_no_ac_penalizes_score() {
        let plugged_in = RawMetrics { cpu_ghz: 3.0, cpu_cores: 4, available_mem_bytes: 8_000_000_000, disk_read_mbps: 300.0, battery_percent: Some(90.0), on_ac_power: true };
        let draining = RawMetrics { cpu_ghz: 3.0, cpu_cores: 4, available_mem_bytes: 8_000_000_000, disk_read_mbps: 300.0, battery_percent: Some(8.0), on_ac_power: false };
        assert!(compute_score(&plugged_in) > compute_score(&draining));
    }

    #[test]
    fn score_is_never_negative() {
        let dying = RawMetrics { cpu_ghz: 1.0, cpu_cores: 1, available_mem_bytes: 1_000_000, disk_read_mbps: 5.0, battery_percent: Some(1.0), on_ac_power: false };
        assert!(compute_score(&dying) >= 0.0);
    }
}
```

```rust
// crates/telemetry/src/lib.rs
pub mod types;
pub mod score;

pub use score::compute_score;
pub use types::{DeviceProfile, RawMetrics};

use systemstat::{Platform, System};

/// Reads real hardware sensors. NOT unit tested — see Task 8 for manual verification.
pub fn collect_raw_metrics() -> RawMetrics {
    let sys = System::new();

    let cpu_load = sys.cpu_load_aggregate().ok();
    let cpu_cores = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1);
    let mem = sys.memory().ok();
    let available_mem_bytes = mem.map(|m| m.free.as_u64()).unwrap_or(0);

    let (battery_percent, on_ac_power) = match battery::Manager::new() {
        Ok(manager) => {
            if let Ok(mut batteries) = manager.batteries() {
                if let Some(Ok(b)) = batteries.next() {
                    let pct = b.state_of_charge().value as f64 * 100.0;
                    let on_ac = b.state() != battery::State::Discharging;
                    (Some(pct), on_ac)
                } else {
                    (None, true) // no battery found -> treat as desktop on AC
                }
            } else {
                (None, true)
            }
        }
        Err(_) => (None, true),
    };

    // cpu_load_aggregate() on most platforms needs a delay+measure() call for a
    // real reading; a fixed nominal clock estimate is an acceptable placeholder
    // for Milestone B (we rank devices relatively, we don't need lab-grade GHz).
    let _ = cpu_load;
    let cpu_ghz = 3.0;

    RawMetrics {
        cpu_ghz,
        cpu_cores,
        available_mem_bytes,
        disk_read_mbps: 0.0, // not used by compute_score yet; wired in when Milestone D needs it
        battery_percent,
        on_ac_power,
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p telemetry`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/telemetry
git commit -m "Add telemetry crate: DeviceProfile, compute_score"
```

---

### Task 3: `device-agent` — telemetry HTTP endpoint

**Files:**
- Create: `crates/device-agent/Cargo.toml`
- Create: `crates/device-agent/src/main.rs`
- Create: `crates/device-agent/src/telemetry_server.rs`

**Interfaces:**
- Consumes: `telemetry::{collect_raw_metrics, compute_score, DeviceProfile, RawMetrics}` from Task 2.
- Produces: `pub fn build_router(device_id: String) -> axum::Router` — Task 5's registry (coordinator) calls `GET /telemetry` on this router's address and expects a JSON body matching `DeviceProfile`.

- [ ] **Step 1: Crate scaffolding**

```toml
# crates/device-agent/Cargo.toml
[package]
name = "device-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
telemetry = { path = "../telemetry" }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }

[dev-dependencies]
reqwest = { version = "0.12", features = ["json"] }
```

- [ ] **Step 2: Write the failing integration test**

```rust
// crates/device-agent/src/telemetry_server.rs
use axum::{routing::get, Json, Router};
use telemetry::{collect_raw_metrics, compute_score, DeviceProfile};

pub fn build_router(device_id: String) -> Router {
    Router::new().route(
        "/telemetry",
        get(move || {
            let device_id = device_id.clone();
            async move {
                let raw = collect_raw_metrics();
                let profile = DeviceProfile {
                    device_id,
                    compute_score: compute_score(&raw),
                    available_mem_bytes: raw.available_mem_bytes,
                    battery_percent: raw.battery_percent,
                };
                Json(profile)
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use telemetry::DeviceProfile;

    #[tokio::test]
    async fn telemetry_endpoint_returns_valid_device_profile() {
        let app = build_router("test-device-1".to_string());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let resp = reqwest::get(format!("http://{addr}/telemetry")).await.unwrap();
        assert_eq!(resp.status(), 200);

        let profile: DeviceProfile = resp.json().await.unwrap();
        assert_eq!(profile.device_id, "test-device-1");
        assert!(profile.compute_score >= 0.0);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p device-agent`
Expected: FAIL — `main.rs` doesn't exist yet / crate won't build without it. (If it
somehow compiles the test alone, it should still pass since Step 2's implementation is
already written — proceed to Step 4 to add `main.rs` and confirm the binary builds too.)

- [ ] **Step 4: Write main.rs**

```rust
// crates/device-agent/src/main.rs
mod telemetry_server;

use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    device_id: String,
    #[arg(long, default_value = "0.0.0.0:7100")]
    listen: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let app = telemetry_server::build_router(args.device_id.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await.unwrap();
    println!("device-agent '{}' listening on {}", args.device_id, args.listen);
    axum::serve(listener, app).await.unwrap();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p device-agent`
Expected: PASS (1 test)

- [ ] **Step 6: Commit**

```bash
git add crates/device-agent
git commit -m "Add device-agent: telemetry HTTP endpoint"
```

---

### Task 4: `device-agent` — rpc-server subprocess supervisor

**Files:**
- Create: `crates/device-agent/src/rpc_supervisor.rs`
- Modify: `crates/device-agent/src/main.rs` (wire in `--rpc-server-path` and start it)

**Interfaces:**
- Produces: `pub fn build_rpc_server_args(port: u16) -> Vec<String>` — pure, unit tested.
- Produces: `pub struct RpcSupervisor` with `pub fn spawn(binary_path: &Path, port: u16) -> std::io::Result<RpcSupervisor>` and `pub fn kill(&mut self)`. Not unit tested (spawns a real process) — manual verification in Task 8.

- [ ] **Step 1: Write the failing test for argv construction**

```rust
// crates/device-agent/src/rpc_supervisor.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_expected_rpc_server_args() {
        let args = build_rpc_server_args(50052);
        assert_eq!(args, vec!["--host", "0.0.0.0", "--port", "50052"]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p device-agent rpc_supervisor::tests`
Expected: FAIL — `build_rpc_server_args` not defined.

- [ ] **Step 3: Implement**

```rust
// crates/device-agent/src/rpc_supervisor.rs
use std::path::Path;
use std::process::{Child, Command};

pub fn build_rpc_server_args(port: u16) -> Vec<String> {
    vec![
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        port.to_string(),
    ]
}

pub struct RpcSupervisor {
    child: Child,
}

impl RpcSupervisor {
    pub fn spawn(binary_path: &Path, port: u16) -> std::io::Result<RpcSupervisor> {
        let child = Command::new(binary_path)
            .args(build_rpc_server_args(port))
            .spawn()?;
        Ok(RpcSupervisor { child })
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for RpcSupervisor {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_expected_rpc_server_args() {
        let args = build_rpc_server_args(50052);
        assert_eq!(args, vec!["--host", "0.0.0.0", "--port", "50052"]);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p device-agent`
Expected: PASS (2 tests total across the crate)

- [ ] **Step 5: Wire into main.rs**

```rust
// crates/device-agent/src/main.rs  (replace previous contents)
mod rpc_supervisor;
mod telemetry_server;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    device_id: String,
    #[arg(long, default_value = "0.0.0.0:7100")]
    listen: String,
    #[arg(long)]
    rpc_server_binary: PathBuf,
    #[arg(long, default_value_t = 50052)]
    rpc_port: u16,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let _rpc = rpc_supervisor::RpcSupervisor::spawn(&args.rpc_server_binary, args.rpc_port)
        .expect("failed to start rpc-server — check --rpc-server-binary path");
    println!("rpc-server running on port {}", args.rpc_port);

    let app = telemetry_server::build_router(args.device_id.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await.unwrap();
    println!("device-agent '{}' listening on {}", args.device_id, args.listen);
    axum::serve(listener, app).await.unwrap();
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/device-agent
git commit -m "device-agent: supervise rpc-server subprocess"
```

---

### Task 5: `coordinator` — device registry (static config)

**Files:**
- Create: `crates/coordinator/Cargo.toml`
- Create: `crates/coordinator/src/main.rs`
- Create: `crates/coordinator/src/registry.rs`
- Create: `config/devices.example.toml`

**Interfaces:**
- Produces: `pub struct DeviceConfig { pub id: String, pub host: String, pub telemetry_port: u16, pub rpc_port: u16 }` with `Deserialize`.
- Produces: `pub fn parse_devices_toml(contents: &str) -> Result<Vec<DeviceConfig>, toml::de::Error>` — pure, unit tested.
- Produces: `pub async fn fetch_all_profiles(devices: &[DeviceConfig]) -> Vec<telemetry::DeviceProfile>` — hits the network, calls each device's `/telemetry` (from Task 3). Not unit tested; manual verification in Task 8.

- [ ] **Step 1: Crate scaffolding**

```toml
# crates/coordinator/Cargo.toml
[package]
name = "coordinator"
version = "0.1.0"
edition = "2021"

[dependencies]
telemetry = { path = "../telemetry" }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 2: Write the failing test for TOML parsing**

```rust
// crates/coordinator/src/registry.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_devices() {
        let contents = r#"
            [[device]]
            id = "desktop"
            host = "192.168.1.10"
            telemetry_port = 7100
            rpc_port = 50052

            [[device]]
            id = "old-laptop"
            host = "192.168.1.11"
            telemetry_port = 7100
            rpc_port = 50052
        "#;
        let devices = parse_devices_toml(contents).unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].id, "desktop");
        assert_eq!(devices[1].host, "192.168.1.11");
    }

    #[test]
    fn rejects_malformed_toml() {
        let result = parse_devices_toml("this is not valid toml [[[");
        assert!(result.is_err());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p coordinator registry::tests`
Expected: FAIL — `parse_devices_toml`/`DeviceConfig` not defined.

- [ ] **Step 4: Implement**

```rust
// crates/coordinator/src/registry.rs
use serde::Deserialize;
use telemetry::DeviceProfile;

#[derive(Debug, Clone, Deserialize)]
pub struct DeviceConfig {
    pub id: String,
    pub host: String,
    pub telemetry_port: u16,
    pub rpc_port: u16,
}

#[derive(Debug, Deserialize)]
struct DevicesFile {
    device: Vec<DeviceConfig>,
}

pub fn parse_devices_toml(contents: &str) -> Result<Vec<DeviceConfig>, toml::de::Error> {
    let parsed: DevicesFile = toml::from_str(contents)?;
    Ok(parsed.device)
}

pub async fn fetch_all_profiles(devices: &[DeviceConfig]) -> Vec<DeviceProfile> {
    let mut profiles = Vec::with_capacity(devices.len());
    for d in devices {
        let url = format!("http://{}:{}/telemetry", d.host, d.telemetry_port);
        match reqwest::get(&url).await {
            Ok(resp) => match resp.json::<DeviceProfile>().await {
                Ok(profile) => profiles.push(profile),
                Err(e) => eprintln!("warning: bad telemetry response from {}: {e}", d.id),
            },
            Err(e) => eprintln!("warning: could not reach {} at {}: {e}", d.id, url),
        }
    }
    profiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_devices() {
        let contents = r#"
            [[device]]
            id = "desktop"
            host = "192.168.1.10"
            telemetry_port = 7100
            rpc_port = 50052

            [[device]]
            id = "old-laptop"
            host = "192.168.1.11"
            telemetry_port = 7100
            rpc_port = 50052
        "#;
        let devices = parse_devices_toml(contents).unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].id, "desktop");
        assert_eq!(devices[1].host, "192.168.1.11");
    }

    #[test]
    fn rejects_malformed_toml() {
        let result = parse_devices_toml("this is not valid toml [[[");
        assert!(result.is_err());
    }
}
```

```toml
# config/devices.example.toml
[[device]]
id = "desktop"
host = "192.168.1.10"
telemetry_port = 7100
rpc_port = 50052

[[device]]
id = "old-laptop"
host = "192.168.1.11"
telemetry_port = 7100
rpc_port = 50052
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p coordinator`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add crates/coordinator config/devices.example.toml
git commit -m "coordinator: device registry, static TOML config"
```

---

### Task 6: `coordinator` — placement algorithm (the algorithmic core)

**Files:**
- Create: `crates/coordinator/src/placement.rs`
- Modify: `crates/coordinator/src/main.rs` (wire `pub mod placement;`)

**Interfaces:**
- Consumes: `telemetry::DeviceProfile` from Task 2/5.
- Produces: `pub struct LayerAssignment { pub device_id: String, pub layer_count: u32 }` — Task 7's launcher consumes this exact struct.
- Produces: `pub fn plan_placement(devices: &[DeviceProfile], total_layers: u32, bytes_per_layer: u64) -> Vec<LayerAssignment>` — deterministic, unit tested with multiple synthetic scenarios. Output order matches input `devices` order.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/coordinator/src/placement.rs
#[cfg(test)]
mod tests {
    use super::*;
    use telemetry::DeviceProfile;

    fn profile(id: &str, score: f64, mem_bytes: u64) -> DeviceProfile {
        DeviceProfile {
            device_id: id.to_string(),
            compute_score: score,
            available_mem_bytes: mem_bytes,
            battery_percent: None,
        }
    }

    #[test]
    fn equal_devices_split_layers_evenly() {
        let devices = vec![
            profile("a", 10.0, 1_000_000_000),
            profile("b", 10.0, 1_000_000_000),
        ];
        let plan = plan_placement(&devices, 32, 1_000_000);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].layer_count, 16);
        assert_eq!(plan[1].layer_count, 16);
    }

    #[test]
    fn faster_device_gets_more_layers() {
        let devices = vec![
            profile("fast", 30.0, 10_000_000_000),
            profile("slow", 10.0, 10_000_000_000),
        ];
        let plan = plan_placement(&devices, 40, 1_000_000);
        // weight ratio 3:1 -> fast should get roughly 3x slow's layers
        assert!(plan[0].layer_count > plan[1].layer_count * 2);
        assert_eq!(plan[0].layer_count + plan[1].layer_count, 40);
    }

    #[test]
    fn memory_constrained_device_spills_remainder_to_others() {
        // "tiny" wants an equal share by compute score but physically cannot
        // hold more than 5 layers; the rest must go to "big".
        let devices = vec![
            profile("tiny", 10.0, 5_000_000), // fits exactly 5 layers at 1MB/layer
            profile("big", 10.0, 1_000_000_000),
        ];
        let plan = plan_placement(&devices, 40, 1_000_000);
        let tiny = plan.iter().find(|p| p.device_id == "tiny").unwrap();
        let big = plan.iter().find(|p| p.device_id == "big").unwrap();
        assert!(tiny.layer_count <= 5);
        assert_eq!(tiny.layer_count + big.layer_count, 40);
    }

    #[test]
    fn all_layers_are_always_assigned() {
        let devices = vec![
            profile("a", 7.0, 2_000_000_000),
            profile("b", 3.0, 2_000_000_000),
            profile("c", 5.0, 2_000_000_000),
        ];
        let plan = plan_placement(&devices, 33, 1_000_000);
        let total: u32 = plan.iter().map(|p| p.layer_count).sum();
        assert_eq!(total, 33);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p coordinator placement::tests`
Expected: FAIL — `plan_placement`/`LayerAssignment` not defined.

- [ ] **Step 3: Implement the placement algorithm**

```rust
// crates/coordinator/src/placement.rs
use telemetry::DeviceProfile;

#[derive(Debug, Clone, PartialEq)]
pub struct LayerAssignment {
    pub device_id: String,
    pub layer_count: u32,
}

/// Halda-inspired greedy placement: weight by compute_score, cap by how many
/// layers actually fit in each device's available memory, then hand any
/// remainder (from rounding, or from capacity caps) to the fastest device(s)
/// with spare capacity, one layer at a time, fastest first.
pub fn plan_placement(
    devices: &[DeviceProfile],
    total_layers: u32,
    bytes_per_layer: u64,
) -> Vec<LayerAssignment> {
    assert!(!devices.is_empty(), "plan_placement requires at least one device");
    assert!(bytes_per_layer > 0, "bytes_per_layer must be positive");

    let capacities: Vec<u32> = devices
        .iter()
        .map(|d| (d.available_mem_bytes / bytes_per_layer) as u32)
        .collect();

    let total_weight: f64 = devices.iter().map(|d| d.compute_score).sum();

    // Initial share: proportional to compute_score, capped by capacity.
    let mut assigned: Vec<u32> = devices
        .iter()
        .zip(&capacities)
        .map(|(d, &cap)| {
            let share = if total_weight > 0.0 {
                ((d.compute_score / total_weight) * total_layers as f64).floor() as u32
            } else {
                total_layers / devices.len() as u32
            };
            share.min(cap)
        })
        .collect();

    // Distribute the remainder (unassigned layers, from flooring or capacity
    // caps) to devices with spare capacity, ordered fastest-first.
    let mut remaining: u32 = total_layers - assigned.iter().sum::<u32>();

    let mut order: Vec<usize> = (0..devices.len()).collect();
    order.sort_by(|&a, &b| {
        devices[b]
            .compute_score
            .partial_cmp(&devices[a].compute_score)
            .unwrap()
    });

    while remaining > 0 {
        let mut progressed = false;
        for &i in &order {
            if remaining == 0 {
                break;
            }
            if assigned[i] < capacities[i] {
                assigned[i] += 1;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            // No device has any spare capacity left — total capacity is less
            // than total_layers. Dump the remainder on the highest-capacity
            // device rather than silently dropping layers; caller (Task 8)
            // is responsible for noticing capacity < total_layers and
            // choosing a smaller model or a lower quantization.
            let biggest = capacities
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap();
            assigned[biggest] += remaining;
            remaining = 0;
        }
    }

    devices
        .iter()
        .zip(assigned)
        .map(|(d, layer_count)| LayerAssignment {
            device_id: d.device_id.clone(),
            layer_count,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use telemetry::DeviceProfile;

    fn profile(id: &str, score: f64, mem_bytes: u64) -> DeviceProfile {
        DeviceProfile {
            device_id: id.to_string(),
            compute_score: score,
            available_mem_bytes: mem_bytes,
            battery_percent: None,
        }
    }

    #[test]
    fn equal_devices_split_layers_evenly() {
        let devices = vec![
            profile("a", 10.0, 1_000_000_000),
            profile("b", 10.0, 1_000_000_000),
        ];
        let plan = plan_placement(&devices, 32, 1_000_000);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].layer_count, 16);
        assert_eq!(plan[1].layer_count, 16);
    }

    #[test]
    fn faster_device_gets_more_layers() {
        let devices = vec![
            profile("fast", 30.0, 10_000_000_000),
            profile("slow", 10.0, 10_000_000_000),
        ];
        let plan = plan_placement(&devices, 40, 1_000_000);
        assert!(plan[0].layer_count > plan[1].layer_count * 2);
        assert_eq!(plan[0].layer_count + plan[1].layer_count, 40);
    }

    #[test]
    fn memory_constrained_device_spills_remainder_to_others() {
        let devices = vec![
            profile("tiny", 10.0, 5_000_000),
            profile("big", 10.0, 1_000_000_000),
        ];
        let plan = plan_placement(&devices, 40, 1_000_000);
        let tiny = plan.iter().find(|p| p.device_id == "tiny").unwrap();
        let big = plan.iter().find(|p| p.device_id == "big").unwrap();
        assert!(tiny.layer_count <= 5);
        assert_eq!(tiny.layer_count + big.layer_count, 40);
    }

    #[test]
    fn all_layers_are_always_assigned() {
        let devices = vec![
            profile("a", 7.0, 2_000_000_000),
            profile("b", 3.0, 2_000_000_000),
            profile("c", 5.0, 2_000_000_000),
        ];
        let plan = plan_placement(&devices, 33, 1_000_000);
        let total: u32 = plan.iter().map(|p| p.layer_count).sum();
        assert_eq!(total, 33);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p coordinator`
Expected: PASS (6 tests total across the crate)

- [ ] **Step 5: Commit**

```bash
git add crates/coordinator/src/placement.rs
git commit -m "coordinator: Halda-inspired placement algorithm"
```

---

### Task 7: `coordinator` — cluster launcher (argv builder)

**Files:**
- Create: `crates/coordinator/src/launcher.rs`
- Modify: `crates/coordinator/src/main.rs` (tie registry → placement → launcher together end-to-end)

**Interfaces:**
- Consumes: `placement::LayerAssignment` from Task 6, `registry::DeviceConfig` from Task 5.
- Produces: `pub fn build_llama_server_args(model_path: &Path, port: u16, devices: &[DeviceConfig], assignments: &[LayerAssignment]) -> Vec<String>` — pure, unit tested for exact argv.

- [ ] **Step 1: Write the failing test**

```rust
// crates/coordinator/src/launcher.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::LayerAssignment;
    use crate::registry::DeviceConfig;
    use std::path::PathBuf;

    #[test]
    fn builds_expected_llama_server_argv() {
        let devices = vec![
            DeviceConfig { id: "desktop".into(), host: "192.168.1.10".into(), telemetry_port: 7100, rpc_port: 50052 },
            DeviceConfig { id: "laptop".into(), host: "192.168.1.11".into(), telemetry_port: 7100, rpc_port: 50052 },
        ];
        let assignments = vec![
            LayerAssignment { device_id: "desktop".into(), layer_count: 24 },
            LayerAssignment { device_id: "laptop".into(), layer_count: 8 },
        ];

        let args = build_llama_server_args(
            &PathBuf::from("/models/model.gguf"),
            8080,
            &devices,
            &assignments,
        );

        assert_eq!(
            args,
            vec![
                "-m", "/models/model.gguf",
                "--port", "8080",
                "--rpc", "192.168.1.10:50052,192.168.1.11:50052",
                "--tensor-split", "24,8",
            ]
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p coordinator launcher::tests`
Expected: FAIL — `build_llama_server_args` not defined.

- [ ] **Step 3: Implement**

```rust
// crates/coordinator/src/launcher.rs
use crate::placement::LayerAssignment;
use crate::registry::DeviceConfig;
use std::path::Path;
use std::process::{Child, Command};

/// Builds argv in device order (the order `devices` is given in — callers
/// must pass `assignments` sorted to match `devices`' order; `plan_placement`
/// already preserves input order so this is satisfied end-to-end).
pub fn build_llama_server_args(
    model_path: &Path,
    port: u16,
    devices: &[DeviceConfig],
    assignments: &[LayerAssignment],
) -> Vec<String> {
    let rpc_hosts: Vec<String> = devices
        .iter()
        .map(|d| format!("{}:{}", d.host, d.rpc_port))
        .collect();

    let splits: Vec<String> = assignments
        .iter()
        .map(|a| a.layer_count.to_string())
        .collect();

    vec![
        "-m".to_string(),
        model_path.to_string_lossy().to_string(),
        "--port".to_string(),
        port.to_string(),
        "--rpc".to_string(),
        rpc_hosts.join(","),
        "--tensor-split".to_string(),
        splits.join(","),
    ]
}

pub struct LlamaServerHandle {
    child: Child,
}

impl LlamaServerHandle {
    pub fn launch(binary_path: &Path, args: &[String]) -> std::io::Result<LlamaServerHandle> {
        let child = Command::new(binary_path).args(args).spawn()?;
        Ok(LlamaServerHandle { child })
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for LlamaServerHandle {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::LayerAssignment;
    use crate::registry::DeviceConfig;
    use std::path::PathBuf;

    #[test]
    fn builds_expected_llama_server_argv() {
        let devices = vec![
            DeviceConfig { id: "desktop".into(), host: "192.168.1.10".into(), telemetry_port: 7100, rpc_port: 50052 },
            DeviceConfig { id: "laptop".into(), host: "192.168.1.11".into(), telemetry_port: 7100, rpc_port: 50052 },
        ];
        let assignments = vec![
            LayerAssignment { device_id: "desktop".into(), layer_count: 24 },
            LayerAssignment { device_id: "laptop".into(), layer_count: 8 },
        ];

        let args = build_llama_server_args(
            &PathBuf::from("/models/model.gguf"),
            8080,
            &devices,
            &assignments,
        );

        assert_eq!(
            args,
            vec![
                "-m", "/models/model.gguf",
                "--port", "8080",
                "--rpc", "192.168.1.10:50052,192.168.1.11:50052",
                "--tensor-split", "24,8",
            ]
        );
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p coordinator`
Expected: PASS (7 tests total across the crate)

- [ ] **Step 5: Wire everything together in main.rs**

```rust
// crates/coordinator/src/main.rs
mod launcher;
mod placement;
mod registry;

use clap::Parser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    devices_config: PathBuf,
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    llama_server_binary: PathBuf,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value_t = 32)]
    total_layers: u32,
    #[arg(long, default_value_t = 200_000_000)]
    bytes_per_layer: u64,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let contents = fs::read_to_string(&args.devices_config).expect("cannot read devices config");
    let devices = registry::parse_devices_toml(&contents).expect("invalid devices.toml");

    println!("Fetching telemetry from {} devices...", devices.len());
    let profiles = registry::fetch_all_profiles(&devices).await;
    if profiles.len() != devices.len() {
        eprintln!(
            "warning: only reached {}/{} devices — proceeding with those",
            profiles.len(),
            devices.len()
        );
    }

    let plan = placement::plan_placement(&profiles, args.total_layers, args.bytes_per_layer);
    println!("Computed placement:");
    for p in &plan {
        println!("  {} -> {} layers", p.device_id, p.layer_count);
    }

    let live_devices: Vec<_> = devices
        .into_iter()
        .filter(|d| profiles.iter().any(|p| p.device_id == d.id))
        .collect();

    let llama_args = launcher::build_llama_server_args(&args.model_path, args.port, &live_devices, &plan);
    println!("Launching: {} {}", args.llama_server_binary.display(), llama_args.join(" "));

    let mut handle = launcher::LlamaServerHandle::launch(&args.llama_server_binary, &llama_args)
        .expect("failed to launch llama-server");

    println!("llama-server running on port {}. Ctrl+C to stop.", args.port);
    tokio::signal::ctrl_c().await.ok();
    handle.kill();
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/coordinator/src/launcher.rs crates/coordinator/src/main.rs
git commit -m "coordinator: cluster launcher, wire registry->placement->launcher end-to-end"
```

---

### Task 8: Milestone B manual validation on real hardware

This task has no unit tests — it is the real-world proof that Tasks 1-7 produce a
working cluster. Follow it exactly and record the numbers; they become the
"before" baseline that Milestones C and D are measured against.

**Files:** none created; this is a verification checklist. Record results in
`docs/milestone-b-results.md`.

- [ ] **Step 1: Build llama.cpp identically on every device**

On the coordinator machine AND every worker device:
```bash
git -C vendor/llama.cpp checkout $(cat docs/llama.cpp.pinned-commit.txt)
cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build --config Release -j
```
Expected: identical commit hash and successful build on all machines.

- [ ] **Step 2: Start a device-agent on each worker device**

```bash
cargo run -p device-agent --release -- \
  --device-id old-laptop \
  --listen 0.0.0.0:7100 \
  --rpc-server-binary /full/path/to/vendor/llama.cpp/build/bin/rpc-server \
  --rpc-port 50052
```
Expected output: `rpc-server running on port 50052` then `device-agent 'old-laptop'
listening on 0.0.0.0:7100`. Repeat for every worker device (different `--device-id`
per device).

- [ ] **Step 3: Write `config/devices.toml` with your real IPs**

Copy `config/devices.example.toml` to `config/devices.toml` and fill in the actual
LAN IPs and ports of the devices started in Step 2.

- [ ] **Step 4: Run the coordinator**

```bash
cargo run -p coordinator --release -- \
  --devices-config config/devices.toml \
  --model-path /path/to/your-model.gguf \
  --llama-server-binary vendor/llama.cpp/build/bin/llama-server \
  --total-layers 32 \
  --bytes-per-layer 200000000
```
Expected: telemetry fetched from all devices, a computed placement printed (record
these numbers), `llama-server` launches successfully and logs it's listening on port
8080.

- [ ] **Step 5: Benchmark the computed placement**

```bash
python bench/harness.py --label milestone-b-computed-split
```

- [ ] **Step 6: Benchmark a naive memory-proportional baseline for comparison**

Stop the coordinator (Ctrl+C). Launch `llama-server` directly with `--rpc` pointing
at the same devices but *without* `--tensor-split` (llama.cpp's own default behavior:
memory-proportional):
```bash
vendor/llama.cpp/build/bin/llama-server \
  -m /path/to/your-model.gguf \
  --port 8080 \
  --rpc 192.168.1.10:50052,192.168.1.11:50052
python bench/harness.py --label milestone-b-naive-split
```

- [ ] **Step 7: Record the comparison**

```markdown
# docs/milestone-b-results.md
| Config | tokens/sec | TTFT (s) |
|---|---|---|
| milestone-a-single-device | ... | ... |
| milestone-b-naive-split | ... | ... |
| milestone-b-computed-split | ... | ... |
```

Expected: `milestone-b-computed-split` outperforms `milestone-b-naive-split` on
tokens/sec — this is the number that justifies everything built in Tasks 2-7. If it
doesn't, that's a real, useful negative result — investigate whether your device
pool's compute-score spread is large enough for placement to matter before Milestone C.

- [ ] **Step 8: Commit results**

```bash
git add docs/milestone-b-results.md config/devices.toml.example
git commit -m "Milestone B: recorded computed-split vs naive-split comparison"
```

Note: do NOT commit your real `config/devices.toml` if it contains your home network's
real IPs and you intend to make this repo public — add it to `.gitignore` and keep only
`devices.example.toml` tracked.
