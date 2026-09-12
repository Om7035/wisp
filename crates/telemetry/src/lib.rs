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
