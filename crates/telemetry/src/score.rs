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
