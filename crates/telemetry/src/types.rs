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
