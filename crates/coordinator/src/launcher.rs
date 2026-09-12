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
