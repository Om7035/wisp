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
