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
