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
