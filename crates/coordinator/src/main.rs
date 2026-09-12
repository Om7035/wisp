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
