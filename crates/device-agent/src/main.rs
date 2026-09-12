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
