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
