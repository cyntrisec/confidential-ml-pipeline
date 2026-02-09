mod executor;
mod model;

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use confidential_ml_transport::{MockProvider, MockVerifier};
use tracing::info;

use confidential_ml_pipeline::{ShardManifest, StageConfig};
use confidential_ml_pipeline::tcp;

use executor::Gpt2StageExecutor;

#[derive(Parser)]
#[command(name = "stage-worker", about = "GPT-2 pipeline stage worker (TCP)")]
struct Args {
    /// Path to the shard manifest JSON file.
    #[arg(long)]
    manifest: String,

    /// Index of this stage in the manifest.
    #[arg(long)]
    stage_idx: usize,

    /// Address to connect data_out to (next stage's data_in or orchestrator's data_out listener).
    #[arg(long)]
    data_out_target: String,

    /// Path to the model directory (containing model.safetensors, config.json).
    #[arg(long)]
    model_dir: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let manifest_json = std::fs::read_to_string(&args.manifest)?;
    let manifest = ShardManifest::from_json(&manifest_json)?;

    let stage_spec = &manifest.stages[args.stage_idx];
    let ctrl_addr: SocketAddr = tcp::resolve_tcp(&stage_spec.endpoint.control)?;
    let din_addr: SocketAddr = tcp::resolve_tcp(&stage_spec.endpoint.data_in)?;
    let dout_target: SocketAddr = args.data_out_target.parse()?;

    info!(
        stage = args.stage_idx,
        ctrl = %ctrl_addr,
        data_in = %din_addr,
        data_out_target = %dout_target,
        model_dir = %args.model_dir,
        "starting GPT-2 stage worker"
    );

    let (ctrl_lis, ctrl_local, din_lis, din_local) =
        tcp::bind_stage_listeners(ctrl_addr, din_addr).await?;

    info!(ctrl = %ctrl_local, data_in = %din_local, "listeners ready");

    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    tcp::run_stage_with_listeners(
        Gpt2StageExecutor::new(PathBuf::from(&args.model_dir)),
        StageConfig::default(),
        ctrl_lis,
        din_lis,
        dout_target,
        &provider,
        &verifier,
    )
    .await?;

    info!("stage worker exiting");
    Ok(())
}
