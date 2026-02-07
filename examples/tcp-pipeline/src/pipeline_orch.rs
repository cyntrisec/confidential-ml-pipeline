use std::net::SocketAddr;

use bytes::Bytes;
use clap::Parser;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};
use tracing::info;

use confidential_ml_pipeline::tcp;
use confidential_ml_pipeline::OrchestratorConfig;
use confidential_ml_pipeline::ShardManifest;

#[derive(Parser)]
#[command(name = "pipeline-orch", about = "Pipeline orchestrator (TCP)")]
struct Args {
    /// Path to the shard manifest JSON file.
    #[arg(long)]
    manifest: String,

    /// Address to listen for the last stage's data_out connection.
    #[arg(long)]
    data_out_listen: String,

    /// Number of micro-batches per inference request.
    #[arg(long, default_value = "1")]
    micro_batches: u32,
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
    let num_stages = manifest.stages.len();

    let dout_listen: SocketAddr = args.data_out_listen.parse()?;

    info!(
        stages = num_stages,
        data_out_listen = %dout_listen,
        "starting orchestrator"
    );

    let dout_listener = tokio::net::TcpListener::bind(dout_listen).await?;
    let dout_local = dout_listener.local_addr()?;
    info!(addr = %dout_local, "data_out listener bound");

    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let mut orch = tcp::init_orchestrator_tcp(
        OrchestratorConfig::default(),
        manifest,
        dout_listener,
        &verifier,
        &provider,
    )
    .await?;

    info!("pipeline initialized, running health check");
    orch.health_check().await?;
    info!("all stages healthy");

    // Create input tensors.
    let input: Vec<Vec<OwnedTensor>> = (0..args.micro_batches)
        .map(|mb| {
            vec![OwnedTensor {
                name: format!("input_mb{mb}"),
                dtype: DType::F32,
                shape: vec![1, 4],
                data: Bytes::from(vec![1u8; 16]),
            }]
        })
        .collect();

    info!(micro_batches = args.micro_batches, "running inference");
    let result = orch.infer(input, 16).await?;

    println!("\n=== Results ===");
    for (mb, tensors) in result.outputs.iter().enumerate() {
        for t in tensors {
            let first_byte = t.data.first().copied().unwrap_or(0);
            // After N stages of doubling: 1 * 2^N
            let expected = 1u8.wrapping_shl(num_stages as u32);
            println!(
                "  mb={mb} tensor=\"{}\" shape={:?} first_byte={first_byte} (expected: {expected})",
                t.name, t.shape,
            );
        }
    }

    info!("shutting down");
    orch.shutdown().await?;
    info!("done");

    Ok(())
}
