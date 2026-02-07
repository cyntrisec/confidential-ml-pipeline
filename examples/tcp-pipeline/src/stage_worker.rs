use std::net::SocketAddr;

use async_trait::async_trait;
use bytes::Bytes;
use clap::Parser;
use confidential_ml_transport::{MockProvider, MockVerifier, OwnedTensor};
use tracing::info;

use confidential_ml_pipeline::{
    ForwardOutput, RequestId, ShardManifest, StageConfig, StageError, StageExecutor, StageSpec,
};
use confidential_ml_pipeline::tcp;

/// Executor that multiplies all tensor data bytes by 2.
struct DoubleExecutor {
    stage_idx: usize,
}

#[async_trait]
impl StageExecutor for DoubleExecutor {
    async fn init(&mut self, stage_spec: &StageSpec) -> Result<(), StageError> {
        self.stage_idx = stage_spec.stage_idx;
        info!(
            stage = stage_spec.stage_idx,
            layers = format!("{}-{}", stage_spec.layer_start, stage_spec.layer_end),
            "initialized"
        );
        Ok(())
    }

    async fn forward(
        &self,
        _request_id: RequestId,
        micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        let outputs: Vec<OwnedTensor> = inputs
            .into_iter()
            .map(|t| {
                let doubled: Vec<u8> = t.data.iter().map(|b| b.wrapping_mul(2)).collect();
                info!(
                    stage = self.stage_idx,
                    micro_batch,
                    tensor = t.name.as_str(),
                    bytes = doubled.len(),
                    "forward"
                );
                OwnedTensor {
                    name: t.name,
                    dtype: t.dtype,
                    shape: t.shape,
                    data: Bytes::from(doubled),
                }
            })
            .collect();
        Ok(ForwardOutput { tensors: outputs })
    }
}

#[derive(Parser)]
#[command(name = "stage-worker", about = "Pipeline stage worker (TCP)")]
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
        "starting stage worker"
    );

    let (ctrl_lis, ctrl_local, din_lis, din_local) =
        tcp::bind_stage_listeners(ctrl_addr, din_addr).await?;

    info!(ctrl = %ctrl_local, data_in = %din_local, "listeners ready");

    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    tcp::run_stage_with_listeners(
        DoubleExecutor { stage_idx: args.stage_idx },
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
