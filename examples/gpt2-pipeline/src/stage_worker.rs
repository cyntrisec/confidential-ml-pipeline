mod executor;
mod model;

use std::path::PathBuf;

use clap::Parser;
use tracing::info;

use confidential_ml_pipeline::StageConfig;

use executor::Gpt2StageExecutor;

#[cfg(feature = "tcp-mock")]
use std::net::SocketAddr;
#[cfg(feature = "tcp-mock")]
use confidential_ml_transport::{MockProvider, MockVerifier};
#[cfg(feature = "tcp-mock")]
use confidential_ml_pipeline::{tcp, ShardManifest};

#[cfg(feature = "vsock-mock")]
use confidential_ml_transport::{MockProvider, MockVerifier};
#[cfg(feature = "vsock-mock")]
use confidential_ml_pipeline::{vsock, ShardManifest};

#[cfg(feature = "vsock-nitro")]
use confidential_ml_transport::{NitroProvider, NitroVerifier};
#[cfg(feature = "vsock-nitro")]
use confidential_ml_pipeline::{vsock, ShardManifest};

#[cfg(feature = "tcp-azure-sev-snp")]
use std::net::SocketAddr;
#[cfg(feature = "tcp-azure-sev-snp")]
use confidential_ml_transport::{AzureSevSnpProvider, AzureSevSnpVerifier};
#[cfg(feature = "tcp-azure-sev-snp")]
use confidential_ml_pipeline::{tcp, ShardManifest};

#[derive(Parser)]
#[command(name = "stage-worker", about = "GPT-2 pipeline stage worker")]
struct Args {
    /// Path to the shard manifest JSON file.
    #[arg(long)]
    manifest: String,

    /// Index of this stage in the manifest.
    #[arg(long)]
    stage_idx: usize,

    /// Path to the model directory (containing model.safetensors, config.json).
    #[arg(long)]
    model_dir: String,

    /// (TCP mode) Address to connect data_out to (next stage's data_in or orchestrator's data_out listener).
    #[cfg(any(feature = "tcp-mock", feature = "tcp-azure-sev-snp"))]
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

    let stage_spec = manifest
        .stages
        .get(args.stage_idx)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "stage_idx {} out of range (manifest has {} stages)",
                args.stage_idx,
                manifest.stages.len()
            )
        })?;

    #[cfg(feature = "tcp-mock")]
    {
        let ctrl_addr: SocketAddr = tcp::resolve_tcp(&stage_spec.endpoint.control)?;
        let din_addr: SocketAddr = tcp::resolve_tcp(&stage_spec.endpoint.data_in)?;
        let dout_target: SocketAddr = args.data_out_target.parse()?;

        info!(
            stage = args.stage_idx,
            ctrl = %ctrl_addr,
            data_in = %din_addr,
            data_out_target = %dout_target,
            model_dir = %args.model_dir,
            "starting GPT-2 stage worker (TCP)"
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
    }

    #[cfg(feature = "tcp-azure-sev-snp")]
    {
        let ctrl_addr: SocketAddr = tcp::resolve_tcp(&stage_spec.endpoint.control)?;
        let din_addr: SocketAddr = tcp::resolve_tcp(&stage_spec.endpoint.data_in)?;
        let dout_target: SocketAddr = args.data_out_target.parse()?;

        info!(
            stage = args.stage_idx,
            ctrl = %ctrl_addr,
            data_in = %din_addr,
            data_out_target = %dout_target,
            model_dir = %args.model_dir,
            "starting GPT-2 stage worker (TCP, Azure SEV-SNP)"
        );

        let (ctrl_lis, ctrl_local, din_lis, din_local) =
            tcp::bind_stage_listeners(ctrl_addr, din_addr).await?;

        info!(ctrl = %ctrl_local, data_in = %din_local, "listeners ready");

        let provider = AzureSevSnpProvider::new()?;
        let verifier = AzureSevSnpVerifier::new(None);

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
    }

    #[cfg(any(feature = "vsock-nitro", feature = "vsock-mock"))]
    {
        let (_, ctrl_port) = vsock::resolve_vsock(&stage_spec.endpoint.control)?;
        let (_, din_port) = vsock::resolve_vsock(&stage_spec.endpoint.data_in)?;
        let (data_out_cid, data_out_port) =
            vsock::resolve_vsock(&stage_spec.endpoint.data_out)?;

        info!(
            stage = args.stage_idx,
            ctrl_port,
            data_in_port = din_port,
            data_out_cid,
            data_out_port,
            model_dir = %args.model_dir,
            "starting GPT-2 stage worker (VSock)"
        );

        let (ctrl_lis, din_lis) = vsock::bind_stage_listeners_vsock(ctrl_port, din_port)?;

        info!(ctrl_port, data_in_port = din_port, "VSock listeners ready");

        #[cfg(feature = "vsock-mock")]
        let (provider, verifier) = (MockProvider::new(), MockVerifier::new());

        #[cfg(feature = "vsock-nitro")]
        let (provider, verifier) = (
            NitroProvider::new()?,
            NitroVerifier::new(std::collections::BTreeMap::new())?,
        );

        vsock::run_stage_with_listeners_vsock(
            Gpt2StageExecutor::new(PathBuf::from(&args.model_dir)),
            StageConfig::default(),
            ctrl_lis,
            din_lis,
            data_out_cid,
            data_out_port,
            &provider,
            &verifier,
        )
        .await?;
    }

    info!("stage worker exiting");
    Ok(())
}
