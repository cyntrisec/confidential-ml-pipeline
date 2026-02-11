use std::time::Instant;

use bytes::Bytes;
use clap::Parser;
use confidential_ml_transport::{DType, OwnedTensor};
use tracing::info;

use confidential_ml_pipeline::OrchestratorConfig;
use confidential_ml_pipeline::ShardManifest;

#[cfg(feature = "tcp-mock")]
use std::net::SocketAddr;
#[cfg(feature = "tcp-mock")]
use confidential_ml_transport::{MockProvider, MockVerifier};
#[cfg(feature = "tcp-mock")]
use confidential_ml_pipeline::tcp;

#[cfg(feature = "vsock-mock")]
use confidential_ml_transport::{MockProvider, MockVerifier};
#[cfg(feature = "vsock-mock")]
use confidential_ml_pipeline::vsock;

#[cfg(feature = "vsock-nitro")]
use confidential_ml_transport::{NitroProvider, NitroVerifier};
#[cfg(feature = "vsock-nitro")]
use confidential_ml_pipeline::vsock;

#[cfg(feature = "tcp-azure-sev-snp")]
use std::net::SocketAddr;
#[cfg(feature = "tcp-azure-sev-snp")]
use confidential_ml_transport::{AzureSevSnpProvider, AzureSevSnpVerifier};
#[cfg(feature = "tcp-azure-sev-snp")]
use confidential_ml_pipeline::tcp;

#[cfg(feature = "tcp-tdx")]
use std::net::SocketAddr;
#[cfg(feature = "tcp-tdx")]
use confidential_ml_transport::{TdxProvider, TdxVerifier};
#[cfg(feature = "tcp-tdx")]
use confidential_ml_pipeline::tcp;

#[derive(Parser)]
#[command(name = "pipeline-orch", about = "GPT-2 pipeline orchestrator")]
struct Args {
    /// Path to the shard manifest JSON file.
    #[arg(long)]
    manifest: String,

    /// Path to the tokenizer.json file.
    #[arg(long)]
    tokenizer: String,

    /// Input text prompt.
    #[arg(long)]
    text: String,

    /// Number of tokens to generate.
    #[arg(long, default_value = "20")]
    max_tokens: usize,

    /// Output per-token latency stats as JSON to this file.
    #[arg(long)]
    latency_out: Option<String>,

    /// (TCP mode) Address to listen for the last stage's data_out connection.
    #[cfg(any(feature = "tcp-mock", feature = "tcp-azure-sev-snp", feature = "tcp-tdx"))]
    #[arg(long)]
    data_out_listen: String,

}

/// Build a cache-clear sentinel tensor (U32 with shape [0]).
fn cache_clear_sentinel() -> OwnedTensor {
    OwnedTensor {
        name: "cache_clear".to_string(),
        dtype: DType::U32,
        shape: vec![0],
        data: Bytes::new(),
    }
}

fn encode_token_ids(token_ids: &[u32]) -> OwnedTensor {
    let seq_len = token_ids.len();
    let data: Vec<u8> = token_ids.iter().flat_map(|&id| id.to_le_bytes()).collect();
    OwnedTensor {
        name: "input_ids".to_string(),
        dtype: DType::U32,
        shape: vec![1, seq_len as u32],
        data: Bytes::from(data),
    }
}

fn decode_logits(tensor: &OwnedTensor) -> u32 {
    let values: Vec<f32> = tensor
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (best_idx, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    best_idx as u32
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

    info!(
        stages = manifest.stages.len(),
        prompt = %args.text,
        max_tokens = args.max_tokens,
        "starting GPT-2 orchestrator"
    );

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let encoding = tokenizer
        .encode(args.text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    info!(prompt_tokens = token_ids.len(), "prompt tokenized");

    // Initialize orchestrator with the appropriate transport.
    #[cfg(feature = "tcp-mock")]
    let mut orch = {
        let dout_listen: SocketAddr = args.data_out_listen.parse()?;
        info!(data_out_listen = %dout_listen, "binding TCP data_out listener");

        let dout_listener = tokio::net::TcpListener::bind(dout_listen).await?;
        let dout_local = dout_listener.local_addr()?;
        info!(addr = %dout_local, "data_out listener bound");

        let verifier = MockVerifier::new();
        let provider = MockProvider::new();

        tcp::init_orchestrator_tcp(
            OrchestratorConfig::default(),
            manifest,
            dout_listener,
            &verifier,
            &provider,
        )
        .await?
    };

    #[cfg(feature = "tcp-azure-sev-snp")]
    let mut orch = {
        let dout_listen: SocketAddr = args.data_out_listen.parse()?;
        info!(data_out_listen = %dout_listen, "binding TCP data_out listener");

        let dout_listener = tokio::net::TcpListener::bind(dout_listen).await?;
        let dout_local = dout_listener.local_addr()?;
        info!(addr = %dout_local, "data_out listener bound");

        let verifier = AzureSevSnpVerifier::new(None);
        let provider = AzureSevSnpProvider::new()?;

        tcp::init_orchestrator_tcp(
            OrchestratorConfig::default(),
            manifest,
            dout_listener,
            &verifier,
            &provider,
        )
        .await?
    };

    #[cfg(feature = "tcp-tdx")]
    let mut orch = {
        let dout_listen: SocketAddr = args.data_out_listen.parse()?;
        info!(data_out_listen = %dout_listen, "binding TCP data_out listener");

        let dout_listener = tokio::net::TcpListener::bind(dout_listen).await?;
        let dout_local = dout_listener.local_addr()?;
        info!(addr = %dout_local, "data_out listener bound");

        let verifier = TdxVerifier::new(None);
        let provider = TdxProvider::new()?;

        tcp::init_orchestrator_tcp(
            OrchestratorConfig::default(),
            manifest,
            dout_listener,
            &verifier,
            &provider,
        )
        .await?
    };

    #[cfg(any(feature = "vsock-nitro", feature = "vsock-mock"))]
    let mut orch = {
        use tokio_vsock::{VsockAddr, VsockListener, VMADDR_CID_ANY};

        // Read data_out port from last stage's endpoint in the manifest.
        let last_stage = manifest
            .stages
            .last()
            .ok_or_else(|| anyhow::anyhow!("manifest has no stages"))?;
        let (_, data_out_port) = vsock::resolve_vsock(&last_stage.endpoint.data_out)?;

        info!(data_out_port, "binding VSock data_out listener");

        let dout_listener = VsockListener::bind(VsockAddr::new(VMADDR_CID_ANY, data_out_port))
            .map_err(|e| anyhow::anyhow!("failed to bind VSock listener: {e}"))?;
        info!(port = data_out_port, "VSock data_out listener bound");

        #[cfg(feature = "vsock-mock")]
        let (verifier, provider) = (MockVerifier::new(), MockProvider::new());

        #[cfg(feature = "vsock-nitro")]
        let (verifier, provider) = (
            NitroVerifier::new(std::collections::BTreeMap::new())?,
            NitroProvider::new()?,
        );

        vsock::init_orchestrator_vsock(
            OrchestratorConfig::default(),
            manifest,
            dout_listener,
            &verifier,
            &provider,
        )
        .await?
    };

    info!("pipeline initialized, running health check");
    orch.health_check().await?;
    info!("all stages healthy");

    print!("{}", args.text);

    let mut latencies_ms: Vec<f64> = Vec::new();

    for step in 0..args.max_tokens {
        let t0 = Instant::now();

        let input = if step == 0 {
            // First step: clear cache + send full prompt
            vec![vec![cache_clear_sentinel(), encode_token_ids(&token_ids)]]
        } else {
            // Subsequent steps: send only the new token (KV-cache handles history)
            let new_token = *token_ids.last().unwrap();
            vec![vec![encode_token_ids(&[new_token])]]
        };

        let seq_len = token_ids.len();
        let result = orch.infer(input, seq_len as u32).await?;

        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        latencies_ms.push(ms);

        let output_tensors = &result.outputs[0];
        let logits = &output_tensors[0];
        let next_token = decode_logits(logits);

        let decoded = tokenizer
            .decode(&[next_token], false)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        print!("{decoded}");
        use std::io::Write;
        std::io::stdout().flush()?;

        token_ids.push(next_token);

        info!(
            step,
            next_token,
            decoded = decoded.as_str(),
            total_tokens = token_ids.len(),
            latency_ms = format!("{ms:.1}"),
            "generated token"
        );
    }

    println!();

    // Print latency summary — percentiles over generation tokens only (excluding prompt).
    if !latencies_ms.is_empty() {
        let prompt_ms = latencies_ms[0];
        let gen_latencies: Vec<f64> = latencies_ms[1..].to_vec();
        let gen_count = gen_latencies.len();
        let gen_avg = if gen_count > 0 {
            gen_latencies.iter().sum::<f64>() / gen_count as f64
        } else {
            0.0
        };
        let (p50, p95) = if gen_count > 0 {
            let mut sorted = gen_latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = sorted[sorted.len() / 2];
            let p95_idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
            let p95 = sorted[p95_idx];
            (p50, p95)
        } else {
            (0.0, 0.0)
        };
        eprintln!();
        eprintln!("--- Latency Summary ---");
        eprintln!("  Prompt (TTFT):         {prompt_ms:.1}ms");
        eprintln!("  Generation avg:        {gen_avg:.1}ms/token");
        eprintln!("  Generation p50:        {p50:.1}ms");
        eprintln!("  Generation p95:        {p95:.1}ms");
        eprintln!("  Tokens generated:      {}", latencies_ms.len());
    }

    // Write latency JSON if requested
    if let Some(path) = &args.latency_out {
        let json = serde_json::json!({
            "latencies_ms": latencies_ms,
            "prompt_ms": latencies_ms.first().copied().unwrap_or(0.0),
            "generation_tokens": if latencies_ms.len() > 1 { latencies_ms.len() - 1 } else { 0 },
        });
        std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
        info!(path, "latency data written");
    }

    info!("generation complete, shutting down");
    orch.shutdown().await?;
    info!("done");

    Ok(())
}
