use std::net::SocketAddr;

use bytes::Bytes;
use clap::Parser;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};
use tracing::info;

use confidential_ml_pipeline::tcp;
use confidential_ml_pipeline::OrchestratorConfig;
use confidential_ml_pipeline::ShardManifest;

#[derive(Parser)]
#[command(name = "pipeline-orch", about = "GPT-2 pipeline orchestrator (TCP)")]
struct Args {
    /// Path to the shard manifest JSON file.
    #[arg(long)]
    manifest: String,

    /// Address to listen for the last stage's data_out connection.
    #[arg(long)]
    data_out_listen: String,

    /// Path to the tokenizer.json file.
    #[arg(long)]
    tokenizer: String,

    /// Input text prompt.
    #[arg(long)]
    text: String,

    /// Number of tokens to generate.
    #[arg(long, default_value = "20")]
    max_tokens: usize,
}

fn encode_token_ids(token_ids: &[u32]) -> Vec<OwnedTensor> {
    let seq_len = token_ids.len();
    let data: Vec<u8> = token_ids.iter().flat_map(|&id| id.to_le_bytes()).collect();
    vec![OwnedTensor {
        name: "input_ids".to_string(),
        dtype: DType::U32,
        shape: vec![1, seq_len as u32],
        data: Bytes::from(data),
    }]
}

fn decode_logits(tensor: &OwnedTensor) -> u32 {
    // logits shape: [1, vocab_size] or [vocab_size], dtype F32
    let values: Vec<f32> = tensor
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // argmax
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

    let dout_listen: SocketAddr = args.data_out_listen.parse()?;

    info!(
        stages = manifest.stages.len(),
        data_out_listen = %dout_listen,
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

    print!("{}", args.text);

    for step in 0..args.max_tokens {
        let seq_len = token_ids.len();
        let input = vec![encode_token_ids(&token_ids)];

        let result = orch.infer(input, seq_len as u32).await?;

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
            "generated token"
        );
    }

    println!();

    info!("generation complete, shutting down");
    orch.shutdown().await?;
    info!("done");

    Ok(())
}
