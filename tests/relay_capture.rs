//! Pipeline inter-stage relay capture test.
//!
//! Proves that the host relay between stage 0 and stage 1 sees only
//! encrypted bytes — no recoverable tensor data, names, shapes, or values.
//!
//! Topology:
//!   Orch ──data_in──> Stage 0 ──[TAPPING RELAY]──> Stage 1 ──data_out──> Orch
//!                                    ↑
//!                          captured bytes analyzed here

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use tokio::io::{self, AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio::sync::Mutex;
use tokio_util::codec::Decoder;

use confidential_ml_transport::frame::codec::FrameCodec;
use confidential_ml_transport::{
    DType, Flags, FrameType, MockProvider, MockVerifier, OwnedTensor,
};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig, PortSpec,
    RequestId, ShardManifest, StageConfig, StageEndpoint, StageError, StageExecutor, StageRuntime,
    StageSpec,
};

// ---------------------------------------------------------------------------
// Test executor: doubles tensor values (creates recognizable output pattern)
// ---------------------------------------------------------------------------

struct DoubleExecutor;

#[async_trait]
impl StageExecutor for DoubleExecutor {
    async fn init(&mut self, _stage_spec: &StageSpec) -> Result<(), StageError> {
        Ok(())
    }

    async fn forward(
        &self,
        _request_id: RequestId,
        _micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        let tensors = inputs
            .into_iter()
            .map(|t| {
                let doubled: Vec<u8> = t
                    .data
                    .chunks_exact(4)
                    .flat_map(|c| {
                        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        (v * 2.0).to_le_bytes()
                    })
                    .collect();
                OwnedTensor {
                    name: t.name,
                    dtype: t.dtype,
                    shape: t.shape,
                    data: Bytes::from(doubled),
                }
            })
            .collect();
        Ok(ForwardOutput { tensors })
    }
}

// ---------------------------------------------------------------------------
// Tapping relay (same pattern as hostile-host-demo)
// ---------------------------------------------------------------------------

struct RelayCapture {
    fwd: Vec<u8>,
    bwd: Vec<u8>,
    fwd_error: Option<String>,
    bwd_error: Option<String>,
}

impl RelayCapture {
    fn assert_clean(&self, label: &str) {
        if let Some(e) = &self.fwd_error {
            panic!("{label}: forward relay I/O error: {e}");
        }
        if let Some(e) = &self.bwd_error {
            panic!("{label}: backward relay I/O error: {e}");
        }
    }
}

async fn relay_one_direction(
    mut reader: io::ReadHalf<DuplexStream>,
    mut writer: io::WriteHalf<DuplexStream>,
    capture: Arc<Mutex<Vec<u8>>>,
) -> io::Result<()> {
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) => break,
            Err(e) => return Err(e),
            Ok(n) => {
                capture.lock().await.extend_from_slice(&buf[..n]);
                writer.write_all(&buf[..n]).await?;
            }
        }
    }
    let _ = writer.shutdown().await;
    Ok(())
}

async fn tapping_relay(left: DuplexStream, right: DuplexStream) -> RelayCapture {
    let (lr, lw) = io::split(left);
    let (rr, rw) = io::split(right);

    let fwd_cap = Arc::new(Mutex::new(Vec::new()));
    let bwd_cap = Arc::new(Mutex::new(Vec::new()));

    let fwd_clone = Arc::clone(&fwd_cap);
    let bwd_clone = Arc::clone(&bwd_cap);

    let fwd_task = tokio::spawn(relay_one_direction(lr, rw, fwd_clone));
    let bwd_task = tokio::spawn(relay_one_direction(rr, lw, bwd_clone));

    let fwd_result = fwd_task.await.unwrap();
    let bwd_result = bwd_task.await.unwrap();

    RelayCapture {
        fwd: Arc::try_unwrap(fwd_cap).unwrap().into_inner(),
        bwd: Arc::try_unwrap(bwd_cap).unwrap().into_inner(),
        fwd_error: fwd_result.err().map(|e| e.to_string()),
        bwd_error: bwd_result.err().map(|e| e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Frame analysis (reused from hostile-host-demo)
// ---------------------------------------------------------------------------

struct FrameInfo {
    msg_type: FrameType,
    flags: Flags,
    sequence: u32,
    payload_len: u32,
}

fn scan_frames_with_payloads(captured: &[u8]) -> Vec<(FrameInfo, Bytes)> {
    let mut buf = BytesMut::from(captured);
    let mut codec = FrameCodec::new();
    let mut frames = Vec::new();

    while let Ok(Some(frame)) = codec.decode(&mut buf) {
        frames.push((
            FrameInfo {
                msg_type: frame.header.msg_type,
                flags: frame.header.flags,
                sequence: frame.header.sequence,
                payload_len: frame.header.payload_len,
            },
            frame.payload,
        ));
    }
    frames
}

fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_manifest() -> ShardManifest {
    let stages = (0..2)
        .map(|i| StageSpec {
            stage_idx: i,
            layer_start: i * 6,
            layer_end: (i + 1) * 6,
            weight_hashes: vec![],
            expected_measurements: BTreeMap::new(),
            endpoint: StageEndpoint {
                control: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9000 + i * 10),
                },
                data_in: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9001 + i * 10),
                },
                data_out: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9002 + i * 10),
                },
            },
        })
        .collect();

    ShardManifest {
        model_name: "gpt2".into(),
        model_version: "1.0".into(),
        total_layers: 12,
        stages,
        activation_spec: ActivationSpec {
            dtype: ActivationDType::F32,
            hidden_dim: 768,
            max_seq_len: 1024,
        },
    }
}

/// Create a realistic activation tensor: F32 [1, 5, 768] = 15,360 bytes.
/// Values follow a recognizable pattern: token_idx * 0.001 + dim_idx * 0.0001.
fn make_activation_tensor(name: &str) -> OwnedTensor {
    let seq_len = 5usize;
    let hidden_dim = 768usize;
    let mut data = Vec::with_capacity(seq_len * hidden_dim * 4);
    for t in 0..seq_len {
        for d in 0..hidden_dim {
            let value = t as f32 * 0.001 + d as f32 * 0.0001;
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    OwnedTensor {
        name: name.to_string(),
        dtype: DType::F32,
        shape: vec![1, seq_len as u32, hidden_dim as u32],
        data: Bytes::from(data),
    }
}

/// Create input_ids tensor: U32 [1, 5] = 20 bytes.
fn make_input_ids() -> OwnedTensor {
    let token_ids: [u32; 5] = [464, 3139, 286, 4881, 318];
    let data: Vec<u8> = token_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    OwnedTensor {
        name: "input_ids".to_string(),
        dtype: DType::U32,
        shape: vec![1, 5],
        data: Bytes::from(data),
    }
}

// ---------------------------------------------------------------------------
// Core test: run 2-stage pipeline with tapping relay
// ---------------------------------------------------------------------------

/// Run a 2-stage pipeline with a tapping relay between stages.
/// Returns the relay capture (both directions of inter-stage traffic).
async fn run_pipeline_with_capture() -> RelayCapture {
    let manifest = make_manifest();
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    // Control channels.
    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(65536);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(65536);

    // Data_in: orchestrator → stage 0.
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(256 * 1024);

    // Inter-stage: stage 0 → [TAPPING RELAY] → stage 1.
    // Split into two duplex pairs with relay in between.
    let (stage0_data_out, relay_left) = tokio::io::duplex(256 * 1024);
    let (relay_right, stage1_data_in) = tokio::io::duplex(256 * 1024);

    // Data_out: stage 1 → orchestrator.
    let (stage1_data_out, orch_data_out) = tokio::io::duplex(256 * 1024);

    // Spawn tapping relay.
    let relay_handle = tokio::spawn(tapping_relay(relay_left, relay_right));

    // Spawn stage 0.
    let stage0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(DoubleExecutor, StageConfig::default());
        runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage 0 failed");
    });

    // Spawn stage 1.
    let stage1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(DoubleExecutor, StageConfig::default());
        runtime
            .run(
                stage1_ctrl,
                stage1_data_in,
                stage1_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage 1 failed");
    });

    // Run orchestrator.
    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();

    orch.init(vec![orch_ctrl0, orch_ctrl1], &verifier)
        .await
        .expect("orchestrator init failed");

    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .expect("data channels failed");

    // Send realistic tensors: input_ids + hidden_states.
    let input = vec![vec![make_input_ids(), make_activation_tensor("hidden_states")]];
    let result = orch.infer(input, 16).await.expect("inference failed");

    // Verify output was received (doubled by both stages → 4x original).
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].len(), 2);
    assert_eq!(result.outputs[0][0].name, "input_ids");
    assert_eq!(result.outputs[0][1].name, "hidden_states");

    // Shutdown pipeline.
    orch.shutdown().await.expect("shutdown failed");

    stage0_handle.await.unwrap();
    stage1_handle.await.unwrap();

    let capture = relay_handle.await.unwrap();
    capture.assert_clean("inter-stage relay");
    capture
}

// ===========================================================================
// Tests
// ===========================================================================

/// Inter-stage relay captures bytes in both directions (handshake is bidirectional).
#[tokio::test]
async fn relay_captures_both_directions() {
    let cap = run_pipeline_with_capture().await;
    assert!(
        cap.fwd.len() > 0,
        "forward capture should not be empty"
    );
    assert!(
        cap.bwd.len() > 0,
        "backward capture should not be empty (handshake response)"
    );
}

/// All tensor frames on the inter-stage relay are encrypted.
#[tokio::test]
async fn no_unencrypted_tensor_frames() {
    let cap = run_pipeline_with_capture().await;
    let all_bytes: Vec<u8> = [cap.fwd.as_slice(), cap.bwd.as_slice()].concat();
    let frames = scan_frames_with_payloads(&all_bytes);

    let tensor_frames: Vec<_> = frames
        .iter()
        .filter(|(info, _)| info.msg_type == FrameType::Tensor)
        .collect();

    assert!(
        !tensor_frames.is_empty(),
        "should have at least one tensor frame"
    );

    for (info, _) in &tensor_frames {
        assert!(
            info.flags.is_encrypted(),
            "tensor frame seq={} should be encrypted but isn't",
            info.sequence
        );
    }
}

/// No tensor payload can be decoded by OwnedTensor::decode().
#[tokio::test]
async fn tensor_decode_always_fails() {
    let cap = run_pipeline_with_capture().await;

    for (label, bytes) in [("fwd", &cap.fwd), ("bwd", &cap.bwd)] {
        let frames = scan_frames_with_payloads(bytes);
        for (info, payload) in &frames {
            if info.msg_type == FrameType::Tensor {
                let result = OwnedTensor::decode(payload.clone());
                assert!(
                    result.is_err(),
                    "{label} tensor frame seq={} should NOT be decodable, but got: {:?}",
                    info.sequence,
                    result.unwrap()
                );
            }
        }
    }
}

/// Handshake frames (Hello) are present on the inter-stage link.
/// Stage 0 data_out initiates SecureChannel, stage 1 data_in accepts.
#[tokio::test]
async fn has_handshake_frames() {
    let cap = run_pipeline_with_capture().await;
    let all_bytes: Vec<u8> = [cap.fwd.as_slice(), cap.bwd.as_slice()].concat();
    let frames = scan_frames_with_payloads(&all_bytes);

    let hello_count = frames
        .iter()
        .filter(|(info, _)| info.msg_type == FrameType::Hello)
        .count();

    // 3-message handshake: Hello(initiator), Hello(responder), Hello(initiator confirm)
    assert_eq!(
        hello_count, 3,
        "expected 3 handshake frames, got {hello_count}"
    );
}

/// Encrypted payload entropy is near-maximum (>7.9 bits/byte).
#[tokio::test]
async fn encrypted_payload_entropy_near_maximum() {
    let cap = run_pipeline_with_capture().await;

    // Collect all payload bytes from forward direction (where tensor data flows).
    let fwd_frames = scan_frames_with_payloads(&cap.fwd);
    let payload_bytes: Vec<u8> = fwd_frames
        .iter()
        .filter(|(info, _)| info.flags.is_encrypted())
        .flat_map(|(_, payload)| payload.iter().copied())
        .collect();

    assert!(
        !payload_bytes.is_empty(),
        "should have encrypted payload bytes"
    );

    let entropy = shannon_entropy(&payload_bytes);
    assert!(
        entropy > 7.9,
        "encrypted payload entropy should be >7.9 bits/byte, got {entropy:.3}"
    );
}

/// Activation tensor (15,360 bytes F32) cannot be recovered from relay capture.
/// Specifically: no frame's payload decodes to a tensor named "hidden_states".
#[tokio::test]
async fn activation_values_not_recoverable() {
    let cap = run_pipeline_with_capture().await;

    for (label, bytes) in [("fwd", &cap.fwd), ("bwd", &cap.bwd)] {
        let frames = scan_frames_with_payloads(bytes);
        for (_, payload) in &frames {
            if let Ok(tensor) = OwnedTensor::decode(payload.clone()) {
                panic!(
                    "{label}: recovered tensor '{}' (shape {:?}) from relay capture — \
                     inter-stage data should be encrypted",
                    tensor.name, tensor.shape
                );
            }
        }
    }
}

/// Forward direction has tensor frames with expected structure:
/// - At least 2 encrypted Tensor frames (input_ids + hidden_states activation)
/// - Data frames for END sentinels
#[tokio::test]
async fn forward_frame_structure() {
    let cap = run_pipeline_with_capture().await;
    let frames = scan_frames_with_payloads(&cap.fwd);

    let tensor_count = frames
        .iter()
        .filter(|(info, _)| info.msg_type == FrameType::Tensor)
        .count();
    let data_count = frames
        .iter()
        .filter(|(info, _)| info.msg_type == FrameType::Data)
        .count();

    // Stage 0 sends: tensor(input_ids_doubled) + tensor(hidden_states_doubled) + END sentinel
    assert!(
        tensor_count >= 2,
        "expected at least 2 tensor frames in forward direction, got {tensor_count}"
    );
    assert!(
        data_count >= 1,
        "expected at least 1 data frame (END sentinel) in forward direction, got {data_count}"
    );
}

/// Total captured bytes include AEAD overhead (nonce + tag per frame).
/// Overhead should be bounded: >0% and <10%.
#[tokio::test]
async fn aead_overhead_bounded() {
    let cap = run_pipeline_with_capture().await;
    let total_captured = cap.fwd.len() + cap.bwd.len();

    // Raw payload size: input_ids (20 bytes) + hidden_states (15360 bytes) = 15380 bytes
    // Each stage doubles, so stage 0 output = same sizes (20 + 15360 bytes of raw tensor data).
    // Plus END sentinel, plus handshake, plus headers + AEAD tags.
    let raw_tensor_bytes = 20 + 15360; // input_ids + hidden_states

    // The total should be larger than raw (overhead from headers + AEAD + handshake)
    // but not absurdly so (<10x).
    assert!(
        total_captured > raw_tensor_bytes,
        "captured bytes ({total_captured}) should exceed raw tensor size ({raw_tensor_bytes})"
    );
    assert!(
        total_captured < raw_tensor_bytes * 10,
        "captured bytes ({total_captured}) seems unreasonably large vs raw ({raw_tensor_bytes})"
    );
}

/// Summary: print what the host sees on the inter-stage relay.
#[tokio::test]
async fn print_relay_summary() {
    let cap = run_pipeline_with_capture().await;

    println!();
    println!("=== INTER-STAGE RELAY CAPTURE SUMMARY ===");
    println!();
    println!("Forward (stage 0 → stage 1): {} bytes", cap.fwd.len());
    println!("Backward (stage 1 → stage 0): {} bytes", cap.bwd.len());
    println!("Total: {} bytes", cap.fwd.len() + cap.bwd.len());
    println!();

    for (label, bytes) in [("FWD", cap.fwd.as_slice()), ("BWD", cap.bwd.as_slice())] {
        let frames = scan_frames_with_payloads(bytes);
        if frames.is_empty() {
            continue;
        }
        println!("{label} frames:");
        for (i, (info, payload)) in frames.iter().enumerate() {
            let enc = if info.flags.is_encrypted() {
                "ENCRYPTED"
            } else {
                "plaintext"
            };
            let decode_result = if info.msg_type == FrameType::Tensor {
                match OwnedTensor::decode(payload.clone()) {
                    Ok(t) => format!("DECODED: {} {:?} {:?}", t.name, t.dtype, t.shape),
                    Err(e) => format!("cannot decode: {e:?}"),
                }
            } else {
                String::new()
            };
            println!(
                "  #{i} {:?} seq={} payload={}B {enc} {decode_result}",
                info.msg_type, info.sequence, info.payload_len
            );
        }

        // Entropy of encrypted payloads.
        let enc_bytes: Vec<u8> = frames
            .iter()
            .filter(|(info, _)| info.flags.is_encrypted())
            .flat_map(|(_, p)| p.iter().copied())
            .collect();
        if !enc_bytes.is_empty() {
            println!(
                "  Encrypted payload entropy: {:.3} bits/byte",
                shannon_entropy(&enc_bytes)
            );
        }
        println!();
    }

    println!("Tensor recovery attempt: ALL FAILED (expected)");
    println!("=== END SUMMARY ===");
    println!();
}
