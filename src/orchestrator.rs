use std::time::Duration;

use bytes::Bytes;
use confidential_ml_transport::{
    AttestationVerifier, Message, OwnedTensor, SecureChannel, SessionConfig,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, info, warn};

use crate::error::PipelineError;
use crate::manifest::ShardManifest;
use crate::protocol::{OrchestratorMsg, StageMsg};
use crate::relay::RelayHandle;
use crate::stage::ERROR_SENTINEL;

/// Configuration for the orchestrator.
pub struct OrchestratorConfig {
    pub session_config: SessionConfig,
    /// Timeout for health-check pings (default: 10 seconds).
    pub health_check_timeout: Duration,
    /// Timeout for a single inference request (default: 60 seconds).
    pub infer_timeout: Duration,
    /// Retry policy for TCP connections (used by TCP helpers).
    pub tcp_retry_policy: confidential_ml_transport::RetryPolicy,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            session_config: SessionConfig::default(),
            health_check_timeout: Duration::from_secs(10),
            infer_timeout: Duration::from_secs(60),
            tcp_retry_policy: confidential_ml_transport::RetryPolicy::default(),
        }
    }
}

/// Result of an inference request.
#[derive(Debug)]
pub struct InferenceResult {
    /// Output tensors from the final stage, grouped by micro-batch.
    pub outputs: Vec<Vec<OwnedTensor>>,
}

/// Handle to a connected stage.
struct StageHandle<T> {
    stage_idx: usize,
    control: SecureChannel<T>,
}

/// Host-side pipeline controller.
///
/// Connects to all stages, verifies attestation, manages the pipeline lifecycle,
/// and dispatches inference requests.
pub struct Orchestrator<T> {
    config: OrchestratorConfig,
    manifest: ShardManifest,
    stages: Vec<StageHandle<T>>,
    relay_handles: Vec<RelayHandle>,
    data_in: Option<SecureChannel<T>>,
    data_out: Option<SecureChannel<T>>,
}

impl<T: AsyncRead + AsyncWrite + Unpin + Send> Orchestrator<T> {
    pub fn new(config: OrchestratorConfig, manifest: ShardManifest) -> crate::error::Result<Self> {
        manifest.validate()?;
        Ok(Self {
            config,
            manifest,
            stages: Vec::new(),
            relay_handles: Vec::new(),
            data_in: None,
            data_out: None,
        })
    }

    /// Initialize the pipeline: connect control channels, verify attestation,
    /// send Init, and wait for all stages to be Ready.
    pub async fn init(
        &mut self,
        control_transports: Vec<T>,
        verifier: &dyn AttestationVerifier,
    ) -> crate::error::Result<()> {
        let num_stages = self.manifest.stages.len();
        if control_transports.len() != num_stages {
            return Err(PipelineError::Protocol(format!(
                "expected {num_stages} control transports, got {}",
                control_transports.len()
            )));
        }

        info!(num_stages, "orchestrator: connecting control channels");

        for (i, transport) in control_transports.into_iter().enumerate() {
            let mut session_config = self.config.session_config.clone();

            if !self.manifest.stages[i].expected_measurements.is_empty() {
                let measurements =
                    self.manifest.stages[i]
                        .to_expected_measurements()
                        .map_err(|e| {
                            PipelineError::Protocol(format!(
                                "invalid measurements for stage {i}: {e}"
                            ))
                        })?;
                session_config.expected_measurements = Some(measurements);
            }

            let channel =
                SecureChannel::connect_with_attestation(transport, verifier, session_config)
                    .await
                    .map_err(PipelineError::Transport)?;

            info!(stage = i, "orchestrator: control channel established");
            self.stages.push(StageHandle {
                stage_idx: i,
                control: channel,
            });
        }

        let activation_spec_json = serde_json::to_string(&self.manifest.activation_spec)
            .map_err(|e| PipelineError::Protocol(format!("activation_spec serialize: {e}")))?;

        for (i, stage) in self.stages.iter_mut().enumerate() {
            let stage_spec_json = serde_json::to_string(&self.manifest.stages[i])
                .map_err(|e| PipelineError::Protocol(format!("stage_spec serialize: {e}")))?;

            let msg = OrchestratorMsg::Init {
                stage_spec_json,
                activation_spec_json: activation_spec_json.clone(),
                num_stages,
            };

            stage
                .control
                .send(msg.to_bytes())
                .await
                .map_err(PipelineError::Transport)?;
        }

        for stage in &mut self.stages {
            let msg = recv_stage_msg(&mut stage.control).await?;
            match msg {
                StageMsg::Ready { stage_idx } => {
                    info!(stage = stage_idx, "orchestrator: stage ready");
                }
                other => {
                    return Err(PipelineError::Protocol(format!(
                        "expected Ready from stage {}, got {other:?}",
                        stage.stage_idx
                    )));
                }
            }
        }

        info!("orchestrator: all stages initialized");
        Ok(())
    }

    /// Establish data channels between stages.
    ///
    /// This is a convenience method that calls [`send_establish_data_channels`]
    /// followed by [`complete_data_channels`]. For TCP deployments where data
    /// transports must be connected between the two calls, use them separately.
    ///
    /// - `data_in_transport`: orchestrator connects (initiator) to stage 0's data_in (responder)
    /// - `data_out_transport`: last stage connects (initiator) to orchestrator's acceptor (responder)
    /// - `provider`: attestation provider for the orchestrator's responder role on data_out
    pub async fn establish_data_channels(
        &mut self,
        data_in_transport: T,
        data_out_transport: T,
        relay_handles: Vec<RelayHandle>,
        verifier: &dyn AttestationVerifier,
        provider: &dyn confidential_ml_transport::AttestationProvider,
    ) -> crate::error::Result<()> {
        self.send_establish_data_channels().await?;
        self.complete_data_channels(
            data_in_transport,
            data_out_transport,
            relay_handles,
            verifier,
            provider,
        )
        .await
    }

    /// Send EstablishDataChannels to all stages.
    ///
    /// After this returns, every stage is waiting for its data transports to
    /// connect. The caller should then provide the actual TCP/VSock transports
    /// and call [`complete_data_channels`].
    pub async fn send_establish_data_channels(&mut self) -> crate::error::Result<()> {
        let num_stages = self.stages.len();

        for (i, stage) in self.stages.iter_mut().enumerate() {
            let msg = OrchestratorMsg::EstablishDataChannels {
                has_upstream: i > 0,
                has_downstream: i < num_stages - 1,
            };
            stage
                .control
                .send(msg.to_bytes())
                .await
                .map_err(PipelineError::Transport)?;
        }

        info!("orchestrator: sent EstablishDataChannels to all stages");
        Ok(())
    }

    /// Complete data channel establishment after transports have been connected.
    ///
    /// Must be called after [`send_establish_data_channels`].
    pub async fn complete_data_channels(
        &mut self,
        data_in_transport: T,
        data_out_transport: T,
        relay_handles: Vec<RelayHandle>,
        verifier: &dyn AttestationVerifier,
        provider: &dyn confidential_ml_transport::AttestationProvider,
    ) -> crate::error::Result<()> {
        self.relay_handles = relay_handles;

        // Connect data_in to stage 0 (orchestrator = initiator, stage 0 = responder).
        self.data_in = Some(
            SecureChannel::connect_with_attestation(
                data_in_transport,
                verifier,
                self.config.session_config.clone(),
            )
            .await
            .map_err(PipelineError::Transport)?,
        );

        // Accept data_out from last stage (last stage = initiator, orchestrator = responder).
        self.data_out = Some(
            SecureChannel::accept_with_attestation(
                data_out_transport,
                provider,
                self.config.session_config.clone(),
            )
            .await
            .map_err(PipelineError::Transport)?,
        );

        for stage in &mut self.stages {
            let msg = recv_stage_msg(&mut stage.control).await?;
            match msg {
                StageMsg::DataChannelsReady { stage_idx } => {
                    info!(stage = stage_idx, "orchestrator: data channels ready");
                }
                other => {
                    return Err(PipelineError::Protocol(format!(
                        "expected DataChannelsReady from stage {}, got {other:?}",
                        stage.stage_idx
                    )));
                }
            }
        }

        info!("orchestrator: all data channels established");
        Ok(())
    }

    /// Access the shard manifest.
    pub fn manifest(&self) -> &ShardManifest {
        &self.manifest
    }

    /// Run an inference request through the pipeline.
    ///
    /// Sends input tensors to stage 0, receives output tensors from the last stage.
    /// If a stage fails, it sends an error sentinel on the data channel, which
    /// unblocks the output receiver. The orchestrator then reads the actual error
    /// from the control channel.
    ///
    /// Subject to `OrchestratorConfig::infer_timeout`.
    pub async fn infer(
        &mut self,
        input_tensors: Vec<Vec<OwnedTensor>>,
        seq_len: u32,
    ) -> crate::error::Result<InferenceResult> {
        let timeout = self.config.infer_timeout;
        tokio::time::timeout(timeout, self.infer_inner(input_tensors, seq_len))
            .await
            .map_err(|_| PipelineError::Timeout("inference timed out".into()))?
    }

    async fn infer_inner(
        &mut self,
        input_tensors: Vec<Vec<OwnedTensor>>,
        seq_len: u32,
    ) -> crate::error::Result<InferenceResult> {
        let request_id = rand_request_id();
        let num_micro_batches = input_tensors.len() as u32;

        if num_micro_batches == 0 {
            return Ok(InferenceResult {
                outputs: Vec::new(),
            });
        }

        let data_in = self
            .data_in
            .as_mut()
            .ok_or_else(|| PipelineError::Protocol("data channels not established".into()))?;
        let data_out = self
            .data_out
            .as_mut()
            .ok_or_else(|| PipelineError::Protocol("data channels not established".into()))?;

        // Send StartRequest to all stages.
        for stage in &mut self.stages {
            let msg = OrchestratorMsg::StartRequest {
                request_id,
                num_micro_batches,
                seq_len,
            };
            stage
                .control
                .send(msg.to_bytes())
                .await
                .map_err(PipelineError::Transport)?;
        }

        debug!(
            request_id,
            num_micro_batches, "orchestrator: sending input tensors"
        );

        // Send input tensors to stage 0.
        for mb_tensors in &input_tensors {
            for t in mb_tensors {
                data_in
                    .send_tensor(t.as_ref())
                    .await
                    .map_err(PipelineError::Transport)?;
            }
            data_in
                .send(Bytes::from_static(b"END"))
                .await
                .map_err(PipelineError::Transport)?;
        }

        // Receive output tensors from last stage.
        // If a stage failed, it sends an ERR sentinel on its data_out, which
        // propagates through relays and surfaces here as a StageFailed error.
        let output_result = receive_all_outputs(data_out, num_micro_batches).await;

        match output_result {
            Ok(outputs) => {
                // Success: collect RequestDone confirmations from all stages.
                for stage in &mut self.stages {
                    let msg = recv_stage_msg(&mut stage.control).await?;
                    match msg {
                        StageMsg::RequestDone { request_id: rid } if rid == request_id => {
                            debug!(stage = stage.stage_idx, "orchestrator: stage done");
                        }
                        StageMsg::RequestError {
                            request_id: rid,
                            error,
                        } if rid == request_id => {
                            return Err(PipelineError::RequestFailed {
                                request_id,
                                reason: format!("stage {} error: {}", stage.stage_idx, error),
                            });
                        }
                        other => {
                            return Err(PipelineError::Protocol(format!(
                                "expected RequestDone/RequestError for {request_id} from stage {}, got {other:?}",
                                stage.stage_idx
                            )));
                        }
                    }
                }

                info!(request_id, "orchestrator: inference complete");
                Ok(InferenceResult { outputs })
            }
            Err(PipelineError::StageFailed { .. }) => {
                // A stage sent an error sentinel. Read control channels for details.
                for stage in &mut self.stages {
                    let msg = recv_stage_msg(&mut stage.control).await?;
                    if let StageMsg::RequestError {
                        request_id: rid,
                        error,
                    } = msg
                    {
                        if rid == request_id {
                            return Err(PipelineError::RequestFailed {
                                request_id,
                                reason: format!("stage {} error: {}", stage.stage_idx, error),
                            });
                        }
                    }
                }
                // If no stage reported an error explicitly, return generic failure.
                Err(PipelineError::RequestFailed {
                    request_id,
                    reason: "stage failed (no error details on control channel)".into(),
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Send a health-check ping to all stages.
    ///
    /// Subject to `OrchestratorConfig::health_check_timeout`.
    pub async fn health_check(&mut self) -> crate::error::Result<()> {
        let timeout = self.config.health_check_timeout;
        tokio::time::timeout(timeout, self.health_check_inner())
            .await
            .map_err(|_| PipelineError::Timeout("health check timed out".into()))?
    }

    async fn health_check_inner(&mut self) -> crate::error::Result<()> {
        let seq = rand_request_id();

        for stage in &mut self.stages {
            stage
                .control
                .send(OrchestratorMsg::Ping { seq }.to_bytes())
                .await
                .map_err(PipelineError::Transport)?;
        }

        for stage in &mut self.stages {
            let msg = recv_stage_msg(&mut stage.control).await?;
            match msg {
                StageMsg::Pong { seq: s } if s == seq => {
                    debug!(stage = stage.stage_idx, "health check OK");
                }
                other => {
                    return Err(PipelineError::StageFailed {
                        stage_idx: stage.stage_idx,
                        reason: format!("expected Pong, got {other:?}"),
                    });
                }
            }
        }

        for (i, relay) in self.relay_handles.iter().enumerate() {
            if relay.is_finished() {
                warn!(relay = i, "relay link has terminated");
            }
        }

        Ok(())
    }

    /// Gracefully shut down all stages.
    pub async fn shutdown(&mut self) -> crate::error::Result<()> {
        info!("orchestrator: shutting down pipeline");

        for stage in &mut self.stages {
            stage
                .control
                .send(OrchestratorMsg::Shutdown.to_bytes())
                .await
                .map_err(PipelineError::Transport)?;
        }

        for stage in &mut self.stages {
            let msg = recv_stage_msg(&mut stage.control).await?;
            match msg {
                StageMsg::ShuttingDown { stage_idx } => {
                    info!(stage = stage_idx, "stage shut down");
                }
                other => {
                    warn!(
                        stage = stage.stage_idx,
                        "expected ShuttingDown, got {other:?}"
                    );
                }
            }
        }

        for relay in &self.relay_handles {
            relay.abort();
        }

        info!("orchestrator: shutdown complete");
        Ok(())
    }
}

/// Receive all output tensors (all micro-batches) from the data_out channel.
async fn receive_all_outputs<T: AsyncRead + AsyncWrite + Unpin + Send>(
    data_out: &mut SecureChannel<T>,
    num_micro_batches: u32,
) -> crate::error::Result<Vec<Vec<OwnedTensor>>> {
    let mut outputs = Vec::with_capacity(num_micro_batches as usize);
    for mb in 0..num_micro_batches {
        debug!(micro_batch = mb, "orchestrator: receiving output");
        let tensors = recv_output_tensors(data_out).await?;
        outputs.push(tensors);
    }
    Ok(outputs)
}

/// Receive a stage message from a control channel.
async fn recv_stage_msg<T: AsyncRead + AsyncWrite + Unpin + Send>(
    channel: &mut SecureChannel<T>,
) -> crate::error::Result<StageMsg> {
    let msg = channel.recv().await.map_err(PipelineError::Transport)?;
    match msg {
        Message::Data(data) => StageMsg::from_bytes(&data)
            .map_err(|e| PipelineError::Protocol(format!("invalid stage message: {e}"))),
        Message::Shutdown => Err(PipelineError::Shutdown),
        other => Err(PipelineError::Protocol(format!(
            "expected Data on control channel, got {other:?}"
        ))),
    }
}

/// Receive tensors from data_out until END sentinel.
/// Returns `PipelineError::StageFailed` if an error sentinel is received.
async fn recv_output_tensors<T: AsyncRead + AsyncWrite + Unpin + Send>(
    channel: &mut SecureChannel<T>,
) -> crate::error::Result<Vec<OwnedTensor>> {
    let mut tensors = Vec::new();
    loop {
        let msg = channel.recv().await.map_err(PipelineError::Transport)?;
        match msg {
            Message::Tensor(t) => tensors.push(t),
            Message::Data(data) if data.as_ref() == b"END" => break,
            Message::Data(data) if data.as_ref() == ERROR_SENTINEL => {
                return Err(PipelineError::StageFailed {
                    stage_idx: 0,
                    reason: "stage reported error on data channel".into(),
                });
            }
            Message::Shutdown => return Err(PipelineError::Shutdown),
            other => {
                return Err(PipelineError::Protocol(format!(
                    "unexpected message on data_out: {other:?}"
                )));
            }
        }
    }
    Ok(tensors)
}

/// Generate a pseudo-random request ID.
fn rand_request_id() -> u64 {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    d.as_nanos() as u64
}
