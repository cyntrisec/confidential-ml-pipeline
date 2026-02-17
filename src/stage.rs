use bytes::Bytes;
use confidential_ml_transport::{
    AttestationProvider, AttestationVerifier, Message, OwnedTensor, SecureChannel, SessionConfig,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, error, info, warn};

use crate::error::PipelineError;
use crate::executor::{ForwardOutput, RequestId, StageExecutor};
use crate::manifest::{ActivationSpec, StageSpec};
use crate::protocol::{OrchestratorMsg, StageMsg};
use crate::scheduler::{InferenceSchedule, PipeOp};

/// Sentinel bytes sent on data_out when a stage request fails.
pub(crate) const ERROR_SENTINEL: &[u8] = b"ERR";

/// Configuration for a stage runtime.
#[derive(Default)]
pub struct StageConfig {
    pub session_config: SessionConfig,
    /// Retry policy for TCP connections (used by TCP helpers).
    pub tcp_retry_policy: confidential_ml_transport::RetryPolicy,
}

/// Result of the control-phase handshake.
///
/// Returned by [`StageRuntime::run_control_phase`] so that callers (e.g. TCP
/// helpers) can inspect the negotiated state before supplying data transports.
pub struct ControlPhaseResult<T> {
    /// The established control channel.
    pub control: SecureChannel<T>,
    /// Whether the orchestrator indicated this stage has an upstream data link.
    pub has_upstream: bool,
    /// Whether the orchestrator indicated this stage has a downstream data link.
    pub has_downstream: bool,
}

/// Runtime environment for a single pipeline stage (runs inside an enclave).
///
/// Accepts three SecureChannel connections:
/// - **control**: accepted from the orchestrator (responder role)
/// - **data_in**: accepted from upstream stage or orchestrator (responder role)
/// - **data_out**: initiated to downstream stage or orchestrator (initiator role)
pub struct StageRuntime<E: StageExecutor> {
    executor: E,
    config: StageConfig,
    stage_idx: usize,
    num_stages: usize,
    stage_spec: Option<StageSpec>,
    activation_spec: Option<ActivationSpec>,
}

impl<E: StageExecutor> StageRuntime<E> {
    pub fn new(executor: E, config: StageConfig) -> Self {
        Self {
            executor,
            config,
            stage_idx: 0,
            num_stages: 0,
            stage_spec: None,
            activation_spec: None,
        }
    }

    /// Run the stage, accepting connections and processing requests until shutdown.
    ///
    /// This is a convenience method that calls [`Self::run_control_phase`] followed by
    /// [`Self::run_data_phase`]. For TCP deployments where transports arrive at
    /// different times, call those two methods separately.
    ///
    /// - `control_transport`: accepted (responder) from orchestrator
    /// - `data_in_transport`: accepted (responder) from upstream or orchestrator
    /// - `data_out_transport`: initiated (initiator) to downstream or orchestrator
    /// - `provider`: attestation provider for accepting channels (responder role)
    /// - `verifier`: attestation verifier for initiating channels (initiator role)
    pub async fn run<CT, DI, DO>(
        &mut self,
        control_transport: CT,
        data_in_transport: DI,
        data_out_transport: DO,
        provider: &dyn AttestationProvider,
        verifier: &dyn AttestationVerifier,
    ) -> crate::error::Result<()>
    where
        CT: AsyncRead + AsyncWrite + Unpin + Send,
        DI: AsyncRead + AsyncWrite + Unpin + Send,
        DO: AsyncRead + AsyncWrite + Unpin + Send,
    {
        let result = self
            .run_control_phase(control_transport, provider, verifier)
            .await?;
        self.run_data_phase(
            result.control,
            data_in_transport,
            data_out_transport,
            provider,
            verifier,
        )
        .await
    }

    /// Phase 1: Accept the control channel, handle Init/Ready, and wait for
    /// EstablishDataChannels.
    ///
    /// Returns a [`ControlPhaseResult`] containing the established control
    /// channel and the upstream/downstream flags from the orchestrator.
    pub async fn run_control_phase<CT>(
        &mut self,
        control_transport: CT,
        provider: &dyn AttestationProvider,
        verifier: &dyn AttestationVerifier,
    ) -> crate::error::Result<ControlPhaseResult<CT>>
    where
        CT: AsyncRead + AsyncWrite + Unpin + Send,
    {
        // Accept control channel (responder with mutual attestation).
        let mut control = SecureChannel::accept_with_attestation(
            control_transport,
            provider,
            verifier,
            self.config.session_config.clone(),
        )
        .await
        .map_err(PipelineError::Transport)?;

        info!("stage: control channel established");

        // Wait for Init.
        let (stage_spec, activation_spec, num_stages) = self.handle_init(&mut control).await?;
        self.stage_idx = stage_spec.stage_idx;
        self.num_stages = num_stages;
        self.stage_spec = Some(stage_spec.clone());
        self.activation_spec = Some(activation_spec);

        // Initialize executor.
        self.executor
            .init(&stage_spec)
            .await
            .map_err(PipelineError::Stage)?;

        // Verify weight hashes if declared in the manifest.
        if !stage_spec.weight_hashes.is_empty() {
            let actual = self.executor.weight_hashes();
            if actual.len() != stage_spec.weight_hashes.len() {
                return Err(PipelineError::StageFailed {
                    stage_idx: stage_spec.stage_idx,
                    reason: format!(
                        "weight hash count mismatch: manifest declares {} hashes, \
                         executor returned {}",
                        stage_spec.weight_hashes.len(),
                        actual.len()
                    ),
                });
            }
            for (i, (expected, got)) in stage_spec
                .weight_hashes
                .iter()
                .zip(actual.iter())
                .enumerate()
            {
                if expected != got {
                    return Err(PipelineError::StageFailed {
                        stage_idx: stage_spec.stage_idx,
                        reason: format!(
                            "weight hash mismatch at index {i}: \
                             expected {expected}, got {got}"
                        ),
                    });
                }
            }
            info!(
                stage = stage_spec.stage_idx,
                "weight hashes verified ({} hashes)",
                actual.len()
            );
        }

        // Send Ready.
        control
            .send(
                StageMsg::Ready {
                    stage_idx: self.stage_idx,
                }
                .to_bytes()?,
            )
            .await
            .map_err(PipelineError::Transport)?;

        info!(stage = self.stage_idx, "stage: ready");

        // Wait for EstablishDataChannels.
        let (has_upstream, has_downstream) =
            self.wait_for_establish_data_channels(&mut control).await?;

        Ok(ControlPhaseResult {
            control,
            has_upstream,
            has_downstream,
        })
    }

    /// Phase 2: Establish data channels and run the processing loop.
    ///
    /// Must be called after [`Self::run_control_phase`] returns. The `control`
    /// channel is the one returned in [`ControlPhaseResult`].
    pub async fn run_data_phase<CT, DI, DO>(
        &self,
        mut control: SecureChannel<CT>,
        data_in_transport: DI,
        data_out_transport: DO,
        provider: &dyn AttestationProvider,
        verifier: &dyn AttestationVerifier,
    ) -> crate::error::Result<()>
    where
        CT: AsyncRead + AsyncWrite + Unpin + Send,
        DI: AsyncRead + AsyncWrite + Unpin + Send,
        DO: AsyncRead + AsyncWrite + Unpin + Send,
    {
        // Build data channel config with this stage's measurements applied.
        let data_session_config = {
            let mut cfg = self.config.session_config.clone();
            if let Some(ref spec) = self.stage_spec {
                if !spec.expected_measurements.is_empty() {
                    cfg.expected_measurements =
                        Some(spec.to_expected_measurements().map_err(|e| {
                            PipelineError::Protocol(format!(
                                "invalid measurements for stage {} data channels: {e}",
                                self.stage_idx
                            ))
                        })?);
                }
            }
            cfg
        };

        // Accept data_in (responder — upstream initiates or orchestrator initiates).
        let mut data_in = SecureChannel::accept_with_attestation(
            data_in_transport,
            provider,
            verifier,
            data_session_config.clone(),
        )
        .await
        .map_err(PipelineError::Transport)?;

        // Initiate data_out (initiator — this stage connects to downstream acceptor).
        let mut data_out = SecureChannel::connect_with_attestation(
            data_out_transport,
            provider,
            verifier,
            data_session_config,
        )
        .await
        .map_err(PipelineError::Transport)?;

        // Send DataChannelsReady.
        control
            .send(
                StageMsg::DataChannelsReady {
                    stage_idx: self.stage_idx,
                }
                .to_bytes()?,
            )
            .await
            .map_err(PipelineError::Transport)?;

        info!(stage = self.stage_idx, "stage: data channels ready");

        // Process requests until shutdown.
        self.process_loop(&mut control, &mut data_in, &mut data_out)
            .await
    }

    async fn handle_init<T: AsyncRead + AsyncWrite + Unpin + Send>(
        &self,
        control: &mut SecureChannel<T>,
    ) -> crate::error::Result<(StageSpec, ActivationSpec, usize)> {
        let msg = recv_control(control).await?;
        match msg {
            OrchestratorMsg::Init {
                stage_spec_json,
                activation_spec_json,
                num_stages,
            } => {
                let stage_spec: StageSpec = serde_json::from_str(&stage_spec_json)
                    .map_err(|e| PipelineError::Protocol(format!("invalid stage_spec: {e}")))?;
                let activation_spec: ActivationSpec = serde_json::from_str(&activation_spec_json)
                    .map_err(|e| {
                    PipelineError::Protocol(format!("invalid activation_spec: {e}"))
                })?;
                Ok((stage_spec, activation_spec, num_stages))
            }
            other => Err(PipelineError::Protocol(format!(
                "expected Init, got {other:?}"
            ))),
        }
    }

    async fn wait_for_establish_data_channels<T: AsyncRead + AsyncWrite + Unpin + Send>(
        &self,
        control: &mut SecureChannel<T>,
    ) -> crate::error::Result<(bool, bool)> {
        loop {
            let msg = recv_control(control).await?;
            match msg {
                OrchestratorMsg::EstablishDataChannels {
                    has_upstream,
                    has_downstream,
                } => return Ok((has_upstream, has_downstream)),
                OrchestratorMsg::Ping { seq } => {
                    control
                        .send(StageMsg::Pong { seq }.to_bytes()?)
                        .await
                        .map_err(PipelineError::Transport)?;
                    // Continue looping; the next message should be EstablishDataChannels.
                }
                other => {
                    return Err(PipelineError::Protocol(format!(
                        "expected EstablishDataChannels, got {other:?}"
                    )));
                }
            }
        }
    }

    async fn process_loop<CT, DI, DO>(
        &self,
        control: &mut SecureChannel<CT>,
        data_in: &mut SecureChannel<DI>,
        data_out: &mut SecureChannel<DO>,
    ) -> crate::error::Result<()>
    where
        CT: AsyncRead + AsyncWrite + Unpin + Send,
        DI: AsyncRead + AsyncWrite + Unpin + Send,
        DO: AsyncRead + AsyncWrite + Unpin + Send,
    {
        loop {
            let msg = recv_control(control).await?;
            match msg {
                OrchestratorMsg::StartRequest {
                    request_id,
                    num_micro_batches,
                    seq_len,
                } => {
                    if let Some(ref spec) = self.activation_spec {
                        if seq_len > spec.max_seq_len {
                            error!(
                                stage = self.stage_idx, seq_len,
                                max = spec.max_seq_len,
                                "seq_len exceeds max_seq_len"
                            );
                            let _ = data_out.send(Bytes::from_static(ERROR_SENTINEL)).await;
                            control
                                .send(
                                    StageMsg::RequestError {
                                        request_id,
                                        error: format!(
                                            "seq_len {} exceeds max_seq_len {}",
                                            seq_len, spec.max_seq_len
                                        ),
                                    }
                                    .to_bytes()?,
                                )
                                .await
                                .map_err(PipelineError::Transport)?;
                            continue;
                        }
                    }

                    debug!(
                        stage = self.stage_idx,
                        request_id, num_micro_batches, "processing request"
                    );

                    // Run process_request with select! so AbortRequest/Ping can
                    // be handled while the request is in progress.
                    // Scoped so process_fut (which borrows data_in/data_out)
                    // is dropped before the error handler needs data_out.
                    let result = {
                        let process_fut = self.process_request(
                            request_id,
                            num_micro_batches,
                            data_in,
                            data_out,
                        );
                        tokio::pin!(process_fut);

                        let mut early_shutdown = false;
                        let res = loop {
                            tokio::select! {
                                res = &mut process_fut => {
                                    break res;
                                }
                                ctrl_msg = recv_control(control) => {
                                    match ctrl_msg? {
                                        OrchestratorMsg::AbortRequest { request_id: rid, reason } => {
                                            warn!(
                                                stage = self.stage_idx,
                                                request_id = rid, reason,
                                                "request aborted by orchestrator — cancelling"
                                            );
                                            break Err(PipelineError::RequestFailed {
                                                request_id,
                                                reason: format!("aborted: {reason}"),
                                            });
                                        }
                                        OrchestratorMsg::Ping { seq } => {
                                            control
                                                .send(StageMsg::Pong { seq }.to_bytes()?)
                                                .await
                                                .map_err(PipelineError::Transport)?;
                                        }
                                        OrchestratorMsg::Shutdown => {
                                            early_shutdown = true;
                                            break Err(PipelineError::Shutdown);
                                        }
                                        other => {
                                            warn!(
                                                stage = self.stage_idx,
                                                "unexpected control message during request: {other:?}"
                                            );
                                        }
                                    }
                                }
                            }
                        };

                        if early_shutdown {
                            // process_fut dropped here (cancelled).
                            drop(process_fut);
                            info!(stage = self.stage_idx, "shutdown during request");
                            control
                                .send(
                                    StageMsg::ShuttingDown {
                                        stage_idx: self.stage_idx,
                                    }
                                    .to_bytes()?,
                                )
                                .await
                                .map_err(PipelineError::Transport)?;
                            return Ok(());
                        }

                        res
                    }; // process_fut dropped here — data_in/data_out borrows released.

                    match result {
                        Ok(()) => {
                            control
                                .send(StageMsg::RequestDone { request_id }.to_bytes()?)
                                .await
                                .map_err(PipelineError::Transport)?;
                        }
                        Err(e) => {
                            error!(stage = self.stage_idx, request_id, error = %e, "request failed");
                            if let Err(e) = data_out.send(Bytes::from_static(ERROR_SENTINEL)).await
                            {
                                warn!(stage = self.stage_idx, error = %e, "failed to send error sentinel on data_out");
                            }
                            control
                                .send(
                                    StageMsg::RequestError {
                                        request_id,
                                        error: e.to_string(),
                                    }
                                    .to_bytes()?,
                                )
                                .await
                                .map_err(PipelineError::Transport)?;
                        }
                    }
                }
                OrchestratorMsg::AbortRequest { request_id, reason } => {
                    // AbortRequest outside of an active request — nothing to cancel.
                    warn!(
                        stage = self.stage_idx,
                        request_id, reason, "abort received but no request in progress"
                    );
                }
                OrchestratorMsg::Ping { seq } => {
                    control
                        .send(StageMsg::Pong { seq }.to_bytes()?)
                        .await
                        .map_err(PipelineError::Transport)?;
                }
                OrchestratorMsg::Shutdown => {
                    info!(stage = self.stage_idx, "shutting down");
                    control
                        .send(
                            StageMsg::ShuttingDown {
                                stage_idx: self.stage_idx,
                            }
                            .to_bytes()?,
                        )
                        .await
                        .map_err(PipelineError::Transport)?;
                    return Ok(());
                }
                other => {
                    return Err(PipelineError::Protocol(format!(
                        "unexpected message in process loop: {other:?}"
                    )));
                }
            }
        }
    }

    async fn process_request<DI, DO>(
        &self,
        request_id: RequestId,
        num_micro_batches: u32,
        data_in: &mut SecureChannel<DI>,
        data_out: &mut SecureChannel<DO>,
    ) -> crate::error::Result<()>
    where
        DI: AsyncRead + AsyncWrite + Unpin + Send,
        DO: AsyncRead + AsyncWrite + Unpin + Send,
    {
        let schedule = InferenceSchedule::generate(self.num_stages, num_micro_batches)?;
        let stage_schedule = schedule.stage(self.stage_idx).ok_or_else(|| {
            PipelineError::Protocol(format!("no schedule for stage {}", self.stage_idx))
        })?;

        for (step, ops) in stage_schedule.ops.iter().enumerate() {
            debug!(
                stage = self.stage_idx,
                step,
                ops = ?ops,
                "executing step"
            );

            for op in ops {
                match op {
                    PipeOp::RecvActivation { .. } => {}
                    PipeOp::Forward { micro_batch } => {
                        let inputs = recv_tensors(data_in).await?;

                        let output: ForwardOutput = self
                            .executor
                            .forward(request_id, *micro_batch, inputs)
                            .await
                            .map_err(PipelineError::Stage)?;

                        send_tensors(data_out, &output.tensors).await?;
                    }
                    PipeOp::SendActivation { .. } => {}
                    PipeOp::Idle => {}
                }
            }
        }

        Ok(())
    }
}

/// Receive a control message from a SecureChannel.
async fn recv_control<T: AsyncRead + AsyncWrite + Unpin + Send>(
    channel: &mut SecureChannel<T>,
) -> crate::error::Result<OrchestratorMsg> {
    let msg = channel.recv().await.map_err(PipelineError::Transport)?;
    match msg {
        Message::Data(data) => OrchestratorMsg::from_bytes(&data)
            .map_err(|e| PipelineError::Protocol(format!("invalid control message: {e}"))),
        Message::Shutdown => Err(PipelineError::Shutdown),
        other => Err(PipelineError::Protocol(format!(
            "expected Data on control channel, got {other:?}"
        ))),
    }
}

/// Receive tensors from a data channel until END sentinel.
async fn recv_tensors<T: AsyncRead + AsyncWrite + Unpin + Send>(
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
                    stage_idx: usize::MAX,
                    reason: "upstream stage reported error".into(),
                });
            }
            Message::Shutdown => return Err(PipelineError::Shutdown),
            other => {
                return Err(PipelineError::Protocol(format!(
                    "unexpected message on data channel: {other:?}"
                )));
            }
        }
    }
    Ok(tensors)
}

/// Send tensors followed by an END sentinel on a data channel.
async fn send_tensors<T: AsyncRead + AsyncWrite + Unpin + Send>(
    channel: &mut SecureChannel<T>,
    tensors: &[OwnedTensor],
) -> crate::error::Result<()> {
    for t in tensors {
        channel
            .send_tensor(t.as_ref())
            .await
            .map_err(PipelineError::Transport)?;
    }
    channel
        .send(Bytes::from_static(b"END"))
        .await
        .map_err(PipelineError::Transport)?;
    Ok(())
}
