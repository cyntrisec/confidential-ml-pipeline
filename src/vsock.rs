use tokio_vsock::{VsockAddr, VsockListener, VsockStream, VMADDR_CID_ANY};
use tracing::{debug, info};

use confidential_ml_transport::{AttestationProvider, AttestationVerifier, RetryPolicy};

use crate::error::PipelineError;
use crate::executor::StageExecutor;
use crate::manifest::PortSpec;
use crate::manifest::ShardManifest;
use crate::orchestrator::{Orchestrator, OrchestratorConfig};
use crate::stage::{StageConfig, StageRuntime};

/// Resolve a [`PortSpec`] to a `(cid, port)` pair.
///
/// Returns an error if the spec is not a VSock address.
pub fn resolve_vsock(spec: &PortSpec) -> crate::error::Result<(u32, u32)> {
    match spec {
        PortSpec::VSock { cid, port } => Ok((*cid, *port)),
        other => Err(PipelineError::Protocol(format!(
            "expected VSock port spec, got {other:?}"
        ))),
    }
}

/// Connect to a VSock address with retry and exponential backoff.
pub async fn connect_vsock_retry(
    cid: u32,
    port: u32,
    policy: &RetryPolicy,
) -> crate::error::Result<VsockStream> {
    for attempt in 0..=policy.max_retries {
        match VsockStream::connect(VsockAddr::new(cid, port)).await {
            Ok(stream) => {
                debug!(cid, port, attempt, "VSock connected");
                return Ok(stream);
            }
            Err(e) if attempt < policy.max_retries => {
                let delay = policy.delay_for_attempt(attempt);
                debug!(cid, port, attempt, error = %e, delay_ms = delay.as_millis(), "VSock connect retry");
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                return Err(PipelineError::Io(e));
            }
        }
    }
    unreachable!()
}

/// Bind VSock listeners for a stage's control and data_in ports.
///
/// Returns `(control_listener, data_in_listener)`.
/// Binds to `VMADDR_CID_ANY` so the enclave accepts connections from any CID.
pub fn bind_stage_listeners_vsock(
    ctrl_port: u32,
    din_port: u32,
) -> crate::error::Result<(VsockListener, VsockListener)> {
    let ctrl_listener = VsockListener::bind(VsockAddr::new(VMADDR_CID_ANY, ctrl_port))
        .map_err(PipelineError::Io)?;

    let din_listener =
        VsockListener::bind(VsockAddr::new(VMADDR_CID_ANY, din_port)).map_err(PipelineError::Io)?;

    info!(
        ctrl_port,
        data_in_port = din_port,
        "stage VSock listeners bound"
    );
    Ok((ctrl_listener, din_listener))
}

/// Run a pipeline stage using pre-bound VSock listeners.
///
/// Flow:
/// 1. Accept control VSock connection
/// 2. Run control phase (Init / Ready / EstablishDataChannels)
/// 3. Concurrently: accept data_in VSock + connect data_out VSock
/// 4. Run data phase (crypto handshakes + process loop)
#[allow(clippy::too_many_arguments)]
pub async fn run_stage_with_listeners_vsock<E: StageExecutor>(
    executor: E,
    config: StageConfig,
    control_listener: VsockListener,
    data_in_listener: VsockListener,
    data_out_cid: u32,
    data_out_port: u32,
    provider: &dyn AttestationProvider,
    verifier: &dyn AttestationVerifier,
) -> crate::error::Result<()> {
    // 1. Accept control connection.
    let (ctrl_stream, ctrl_peer) = control_listener.accept().await.map_err(PipelineError::Io)?;
    info!(peer = ?ctrl_peer, "stage: accepted control VSock");

    // Clone retry policy before config is moved into the runtime.
    let retry_policy = config.tcp_retry_policy.clone();

    // 2. Control phase.
    let mut runtime = StageRuntime::new(executor, config);
    let result = runtime.run_control_phase(ctrl_stream, provider, verifier).await?;

    // 3. Concurrently accept data_in and connect data_out.
    let (din_result, dout_result) = tokio::try_join!(
        accept_vsock(&data_in_listener),
        connect_vsock_retry(data_out_cid, data_out_port, &retry_policy),
    )?;

    info!("stage: VSock data transports connected");

    // 4. Data phase.
    runtime
        .run_data_phase(result.control, din_result, dout_result, provider, verifier)
        .await
}

/// Initialize an orchestrator over VSock connections.
///
/// The `data_out_listener` must already be bound; its port should be
/// communicated to the last stage as that stage's `data_out_target`.
///
/// For multi-stage pipelines, the host relays inter-stage traffic because
/// enclave-to-enclave VSock is not supported. Relay listeners are bound
/// on the ports specified by each non-final stage's `endpoint.data_out`.
///
/// Flow:
/// 1. VSock connect to each stage's control port
/// 2. `orch.init()` — handshake + Init/Ready on all control channels
/// 3. `orch.send_establish_data_channels()`
/// 4. Bind relay listeners for inter-stage data
/// 5. Concurrently: connect data_in, accept data_out, establish relay links
/// 6. `orch.complete_data_channels()`
pub async fn init_orchestrator_vsock(
    config: OrchestratorConfig,
    manifest: ShardManifest,
    data_out_listener: VsockListener,
    verifier: &dyn AttestationVerifier,
    provider: &dyn AttestationProvider,
) -> crate::error::Result<Orchestrator<VsockStream>> {
    let num_stages = manifest.stages.len();

    // Clone retry policy before config is moved into the orchestrator.
    let retry_policy = config.tcp_retry_policy.clone();

    // 1. Connect control channels to all stages.
    let mut ctrl_streams = Vec::with_capacity(num_stages);
    for (i, stage) in manifest.stages.iter().enumerate() {
        let (cid, port) = resolve_vsock(&stage.endpoint.control)?;
        let stream = connect_vsock_retry(cid, port, &retry_policy).await?;
        info!(
            stage = i,
            cid, port, "orchestrator: control VSock connected"
        );
        ctrl_streams.push(stream);
    }

    // 2. Init.
    let mut orch = Orchestrator::new(config, manifest)?;
    orch.init(ctrl_streams, provider, verifier).await?;

    // 3. Send EstablishDataChannels.
    orch.send_establish_data_channels().await?;

    // 4. Bind relay listeners for inter-stage data (host relays because
    //    enclave-to-enclave VSock is not supported).
    let mut relay_listeners = Vec::new();
    for i in 0..num_stages.saturating_sub(1) {
        let (_, relay_port) = resolve_vsock(&orch.manifest().stages[i].endpoint.data_out)?;
        let listener = VsockListener::bind(VsockAddr::new(VMADDR_CID_ANY, relay_port))
            .map_err(PipelineError::Io)?;
        info!(stage = i, relay_port, "orchestrator: relay listener bound");
        relay_listeners.push(listener);
    }

    // Collect downstream addresses (stage[i+1].data_in) for relay connections.
    let mut relay_downstream_addrs = Vec::new();
    for i in 1..num_stages {
        relay_downstream_addrs.push(resolve_vsock(&orch.manifest().stages[i].endpoint.data_in)?);
    }

    // 5. Concurrently connect data endpoints and establish relay links.
    let (stage0_cid, stage0_din_port) = resolve_vsock(&orch.manifest().stages[0].endpoint.data_in)?;

    let relay_policy = retry_policy.clone();
    let relay_fut = async {
        let mut handles = Vec::new();
        for (i, listener) in relay_listeners.iter().enumerate() {
            let (cid, port) = relay_downstream_addrs[i];
            let (upstream, downstream) = tokio::try_join!(
                accept_vsock(listener),
                connect_vsock_retry(cid, port, &relay_policy),
            )?;
            info!(
                upstream_stage = i,
                downstream_stage = i + 1,
                "orchestrator: relay link established"
            );
            handles.push(crate::relay::start_relay_link(upstream, downstream));
        }
        Ok::<Vec<crate::relay::RelayHandle>, PipelineError>(handles)
    };

    let (din_stream, dout_stream, relay_handles) = tokio::try_join!(
        connect_vsock_retry(stage0_cid, stage0_din_port, &retry_policy),
        accept_vsock(&data_out_listener),
        relay_fut,
    )?;

    info!("orchestrator: all VSock data transports connected");

    // 6. Complete data channels.
    orch.complete_data_channels(din_stream, dout_stream, relay_handles, provider, verifier)
        .await?;

    Ok(orch)
}

/// Accept a single VSock connection from a listener.
async fn accept_vsock(listener: &VsockListener) -> crate::error::Result<VsockStream> {
    let (stream, peer) = listener.accept().await.map_err(PipelineError::Io)?;
    debug!(peer = ?peer, "VSock accepted");
    Ok(stream)
}
