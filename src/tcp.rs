use std::net::SocketAddr;

use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, info};

use confidential_ml_transport::{AttestationProvider, AttestationVerifier, RetryPolicy};

use crate::error::PipelineError;
use crate::executor::StageExecutor;
use crate::manifest::PortSpec;
use crate::manifest::ShardManifest;
use crate::orchestrator::{Orchestrator, OrchestratorConfig};
use crate::stage::{StageConfig, StageRuntime};

/// Resolve a [`PortSpec`] to a [`SocketAddr`].
///
/// Returns an error if the spec is not a TCP address or if parsing fails.
pub fn resolve_tcp(spec: &PortSpec) -> crate::error::Result<SocketAddr> {
    match spec {
        PortSpec::Tcp { addr } => addr
            .parse()
            .map_err(|e| PipelineError::Protocol(format!("invalid TCP address '{addr}': {e}"))),
        other => Err(PipelineError::Protocol(format!(
            "expected TCP port spec, got {other:?}"
        ))),
    }
}

/// Connect to a TCP address with retry and exponential backoff.
///
/// Uses the given [`RetryPolicy`] for backoff delays and attempt limits.
pub async fn connect_tcp_retry(
    addr: SocketAddr,
    policy: &RetryPolicy,
) -> crate::error::Result<TcpStream> {
    for attempt in 0..=policy.max_retries {
        match TcpStream::connect(addr).await {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                debug!(addr = %addr, attempt, "TCP connected");
                return Ok(stream);
            }
            Err(e) if attempt < policy.max_retries => {
                let delay = policy.delay_for_attempt(attempt);
                debug!(addr = %addr, attempt, error = %e, delay_ms = delay.as_millis(), "TCP connect retry");
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                let attempts = attempt + 1;
                return Err(PipelineError::Io(std::io::Error::new(
                    e.kind(),
                    format!("TCP connect to {addr} failed after {attempts} attempt(s): {e}"),
                )));
            }
        }
    }
    unreachable!()
}

/// Bind TCP listeners for a stage's control and data_in ports.
///
/// Returns `(control_listener, control_addr, data_in_listener, data_in_addr)`.
/// Use `0.0.0.0:0` or `127.0.0.1:0` for OS-assigned ports.
pub async fn bind_stage_listeners(
    ctrl_addr: SocketAddr,
    din_addr: SocketAddr,
) -> crate::error::Result<(TcpListener, SocketAddr, TcpListener, SocketAddr)> {
    let ctrl_listener = TcpListener::bind(ctrl_addr)
        .await
        .map_err(PipelineError::Io)?;
    let ctrl_local = ctrl_listener.local_addr().map_err(PipelineError::Io)?;

    let din_listener = TcpListener::bind(din_addr)
        .await
        .map_err(PipelineError::Io)?;
    let din_local = din_listener.local_addr().map_err(PipelineError::Io)?;

    info!(ctrl = %ctrl_local, data_in = %din_local, "stage listeners bound");
    Ok((ctrl_listener, ctrl_local, din_listener, din_local))
}

/// Run a pipeline stage using pre-bound TCP listeners.
///
/// Flow:
/// 1. Accept control TCP connection
/// 2. Run control phase (Init / Ready / EstablishDataChannels)
/// 3. Concurrently: accept data_in TCP + connect data_out TCP
/// 4. Run data phase (crypto handshakes + process loop)
pub async fn run_stage_with_listeners<E: StageExecutor>(
    executor: E,
    config: StageConfig,
    control_listener: TcpListener,
    data_in_listener: TcpListener,
    data_out_target: SocketAddr,
    provider: &dyn AttestationProvider,
    verifier: &dyn AttestationVerifier,
) -> crate::error::Result<()> {
    // 1. Accept control connection.
    let (ctrl_stream, ctrl_peer) = control_listener.accept().await.map_err(PipelineError::Io)?;
    ctrl_stream.set_nodelay(true).ok();
    info!(peer = %ctrl_peer, "stage: accepted control TCP");

    // Clone retry policy before config is moved into the runtime.
    let retry_policy = config.tcp_retry_policy.clone();

    // 2. Control phase.
    let mut runtime = StageRuntime::new(executor, config);
    let result = runtime
        .run_control_phase(ctrl_stream, provider, verifier)
        .await?;

    // 3. Concurrently accept data_in and connect data_out.
    let (din_result, dout_result) = tokio::try_join!(
        accept_tcp(&data_in_listener),
        connect_tcp_retry(data_out_target, &retry_policy),
    )?;

    info!("stage: data transports connected");

    // 4. Data phase.
    runtime
        .run_data_phase(result.control, din_result, dout_result, provider, verifier)
        .await
}

/// Initialize an orchestrator over real TCP connections.
///
/// The `data_out_listener` must already be bound; its address should be
/// communicated to the last stage as that stage's `data_out_target`.
///
/// Flow:
/// 1. TCP connect to each stage's control port
/// 2. `orch.init()` — handshake + Init/Ready on all control channels
/// 3. `orch.send_establish_data_channels()`
/// 4. Concurrently connect data_in to stage 0 + accept data_out from last stage
/// 5. `orch.complete_data_channels()`
pub async fn init_orchestrator_tcp(
    config: OrchestratorConfig,
    manifest: ShardManifest,
    data_out_listener: TcpListener,
    verifier: &dyn AttestationVerifier,
    provider: &dyn AttestationProvider,
) -> crate::error::Result<Orchestrator<TcpStream>> {
    let num_stages = manifest.stages.len();

    // Clone retry policy before config is moved into the orchestrator.
    let retry_policy = config.tcp_retry_policy.clone();

    // 1. Connect control channels to all stages.
    let mut ctrl_streams = Vec::with_capacity(num_stages);
    for (i, stage) in manifest.stages.iter().enumerate() {
        let addr = resolve_tcp(&stage.endpoint.control)?;
        let stream = connect_tcp_retry(addr, &retry_policy).await?;
        info!(stage = i, addr = %addr, "orchestrator: control TCP connected");
        ctrl_streams.push(stream);
    }

    // 2. Init.
    let mut orch = Orchestrator::new(config, manifest)?;
    orch.init(ctrl_streams, provider, verifier).await?;

    // 3. Send EstablishDataChannels.
    orch.send_establish_data_channels().await?;

    // 4. Concurrently connect data_in + accept data_out.
    let stage0_din_addr = resolve_tcp(&orch.manifest().stages[0].endpoint.data_in)?;

    let (din_stream, dout_stream) = tokio::try_join!(
        connect_tcp_retry(stage0_din_addr, &retry_policy),
        accept_tcp(&data_out_listener),
    )?;

    // 5. Complete data channels.
    orch.complete_data_channels(din_stream, dout_stream, vec![], provider, verifier)
        .await?;

    Ok(orch)
}

/// Accept a single TCP connection from a listener.
async fn accept_tcp(listener: &TcpListener) -> crate::error::Result<TcpStream> {
    let (stream, peer) = listener.accept().await.map_err(PipelineError::Io)?;
    stream.set_nodelay(true).ok();
    debug!(peer = %peer, "TCP accepted");
    Ok(stream)
}
