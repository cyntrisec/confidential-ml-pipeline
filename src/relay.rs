use tokio::io::{AsyncRead, AsyncWrite};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

/// Handle to a running relay task. Dropping it does not cancel the task;
/// call `abort()` or `is_finished()` to manage lifecycle.
pub struct RelayHandle {
    pub upstream_to_downstream: JoinHandle<std::io::Result<u64>>,
    pub downstream_to_upstream: JoinHandle<std::io::Result<u64>>,
}

impl RelayHandle {
    /// Check if both directions have completed.
    pub fn is_finished(&self) -> bool {
        self.upstream_to_downstream.is_finished() && self.downstream_to_upstream.is_finished()
    }

    /// Abort both relay directions.
    pub fn abort(&self) {
        self.upstream_to_downstream.abort();
        self.downstream_to_upstream.abort();
    }
}

/// Start a bidirectional byte relay between two transports.
///
/// This is a "dumb pipe" — it never inspects or decrypts the bytes.
/// SecureChannel handshakes and encrypted data traverse the relay transparently.
///
/// Each direction runs as a separate tokio task using `tokio::io::copy`.
pub fn start_relay_link<U, D>(upstream: U, downstream: D) -> RelayHandle
where
    U: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    D: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (upstream_read, upstream_write) = tokio::io::split(upstream);
    let (downstream_read, downstream_write) = tokio::io::split(downstream);

    let u2d = tokio::spawn(async move {
        let mut r = upstream_read;
        let mut w = downstream_write;
        let bytes = tokio::io::copy(&mut r, &mut w).await;
        debug!(bytes = ?bytes, "relay upstream→downstream finished");
        bytes
    });

    let d2u = tokio::spawn(async move {
        let mut r = downstream_read;
        let mut w = upstream_write;
        let bytes = tokio::io::copy(&mut r, &mut w).await;
        debug!(bytes = ?bytes, "relay downstream→upstream finished");
        bytes
    });

    RelayHandle {
        upstream_to_downstream: u2d,
        downstream_to_upstream: d2u,
    }
}

/// Start relay links for a linear pipeline of N stages.
///
/// Returns `N - 1` relay handles connecting stage[i].data_out → stage[i+1].data_in.
///
/// The `transport_factory` is called with `(upstream_stage_idx, downstream_stage_idx)`
/// and must return a pair of connected transports (upstream_side, downstream_side).
pub async fn start_relay_mesh<F, Fut, T>(
    num_stages: usize,
    transport_factory: F,
) -> Vec<RelayHandle>
where
    F: Fn(usize, usize) -> Fut,
    Fut: std::future::Future<Output = (T, T)>,
    T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let mut handles = Vec::with_capacity(num_stages.saturating_sub(1));

    for i in 0..num_stages.saturating_sub(1) {
        let (upstream_side, downstream_side) = transport_factory(i, i + 1).await;
        debug!(upstream = i, downstream = i + 1, "starting relay link");
        handles.push(start_relay_link(upstream_side, downstream_side));
    }

    if handles.is_empty() && num_stages > 0 {
        warn!("single-stage pipeline: no relay links needed");
    }

    handles
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn relay_forwards_bytes() {
        // Create two duplex pairs: client↔relay↔server
        let (client, relay_left) = tokio::io::duplex(4096);
        let (relay_right, server) = tokio::io::duplex(4096);

        let mut handle = start_relay_link(relay_left, relay_right);

        let (mut client_read, mut client_write) = tokio::io::split(client);
        let (mut server_read, mut server_write) = tokio::io::split(server);

        // client → server
        client_write.write_all(b"hello server").await.unwrap();
        drop(client_write); // signal EOF

        let mut buf = vec![0u8; 64];
        let n = server_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"hello server");

        // server → client
        server_write.write_all(b"hello client").await.unwrap();
        drop(server_write);

        let mut buf = vec![0u8; 64];
        let n = client_read.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"hello client");

        // Wait for relay to finish
        drop(client_read);
        drop(server_read);
        let _ = (&mut handle.upstream_to_downstream).await;
        let _ = (&mut handle.downstream_to_upstream).await;
        assert!(handle.is_finished());
    }

    #[tokio::test]
    async fn relay_mesh_creates_correct_links() {
        let handles = start_relay_mesh(3, |i, j| async move {
            assert_eq!(j, i + 1);
            tokio::io::duplex(1024)
        })
        .await;

        assert_eq!(handles.len(), 2); // 3 stages → 2 relay links

        for h in &handles {
            h.abort();
        }
    }

    #[tokio::test]
    async fn single_stage_no_relays() {
        let handles = start_relay_mesh(1, |_, _| async { tokio::io::duplex(1024) }).await;
        assert!(handles.is_empty());
    }
}
