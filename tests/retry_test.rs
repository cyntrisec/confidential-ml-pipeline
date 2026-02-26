#![cfg(feature = "tcp")]

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;

use confidential_ml_pipeline::{tcp::connect_tcp_retry, PipelineError};
use confidential_ml_transport::RetryPolicy;
use tokio::net::TcpListener;

fn test_retry_policy(max_retries: u32, delay_ms: u64) -> RetryPolicy {
    RetryPolicy {
        max_retries,
        initial_delay: Duration::from_millis(delay_ms),
        max_delay: Duration::from_millis(delay_ms),
        backoff_multiplier: 1.0,
    }
}

async fn reserve_local_port() -> u16 {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

#[tokio::test]
async fn connect_tcp_retry_error_includes_target_and_attempts() {
    let port = reserve_local_port().await;
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    let policy = test_retry_policy(2, 5);

    let err = connect_tcp_retry(addr, &policy)
        .await
        .expect_err("connect should fail with no listener");

    let msg = err.to_string();
    assert!(
        msg.contains(&addr.to_string()),
        "error should include target addr, got: {msg}"
    );
    assert!(
        msg.contains("after 3 attempt(s)"),
        "error should include attempt count, got: {msg}"
    );
    assert!(
        matches!(err, PipelineError::Io(_)),
        "expected PipelineError::Io, got: {err:?}"
    );
}

#[tokio::test]
async fn connect_tcp_retry_none_policy_reports_single_attempt() {
    let port = reserve_local_port().await;
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    let policy = RetryPolicy::none();

    let err = connect_tcp_retry(addr, &policy)
        .await
        .expect_err("connect should fail with no listener");

    let msg = err.to_string();
    assert!(
        msg.contains("after 1 attempt(s)"),
        "expected single-attempt message, got: {msg}"
    );
}

#[tokio::test]
async fn connect_tcp_retry_recovers_when_listener_appears_later() {
    let port = reserve_local_port().await;
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
    let policy = test_retry_policy(10, 20);

    let listener_task = tokio::spawn(async move {
        // Force at least one failed attempt before the listener exists.
        tokio::time::sleep(Duration::from_millis(120)).await;
        let listener = TcpListener::bind(addr).await.expect("delayed bind");
        let (_stream, _) = listener.accept().await.expect("accept delayed client");
    });

    let stream = connect_tcp_retry(addr, &policy)
        .await
        .expect("connect should eventually succeed");
    drop(stream);

    listener_task.await.expect("listener task join");
}
