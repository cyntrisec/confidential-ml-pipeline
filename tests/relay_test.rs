use bytes::Bytes;
use confidential_ml_transport::frame::tensor::TensorRef;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, SecureChannel, SessionConfig};

use confidential_ml_pipeline::start_relay_link;

/// Test that a SecureChannel handshake + tensor exchange works through a relay.
///
/// Layout: initiator ↔ relay_left | relay | relay_right ↔ responder
#[tokio::test]
async fn secure_channel_through_relay() {
    // initiator ↔ relay_left
    let (initiator_transport, relay_left) = tokio::io::duplex(65536);
    // relay_right ↔ responder
    let (relay_right, responder_transport) = tokio::io::duplex(65536);

    // Start relay.
    let relay_handle = start_relay_link(relay_left, relay_right);

    // Responder side (accepts).
    let responder = tokio::spawn(async move {
        let provider = MockProvider::new();
        let mut channel = SecureChannel::accept_with_attestation(
            responder_transport,
            &provider,
            SessionConfig::default(),
        )
        .await
        .expect("responder handshake failed");

        // Receive tensor.
        let msg = channel.recv().await.expect("responder recv failed");
        match msg {
            confidential_ml_transport::Message::Tensor(t) => {
                assert_eq!(t.name, "activation");
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.shape, vec![1, 4]);
            }
            other => panic!("expected Tensor, got {:?}", other),
        }

        // Send back a data message.
        channel
            .send(Bytes::from_static(b"ack"))
            .await
            .expect("responder send failed");

        channel.shutdown().await.expect("responder shutdown failed");
    });

    // Initiator side (connects).
    let initiator = tokio::spawn(async move {
        let verifier = MockVerifier::new();
        let mut channel = SecureChannel::connect_with_attestation(
            initiator_transport,
            &verifier,
            SessionConfig::default(),
        )
        .await
        .expect("initiator handshake failed");

        // Send tensor through relay.
        let data = vec![0u8; 16]; // [1, 4] f32
        let tensor = TensorRef {
            name: "activation",
            dtype: DType::F32,
            shape: &[1, 4],
            data: &data,
        };
        channel
            .send_tensor(tensor)
            .await
            .expect("initiator send_tensor failed");

        // Receive ack.
        let msg = channel.recv().await.expect("initiator recv failed");
        match msg {
            confidential_ml_transport::Message::Data(data) => {
                assert_eq!(&data[..], b"ack");
            }
            other => panic!("expected Data, got {:?}", other),
        }

        // Receive shutdown.
        let msg = channel
            .recv()
            .await
            .expect("initiator recv shutdown failed");
        assert!(matches!(msg, confidential_ml_transport::Message::Shutdown));
    });

    responder.await.unwrap();
    initiator.await.unwrap();

    // Relay should be done now.
    assert!(relay_handle.is_finished());
}

/// Test relay mesh creates correct number of links.
#[tokio::test]
async fn relay_mesh_links() {
    let handles = confidential_ml_pipeline::start_relay_mesh(4, |i, j| async move {
        assert_eq!(j, i + 1);
        tokio::io::duplex(1024)
    })
    .await;

    assert_eq!(handles.len(), 3); // 4 stages → 3 links

    for h in &handles {
        h.abort();
    }
}
