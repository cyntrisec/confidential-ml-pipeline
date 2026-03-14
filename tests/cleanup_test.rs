//! SEC-705: Verify that explicit tensor cleanup after forward/send does not panic.

use bytes::Bytes;
use confidential_ml_transport::OwnedTensor;
use confidential_ml_pipeline::ForwardOutput;
use zeroize::Zeroize;

/// Create a dummy OwnedTensor with some non-empty fields.
fn dummy_tensor(name: &str, shape: &[u32], data_len: usize) -> OwnedTensor {
    OwnedTensor {
        name: name.to_string(),
        dtype: confidential_ml_transport::DType::F32,
        shape: shape.to_vec(),
        data: Bytes::from(vec![0xABu8; data_len]),
    }
}

#[test]
fn forward_output_cleanup_does_not_panic() {
    let mut output = ForwardOutput {
        tensors: vec![
            dummy_tensor("activation_0", &[1, 384], 1536),
            dummy_tensor("activation_1", &[2, 768], 6144),
        ],
    };

    // Simulate the cleanup path from stage.rs process_request().
    for tensor in &mut output.tensors {
        tensor.name.zeroize();
        tensor.shape.zeroize();
        // tensor.data is Bytes (Arc-backed) — cannot reliably zeroize.
    }

    // Verify metadata was cleared.
    for tensor in &output.tensors {
        assert!(tensor.name.is_empty(), "name should be zeroized (empty)");
        assert!(tensor.shape.is_empty(), "shape should be zeroized (empty)");
    }

    drop(output);
}

#[test]
fn input_tensors_cleanup_does_not_panic() {
    let mut input_tensors: Vec<Vec<OwnedTensor>> = vec![
        vec![dummy_tensor("input_0", &[1, 128], 512)],
        vec![
            dummy_tensor("input_1a", &[4, 256], 4096),
            dummy_tensor("input_1b", &[4, 256], 4096),
        ],
    ];

    // Simulate the cleanup path from orchestrator.rs infer_inner().
    for mb_tensors in &mut input_tensors {
        for tensor in mb_tensors.iter_mut() {
            tensor.name.zeroize();
            tensor.shape.zeroize();
        }
    }

    // Verify metadata was cleared.
    for mb_tensors in &input_tensors {
        for tensor in mb_tensors {
            assert!(tensor.name.is_empty(), "name should be zeroized (empty)");
            assert!(tensor.shape.is_empty(), "shape should be zeroized (empty)");
        }
    }

    drop(input_tensors);
}

#[test]
fn cleanup_empty_forward_output() {
    let mut output = ForwardOutput {
        tensors: Vec::new(),
    };

    // Cleanup on empty output should be a no-op and not panic.
    for tensor in &mut output.tensors {
        tensor.name.zeroize();
        tensor.shape.zeroize();
    }

    assert!(output.tensors.is_empty());
    drop(output);
}
