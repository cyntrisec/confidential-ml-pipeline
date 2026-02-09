use std::collections::BTreeMap;

use confidential_ml_transport::ExpectedMeasurements;
use serde::{Deserialize, Serialize};

use crate::error::ManifestError;

/// Describes how a model is sharded across pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub model_name: String,
    pub model_version: String,
    pub total_layers: usize,
    pub stages: Vec<StageSpec>,
    pub activation_spec: ActivationSpec,
}

/// Specification for a single pipeline stage (enclave).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSpec {
    pub stage_idx: usize,
    /// First layer (inclusive).
    pub layer_start: usize,
    /// Last layer (exclusive).
    pub layer_end: usize,
    /// SHA-256 hashes (hex-encoded) of model weight files for this stage.
    pub weight_hashes: Vec<String>,
    /// Expected attestation measurements: register index -> hex-encoded hash.
    pub expected_measurements: BTreeMap<usize, String>,
    pub endpoint: StageEndpoint,
}

/// Network endpoints for a stage's control and data channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEndpoint {
    pub control: PortSpec,
    pub data_in: PortSpec,
    pub data_out: PortSpec,
}

/// Transport-level address for a port.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PortSpec {
    #[serde(rename = "tcp")]
    Tcp { addr: String },
    #[serde(rename = "vsock")]
    VSock { cid: u32, port: u32 },
}

/// Describes the activation tensor format exchanged between stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSpec {
    pub dtype: ActivationDType,
    pub hidden_dim: u32,
    pub max_seq_len: u32,
}

/// Data type for inter-stage activation tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationDType {
    F32,
    F16,
    BF16,
}

impl ShardManifest {
    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> std::result::Result<Self, ManifestError> {
        let manifest: Self = serde_json::from_str(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> std::result::Result<String, ManifestError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Validate that stages are contiguous, correctly indexed, and cover all layers.
    pub fn validate(&self) -> std::result::Result<(), ManifestError> {
        if self.stages.is_empty() {
            return Err(ManifestError::EmptyStages);
        }

        for (i, stage) in self.stages.iter().enumerate() {
            if stage.stage_idx != i {
                return Err(ManifestError::WrongStageIndex {
                    stage_idx: i,
                    actual: stage.stage_idx,
                });
            }
            if stage.layer_start >= stage.layer_end {
                return Err(ManifestError::InvalidLayerRange {
                    stage_idx: i,
                    start: stage.layer_start,
                    end: stage.layer_end,
                });
            }
        }

        // Check contiguity.
        for i in 0..self.stages.len() - 1 {
            let end = self.stages[i].layer_end;
            let next_start = self.stages[i + 1].layer_start;
            if end != next_start {
                return Err(ManifestError::NonContiguousLayers {
                    stage_idx: i,
                    end,
                    next_start,
                });
            }
        }

        // Layers must start at 0.
        if self.stages[0].layer_start != 0 {
            return Err(ManifestError::LayerStartNotZero {
                start: self.stages[0].layer_start,
            });
        }

        // Check total coverage.
        let last_end = self.stages.last().unwrap().layer_end;
        if last_end != self.total_layers {
            return Err(ManifestError::LayerCountMismatch {
                covered: last_end,
                total: self.total_layers,
            });
        }

        Ok(())
    }
}

impl StageSpec {
    /// Convert hex-encoded expected measurements to the transport crate's type.
    pub fn to_expected_measurements(
        &self,
    ) -> std::result::Result<ExpectedMeasurements, hex::FromHexError> {
        let mut values = BTreeMap::new();
        for (register, hex_hash) in &self.expected_measurements {
            values.insert(*register, hex::decode(hex_hash)?);
        }
        Ok(ExpectedMeasurements::new(values))
    }

    /// Number of layers assigned to this stage.
    pub fn num_layers(&self) -> usize {
        self.layer_end - self.layer_start
    }
}

impl ActivationDType {
    /// Size of one element in bytes.
    pub const fn element_size(self) -> usize {
        match self {
            ActivationDType::F32 => 4,
            ActivationDType::F16 | ActivationDType::BF16 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoint(base_port: u32) -> StageEndpoint {
        StageEndpoint {
            control: PortSpec::Tcp {
                addr: format!("127.0.0.1:{}", base_port),
            },
            data_in: PortSpec::Tcp {
                addr: format!("127.0.0.1:{}", base_port + 1),
            },
            data_out: PortSpec::Tcp {
                addr: format!("127.0.0.1:{}", base_port + 2),
            },
        }
    }

    fn make_manifest(num_stages: usize, layers_per_stage: usize) -> ShardManifest {
        let stages = (0..num_stages)
            .map(|i| StageSpec {
                stage_idx: i,
                layer_start: i * layers_per_stage,
                layer_end: (i + 1) * layers_per_stage,
                weight_hashes: vec![],
                expected_measurements: BTreeMap::new(),
                endpoint: make_endpoint((9000 + i * 10) as u32),
            })
            .collect();

        ShardManifest {
            model_name: "test-model".into(),
            model_version: "1.0".into(),
            total_layers: num_stages * layers_per_stage,
            stages,
            activation_spec: ActivationSpec {
                dtype: ActivationDType::F32,
                hidden_dim: 768,
                max_seq_len: 512,
            },
        }
    }

    #[test]
    fn valid_manifest() {
        let m = make_manifest(3, 4);
        assert!(m.validate().is_ok());
    }

    #[test]
    fn json_roundtrip() {
        let m = make_manifest(2, 6);
        let json = m.to_json().unwrap();
        let m2 = ShardManifest::from_json(&json).unwrap();
        assert_eq!(m2.model_name, "test-model");
        assert_eq!(m2.stages.len(), 2);
        assert_eq!(m2.stages[1].layer_start, 6);
    }

    #[test]
    fn empty_stages() {
        let m = ShardManifest {
            model_name: "test".into(),
            model_version: "1".into(),
            total_layers: 0,
            stages: vec![],
            activation_spec: ActivationSpec {
                dtype: ActivationDType::F32,
                hidden_dim: 768,
                max_seq_len: 512,
            },
        };
        assert!(matches!(m.validate(), Err(ManifestError::EmptyStages)));
    }

    #[test]
    fn non_contiguous_layers() {
        let mut m = make_manifest(2, 4);
        m.stages[1].layer_start = 5; // gap
        assert!(matches!(
            m.validate(),
            Err(ManifestError::NonContiguousLayers { .. })
        ));
    }

    #[test]
    fn wrong_stage_index() {
        let mut m = make_manifest(2, 4);
        m.stages[1].stage_idx = 5;
        assert!(matches!(
            m.validate(),
            Err(ManifestError::WrongStageIndex { .. })
        ));
    }

    #[test]
    fn layer_count_mismatch() {
        let mut m = make_manifest(2, 4);
        m.total_layers = 100;
        assert!(matches!(
            m.validate(),
            Err(ManifestError::LayerCountMismatch { .. })
        ));
    }

    #[test]
    fn layer_start_not_zero() {
        let mut m = make_manifest(2, 5);
        // Shift both stages so they start at 10 instead of 0.
        m.stages[0].layer_start = 10;
        m.stages[0].layer_end = 15;
        m.stages[1].layer_start = 15;
        m.stages[1].layer_end = 20;
        // Coverage is 10 layers, matching total_layers, but doesn't start at 0.
        assert!(matches!(
            m.validate(),
            Err(ManifestError::LayerStartNotZero { start: 10 })
        ));
    }

    #[test]
    fn invalid_layer_range() {
        let mut m = make_manifest(2, 4);
        m.stages[0].layer_start = 10;
        m.stages[0].layer_end = 5;
        assert!(matches!(
            m.validate(),
            Err(ManifestError::InvalidLayerRange { .. })
        ));
    }

    #[test]
    fn expected_measurements_conversion() {
        let stage = StageSpec {
            stage_idx: 0,
            layer_start: 0,
            layer_end: 4,
            weight_hashes: vec![],
            expected_measurements: BTreeMap::from([(0, "abcd1234".into()), (1, "deadbeef".into())]),
            endpoint: make_endpoint(9000),
        };
        let em = stage.to_expected_measurements().unwrap();
        assert_eq!(em.values.len(), 2);
        assert_eq!(em.values[&0], hex::decode("abcd1234").unwrap());
    }

    #[test]
    fn vsock_port_spec_serde() {
        let spec = PortSpec::VSock {
            cid: 16,
            port: 5000,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("vsock"));
        let parsed: PortSpec = serde_json::from_str(&json).unwrap();
        match parsed {
            PortSpec::VSock { cid, port } => {
                assert_eq!(cid, 16);
                assert_eq!(port, 5000);
            }
            _ => panic!("expected VSock"),
        }
    }
}
