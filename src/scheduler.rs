use crate::error::SchedulerError;

/// An operation in the pipeline schedule for a single time step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipeOp {
    /// Receive activation tensors from the upstream stage.
    RecvActivation { micro_batch: u32 },
    /// Execute the forward pass on a micro-batch.
    Forward { micro_batch: u32 },
    /// Send activation tensors to the downstream stage.
    SendActivation { micro_batch: u32 },
    /// No work this step (pipeline bubble).
    Idle,
}

/// The schedule for a single stage: a sequence of operations per time step.
#[derive(Debug, Clone)]
pub struct StageSchedule {
    pub stage_idx: usize,
    pub ops: Vec<Vec<PipeOp>>,
}

/// Forward-only inference schedule (fill-drain pattern).
///
/// With `p` stages and `m` micro-batches, the schedule has `m + p - 1` time steps.
/// Bubble fraction = `(p - 1) / (m + p - 1)`.
#[derive(Debug, Clone)]
pub struct InferenceSchedule {
    pub num_stages: usize,
    pub num_micro_batches: u32,
    /// Total time steps: m + p - 1.
    pub total_steps: usize,
    pub stage_schedules: Vec<StageSchedule>,
}

impl InferenceSchedule {
    /// Generate a forward-only fill-drain schedule.
    ///
    /// - Stage 0 has no `RecvActivation` (it receives input directly from the client).
    /// - The last stage has no `SendActivation` (it outputs results directly).
    /// - Each stage starts its first forward at time step `stage_idx` (staggered fill).
    pub fn generate(
        num_stages: usize,
        num_micro_batches: u32,
    ) -> std::result::Result<Self, SchedulerError> {
        if num_stages == 0 {
            return Err(SchedulerError::ZeroStages);
        }
        if num_micro_batches == 0 {
            return Err(SchedulerError::ZeroMicroBatches);
        }

        let p = num_stages;
        let m = num_micro_batches as usize;
        let total_steps = m + p - 1;
        let is_first_stage = |s: usize| s == 0;
        let is_last_stage = |s: usize| s == p - 1;

        let stage_schedules = (0..p)
            .map(|s| {
                let mut ops = Vec::with_capacity(total_steps);
                for t in 0..total_steps {
                    // This stage processes micro-batch `mb` at time step `t`
                    // where mb = t - s (the stagger offset).
                    let mut step_ops = Vec::new();
                    if t >= s && (t - s) < m {
                        let mb = (t - s) as u32;
                        if !is_first_stage(s) {
                            step_ops.push(PipeOp::RecvActivation { micro_batch: mb });
                        }
                        step_ops.push(PipeOp::Forward { micro_batch: mb });
                        if !is_last_stage(s) {
                            step_ops.push(PipeOp::SendActivation { micro_batch: mb });
                        }
                    } else {
                        step_ops.push(PipeOp::Idle);
                    }
                    ops.push(step_ops);
                }
                StageSchedule { stage_idx: s, ops }
            })
            .collect();

        Ok(InferenceSchedule {
            num_stages,
            num_micro_batches,
            total_steps,
            stage_schedules,
        })
    }

    /// Bubble fraction: (p-1) / (m+p-1).
    pub fn bubble_fraction(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.num_stages - 1) as f64 / self.total_steps as f64
    }

    /// Get the schedule for a specific stage.
    pub fn stage(&self, stage_idx: usize) -> Option<&StageSchedule> {
        self.stage_schedules.get(stage_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_stage_single_batch() {
        let s = InferenceSchedule::generate(1, 1).unwrap();
        assert_eq!(s.total_steps, 1);
        assert_eq!(s.bubble_fraction(), 0.0);
        let ops = &s.stage_schedules[0].ops;
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], vec![PipeOp::Forward { micro_batch: 0 }]);
    }

    #[test]
    fn two_stages_two_batches() {
        let s = InferenceSchedule::generate(2, 2).unwrap();
        // total_steps = 2 + 2 - 1 = 3
        assert_eq!(s.total_steps, 3);
        // bubble = 1/3
        assert!((s.bubble_fraction() - 1.0 / 3.0).abs() < 1e-10);

        // Stage 0: forward mb0 at t=0, forward mb1 at t=1, idle at t=2
        let s0 = &s.stage_schedules[0].ops;
        assert_eq!(
            s0[0],
            vec![
                PipeOp::Forward { micro_batch: 0 },
                PipeOp::SendActivation { micro_batch: 0 },
            ]
        );
        assert_eq!(
            s0[1],
            vec![
                PipeOp::Forward { micro_batch: 1 },
                PipeOp::SendActivation { micro_batch: 1 },
            ]
        );
        assert_eq!(s0[2], vec![PipeOp::Idle]);

        // Stage 1: idle at t=0, forward mb0 at t=1, forward mb1 at t=2
        let s1 = &s.stage_schedules[1].ops;
        assert_eq!(s1[0], vec![PipeOp::Idle]);
        assert_eq!(
            s1[1],
            vec![
                PipeOp::RecvActivation { micro_batch: 0 },
                PipeOp::Forward { micro_batch: 0 },
            ]
        );
        assert_eq!(
            s1[2],
            vec![
                PipeOp::RecvActivation { micro_batch: 1 },
                PipeOp::Forward { micro_batch: 1 },
            ]
        );
    }

    #[test]
    fn four_stages_sixteen_batches_bubble() {
        let s = InferenceSchedule::generate(4, 16).unwrap();
        assert_eq!(s.total_steps, 19); // 16 + 4 - 1
        let bubble = s.bubble_fraction();
        // (4-1)/19 ≈ 0.1578
        assert!((bubble - 3.0 / 19.0).abs() < 1e-10);
    }

    #[test]
    fn every_micro_batch_covered() {
        let p = 3;
        let m = 5u32;
        let s = InferenceSchedule::generate(p, m).unwrap();

        for stage_idx in 0..p {
            let schedule = &s.stage_schedules[stage_idx];
            let mut forward_batches: Vec<u32> = Vec::new();
            for step in &schedule.ops {
                for op in step {
                    if let PipeOp::Forward { micro_batch } = op {
                        forward_batches.push(*micro_batch);
                    }
                }
            }
            forward_batches.sort();
            let expected: Vec<u32> = (0..m).collect();
            assert_eq!(
                forward_batches, expected,
                "stage {stage_idx} missing micro-batches"
            );
        }
    }

    #[test]
    fn first_stage_no_recv() {
        let s = InferenceSchedule::generate(3, 4).unwrap();
        for step in &s.stage_schedules[0].ops {
            for op in step {
                assert!(
                    !matches!(op, PipeOp::RecvActivation { .. }),
                    "stage 0 should not have RecvActivation"
                );
            }
        }
    }

    #[test]
    fn last_stage_no_send() {
        let s = InferenceSchedule::generate(3, 4).unwrap();
        for step in &s.stage_schedules[2].ops {
            for op in step {
                assert!(
                    !matches!(op, PipeOp::SendActivation { .. }),
                    "last stage should not have SendActivation"
                );
            }
        }
    }

    #[test]
    fn zero_stages_error() {
        assert!(matches!(
            InferenceSchedule::generate(0, 4),
            Err(SchedulerError::ZeroStages)
        ));
    }

    #[test]
    fn zero_micro_batches_error() {
        assert!(matches!(
            InferenceSchedule::generate(3, 0),
            Err(SchedulerError::ZeroMicroBatches)
        ));
    }
}
