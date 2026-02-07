use confidential_ml_pipeline::{InferenceSchedule, PipeOp, SchedulerError};

#[test]
fn single_stage_single_batch() {
    let s = InferenceSchedule::generate(1, 1).unwrap();
    assert_eq!(s.total_steps, 1);
    assert_eq!(s.bubble_fraction(), 0.0);
}

#[test]
fn bubble_fraction_4_stages_16_batches() {
    let s = InferenceSchedule::generate(4, 16).unwrap();
    assert_eq!(s.total_steps, 19);
    let expected = 3.0 / 19.0;
    assert!((s.bubble_fraction() - expected).abs() < 1e-10);
}

#[test]
fn bubble_fraction_4_stages_64_batches() {
    let s = InferenceSchedule::generate(4, 64).unwrap();
    assert_eq!(s.total_steps, 67);
    let expected = 3.0 / 67.0;
    assert!((s.bubble_fraction() - expected).abs() < 1e-10);
}

#[test]
fn all_micro_batches_processed_by_all_stages() {
    for p in 1..=5 {
        for m in 1..=10u32 {
            let s = InferenceSchedule::generate(p, m).unwrap();

            for stage_idx in 0..p {
                let schedule = s.stage(stage_idx).unwrap();
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
                    "p={p}, m={m}, stage={stage_idx}: micro-batches don't match"
                );
            }
        }
    }
}

#[test]
fn first_stage_never_receives() {
    for p in 1..=4 {
        for m in 1..=8u32 {
            let s = InferenceSchedule::generate(p, m).unwrap();
            for step in &s.stage(0).unwrap().ops {
                for op in step {
                    assert!(
                        !matches!(op, PipeOp::RecvActivation { .. }),
                        "p={p}, m={m}: stage 0 should never RecvActivation"
                    );
                }
            }
        }
    }
}

#[test]
fn last_stage_never_sends() {
    for p in 1..=4 {
        for m in 1..=8u32 {
            let s = InferenceSchedule::generate(p, m).unwrap();
            let last = p - 1;
            for step in &s.stage(last).unwrap().ops {
                for op in step {
                    assert!(
                        !matches!(op, PipeOp::SendActivation { .. }),
                        "p={p}, m={m}: last stage should never SendActivation"
                    );
                }
            }
        }
    }
}

#[test]
fn total_steps_formula() {
    for p in 1..=5 {
        for m in 1..=10u32 {
            let s = InferenceSchedule::generate(p, m).unwrap();
            assert_eq!(
                s.total_steps,
                m as usize + p - 1,
                "p={p}, m={m}: total_steps should be m + p - 1"
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

/// Forward order within each stage: micro-batch 0 before 1 before 2...
#[test]
fn forward_order_preserved() {
    let s = InferenceSchedule::generate(3, 5).unwrap();
    for stage_idx in 0..3 {
        let schedule = s.stage(stage_idx).unwrap();
        let mut last_mb: Option<u32> = None;
        for step in &schedule.ops {
            for op in step {
                if let PipeOp::Forward { micro_batch } = op {
                    if let Some(prev) = last_mb {
                        assert!(
                            *micro_batch > prev,
                            "stage {stage_idx}: micro-batch {micro_batch} after {prev}"
                        );
                    }
                    last_mb = Some(*micro_batch);
                }
            }
        }
    }
}
