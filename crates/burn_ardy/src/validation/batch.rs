//! Complete-clip qualification against independently generated serial fixtures.
use crate::Ardy;
use anyhow::{Context, Result, ensure};
use burn::prelude::Backend;
use burn_human_motion::{MotionClip, MotionRequest, TextEmbedding};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use web_time::Instant;
mod fixed;

#[derive(Serialize, Deserialize)]
pub struct Suite {
    pub baseline_commit: String,
    pub embeddings: Vec<TextEmbedding>,
    pub cases: Vec<Case>,
}

#[derive(Serialize, Deserialize)]
pub struct Case {
    pub name: String,
    pub requests: Vec<MotionRequest>,
    pub embedding_indices: Vec<usize>,
    pub reference: Vec<MotionClip>,
}

#[derive(Clone, Copy, Serialize)]
struct ClipLimits {
    fk_max_m: f32,
    fk_rms_m: f64,
    rotation_max_rad: f64,
    contact_fraction: f64,
}

impl ClipLimits {
    const STABLE: Self = Self {
        fk_max_m: 0.000001,
        fk_rms_m: 0.0000001,
        rotation_max_rad: 0.000001,
        contact_fraction: 0.0,
    };
    fn baseline(fk_max_m: f32, fk_rms_m: f64) -> Self {
        Self {
            fk_max_m,
            fk_rms_m,
            rotation_max_rad: 0.05,
            contact_fraction: 0.01,
        }
    }
}

fn compare(
    actual: &MotionClip,
    expected: &MotionClip,
    limits: Option<ClipLimits>,
) -> Result<Value> {
    actual.validate()?;
    expected.validate()?;
    ensure!(
        actual.frames.len() == expected.frames.len()
            && actual.fps == expected.fps
            && actual.provenance == expected.provenance
            && serde_json::to_value(&actual.rig)? == serde_json::to_value(&expected.rig)?,
        "clip metadata/order mismatch"
    );
    let mut max = 0.0f32;
    let mut root_max = 0.0f32;
    let mut rotation_max = 0.0f64;
    let mut sum = 0.0f64;
    let mut count = 0;
    let mut contacts = 0;
    for (a, b) in actual.frames.iter().zip(&expected.frames) {
        root_max = root_max.max(a.root_translation.distance(b.root_translation));
        let (a_pos, _) = actual.rig.forward(a)?;
        let (b_pos, _) = expected.rig.forward(b)?;
        for (a, b) in a_pos.iter().zip(&b_pos) {
            let distance = a.distance(*b);
            max = max.max(distance);
            sum += f64::from(distance).powi(2);
            count += 1;
        }
        for (a, b) in a.local_rotations.iter().zip(&b.local_rotations) {
            let a = a.as_dquat().normalize();
            let b = b.as_dquat().normalize();
            rotation_max = rotation_max.max(2.0 * a.dot(b).abs().min(1.0).acos());
        }
        contacts += a
            .foot_contacts
            .iter()
            .zip(b.foot_contacts)
            .filter(|(a, b)| **a != *b)
            .count();
    }
    let rms = (sum / count as f64).sqrt();
    let contact_fraction = contacts as f64 / (actual.frames.len() * 4) as f64;
    if let Some(limits) = limits {
        ensure!(
            max <= limits.fk_max_m
                && rms <= limits.fk_rms_m
                && rotation_max <= limits.rotation_max_rad
                && contact_fraction <= limits.contact_fraction,
            "clip parity failed: FK max={max}m rms={rms}m rotation={rotation_max}rad contact_fraction={contact_fraction}"
        );
    }
    Ok(json!({"fk_max_m":max,"fk_rms_m":rms,"root_max_m":root_max,
        "local_rotation_max_rad":rotation_max,"contact_disagreements":contacts,
        "limits":limits,"gate_enforced":limits.is_some()}))
}

async fn serial<B: Backend>(
    model: &Ardy<B>,
    requests: &[MotionRequest],
    embeddings: &[TextEmbedding],
) -> Result<Vec<MotionClip>> {
    let mut clips = Vec::new();
    for (request, embedding) in requests.iter().zip(embeddings) {
        clips.push(model.generate(request, embedding, |_| true).await?);
    }
    Ok(clips)
}

/// Times fully decoded, host-visible clips, excluding load and text encoding.
/// Missing independent references are fatal, rather than silently self-comparing.
pub async fn validate<B: Backend>(model: &Ardy<B>, suite: Suite) -> Result<Value> {
    ensure!(
        !suite.cases.is_empty() && !suite.baseline_commit.is_empty(),
        "missing baseline"
    );
    let mut reports = Vec::new();
    for case in &suite.cases {
        ensure!(
            !case.requests.is_empty()
                && case.requests.len() == case.reference.len()
                && case.requests.len() == case.embedding_indices.len(),
            "missing case references"
        );
        let embeddings: Vec<_> = case
            .embedding_indices
            .iter()
            .map(|&i| {
                suite
                    .embeddings
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("embedding index out of range"))
            })
            .collect::<Result<_>>()?;
        crate::batch::validate_batch(&case.requests, &embeddings)?;
        let fixed_inputs = fixed::validate(model, &case.requests, &embeddings)
            .await
            .with_context(|| case.name.clone())?;
        let serial_clips = serial(model, &case.requests, &embeddings).await?;
        let mut progress = Vec::new();
        let batch = model
            .generate_batch(&case.requests, &embeddings, |n| {
                progress.push(n);
                true
            })
            .await?;
        let frames = case.requests[0].frames;
        let expected_progress: Vec<_> = (0..frames)
            .step_by(40)
            .chain(std::iter::once(frames))
            .collect();
        ensure!(
            progress == expected_progress,
            "incorrect progress notifications"
        );
        let (max_m, rms_m) = if frames > 80 {
            (0.03, 0.005)
        } else {
            (0.01, 0.002)
        };
        let serial_parity = serial_clips
            .iter()
            .zip(&case.reference)
            .enumerate()
            .map(|(i, (a, b))| {
                compare(a, b, Some(ClipLimits::baseline(max_m, rms_m)))
                    .with_context(|| format!("{} serial actor {i}", case.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let batch_parity = batch
            .iter()
            .zip(&case.reference)
            .enumerate()
            .map(|(i, (a, b))| {
                compare(a, b, None).with_context(|| format!("{} batch actor {i}", case.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let batch_vs_serial = batch
            .iter()
            .zip(&serial_clips)
            .enumerate()
            .map(|(i, (a, b))| {
                compare(a, b, Some(ClipLimits::STABLE))
                    .with_context(|| format!("{} batch vs serial actor {i}", case.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let reverse_requests: Vec<_> = case.requests.iter().rev().cloned().collect();
        let reverse_embeddings: Vec<_> = embeddings.iter().rev().cloned().collect();
        let reversed = model
            .generate_batch(&reverse_requests, &reverse_embeddings, |_| true)
            .await?;
        let permutation = reversed
            .iter()
            .rev()
            .zip(&batch)
            .map(|(a, b)| compare(a, b, Some(ClipLimits::STABLE)))
            .collect::<Result<Vec<_>>>()?;
        let mut serial_seconds = Vec::new();
        let mut batch_seconds = Vec::new();
        // Alternate execution order to reduce systematic clock/thermal bias.
        for round in 0..3 {
            for batched in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                if batched {
                    let _ = model
                        .generate_batch(&case.requests, &embeddings, |_| true)
                        .await?;
                    batch_seconds.push(start.elapsed().as_secs_f64());
                } else {
                    let _ = serial(model, &case.requests, &embeddings).await?;
                    serial_seconds.push(start.elapsed().as_secs_f64());
                }
            }
        }
        serial_seconds.sort_by(f64::total_cmp);
        batch_seconds.sort_by(f64::total_cmp);
        eprintln!(
            "{}: serial={:.3}s batch={:.3}s",
            case.name, serial_seconds[1], batch_seconds[1]
        );
        reports.push(json!({"name":case.name,"actors":case.requests.len(),"frames_per_actor":frames,
            "history_frames":case.requests[0].history_frames,"progress":progress,
            "fixed_inputs":fixed_inputs,"serial_vs_baseline":serial_parity,"batch_vs_baseline_diagnostic":batch_parity,"batch_vs_serial":batch_vs_serial,"permutation":permutation,
            "warm_serial_seconds":serial_seconds,"warm_batch_seconds":batch_seconds,
            "throughput_speedup":serial_seconds[1]/batch_seconds[1],
            "batch_generated_frames_per_second":(frames*case.requests.len()) as f64/batch_seconds[1]}));
    }
    let first = &suite.cases[0];
    let embedding = suite.embeddings[first.embedding_indices[0]].clone();
    let mut request = first.requests[0].clone();
    ensure!(
        model
            .generate(&request, &embedding, |_| false)
            .await
            .is_err(),
        "initial cancellation ignored"
    );
    request.frames = 12000;
    request.history_frames = 160;
    let requests = vec![request; 8];
    let embeddings = vec![embedding; 8];
    let mut progress = Vec::new();
    ensure!(
        model
            .generate_batch(&requests, &embeddings, |n| {
                progress.push(n);
                n == 0
            })
            .await
            .is_err(),
        "window cancellation ignored"
    );
    ensure!(progress == [0, 40], "cancellation ran another window");
    let completed = model
        .generate(&first.requests[0], &embeddings[0], |n| {
            n != first.requests[0].frames
        })
        .await?;
    ensure!(
        completed.frames.len() == first.requests[0].frames,
        "completion notification discarded result"
    );
    Ok(
        json!({"passed":true,"baseline_commit":suite.baseline_commit,"cases":reports,
        "cancellation":{"before_allocation":true,"max_request_batch":8,"max_request_frames":12000,"progress":progress,"completed_notification":true},
        "synchronization":"full decoded clip readback","includes_decoder":true,"includes_text_encoder":false,
        "semantic_quality":"real prompt embeddings; numerical regression, not human-rated semantic quality"}),
    )
}
