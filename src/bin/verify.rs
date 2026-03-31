//! Verification binary: compare Rust MERT pipeline against Python reference output.
//!
//! Usage:
//!   cargo run --bin verify -- test.wav ml/verification/reference.json

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tch::{CModule, Device, IValue, Tensor};

#[derive(Debug, Deserialize)]
struct Reference {
    audio_samples: usize,
    sr: u32,
    duration: f64,
    embeddings_shape: Vec<usize>,
    embeddings_hash: String,
    embeddings_head: Vec<f64>,
    embeddings_tail: Vec<f64>,
    embeddings_mean: f64,
    embeddings_std: f64,
    boundary_probs_max: f64,
    boundary_probs_mean: f64,
    boundaries: Vec<f64>,
    labels: Vec<String>,
    num_segments: usize,
    beats: Vec<f64>,
    bpm: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: verify <audio_path> <reference_json>");
        std::process::exit(1);
    }

    let audio_path = Path::new(&args[1]);
    let reference_path = Path::new(&args[2]);

    println!("=== Mertalizer Rust Pipeline Verification ===\n");

    // Load reference
    let ref_str = std::fs::read_to_string(reference_path)
        .with_context(|| format!("Failed to read {}", reference_path.display()))?;
    let reference: Reference =
        serde_json::from_str(&ref_str).context("Failed to parse reference JSON")?;

    println!("Reference: {} samples, {:.2}s, {} segments",
             reference.audio_samples, reference.duration, reference.num_segments);
    println!("Reference embeddings: {:?}", reference.embeddings_shape);
    println!("Reference labels: {:?}\n", reference.labels);

    let device = Device::Cpu;
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // --- Step 1: Load and normalize audio ---
    println!("--- Step 1: Load and normalize audio ---");
    let (audio_tensor, sr, duration) = load_and_normalize_wav(audio_path, reference.sr)?;
    let audio_shape = audio_tensor.size();

    println!("  Rust: {} samples, {:.2}s @ {}Hz", audio_shape[1], duration, sr);
    println!("  Ref:  {} samples, {:.2}s @ {}Hz", reference.audio_samples, reference.duration, reference.sr);

    let dur_diff = (duration - reference.duration).abs();
    check("Duration match", dur_diff < 0.1, &format!("diff={:.4}s", dur_diff));

    // --- Step 2: Run MERT encoder ---
    println!("\n--- Step 2: Run MERT encoder ---");
    let mert_path = project_root.join("models").join("mert_encoder.pt");
    println!("  Loading MERT from: {}", mert_path.display());

    let mert = CModule::load_on_device(&mert_path, device)
        .context("Failed to load MERT encoder")?;

    let embeddings = {
        let output = mert
            .forward_is(&[IValue::Tensor(audio_tensor.shallow_clone())])
            .context("MERT forward failed")?;
        match output {
            IValue::Tensor(t) => t,
            _ => return Err(anyhow!("Expected tensor from MERT")),
        }
    };

    let emb_shape = embeddings.size();
    println!("  Rust embeddings shape: {:?}", emb_shape);
    println!("  Ref  embeddings shape: {:?}", reference.embeddings_shape);

    // Check shape
    let ref_frames = reference.embeddings_shape[0] as i64;
    let ref_dim = reference.embeddings_shape[1] as i64;
    check(
        "Embedding frames",
        emb_shape[1] == ref_frames,
        &format!("rust={}, ref={}", emb_shape[1], ref_frames),
    );
    check(
        "Embedding dim",
        emb_shape[2] == ref_dim,
        &format!("rust={}, ref={}", emb_shape[2], ref_dim),
    );

    // Check embedding statistics
    let emb_mean: f64 = embeddings.mean(tch::Kind::Float).double_value(&[]);
    let emb_std: f64 = embeddings.std(true).double_value(&[]);
    let mean_diff = (emb_mean - reference.embeddings_mean).abs();
    let std_diff = (emb_std - reference.embeddings_std).abs();
    println!("  Rust emb mean={:.6}, std={:.6}", emb_mean, emb_std);
    println!("  Ref  emb mean={:.6}, std={:.6}", reference.embeddings_mean, reference.embeddings_std);
    check("Embedding mean", mean_diff < 0.01, &format!("diff={:.6}", mean_diff));
    check("Embedding std", std_diff < 0.01, &format!("diff={:.6}", std_diff));

    // Check first frame values
    let first_frame: Vec<f32> = embeddings
        .get(0)
        .get(0)
        .narrow(0, 0, 10)
        .try_into()
        .context("Failed to extract first frame")?;
    let ref_head: Vec<f32> = reference.embeddings_head.iter().map(|x| *x as f32).collect();
    let head_max_diff: f32 = first_frame
        .iter()
        .zip(ref_head.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  First frame max diff: {:.6}", head_max_diff);
    check("First frame values", head_max_diff < 0.05, &format!("max_diff={:.6}", head_max_diff));

    // --- Step 3: Run head model ---
    println!("\n--- Step 3: Run boundary/label heads ---");
    let head_path = project_root.join("models").join("mertalizer_traced.pt");
    println!("  Loading heads from: {}", head_path.display());

    let head_model = CModule::load_on_device(&head_path, device)
        .context("Failed to load head model")?;

    let outputs = head_model
        .forward_is(&[IValue::Tensor(embeddings.shallow_clone())])
        .context("Head model forward failed")?;

    let (boundary_logits, label_logits) = match outputs {
        IValue::Tuple(t) if t.len() == 2 => {
            let bl = match &t[0] {
                IValue::Tensor(t) => t.shallow_clone(),
                _ => return Err(anyhow!("Expected tensor")),
            };
            let ll = match &t[1] {
                IValue::Tensor(t) => t.shallow_clone(),
                _ => return Err(anyhow!("Expected tensor")),
            };
            (bl, ll)
        }
        _ => return Err(anyhow!("Expected tuple of 2 tensors")),
    };

    let boundary_probs = boundary_logits.sigmoid();
    let bp_max: f64 = boundary_probs.max().double_value(&[]);
    let bp_mean: f64 = boundary_probs.mean(tch::Kind::Float).double_value(&[]);
    println!("  Boundary probs: max={:.4}, mean={:.4}", bp_max, bp_mean);
    println!("  Reference:      max={:.4}, mean={:.4}", reference.boundary_probs_max, reference.boundary_probs_mean);

    let bp_max_diff = (bp_max - reference.boundary_probs_max).abs();
    let bp_mean_diff = (bp_mean - reference.boundary_probs_mean).abs();
    check("Boundary prob max", bp_max_diff < 0.05, &format!("diff={:.4}", bp_max_diff));
    check("Boundary prob mean", bp_mean_diff < 0.05, &format!("diff={:.4}", bp_mean_diff));

    // --- Step 4: Post-process boundaries and labels ---
    println!("\n--- Step 4: Post-process ---");

    let threshold = 0.5;
    let probs_vec: Vec<f32> = boundary_probs
        .view([-1])
        .try_into()
        .context("Failed to convert probs")?;

    let frame_rate = probs_vec.len() as f64 / duration;
    let mut boundaries = vec![0.0];
    for (i, &prob) in probs_vec.iter().enumerate() {
        if prob > threshold as f32 && i > 0 {
            let time = i as f64 / frame_rate;
            if time - boundaries.last().unwrap() > 1.0 {
                boundaries.push(time);
            }
        }
    }
    if (duration - boundaries.last().unwrap()).abs() > 0.01 {
        boundaries.push(duration);
    }

    println!("  Rust boundaries: {} ({:?})", boundaries.len(), boundaries);
    println!("  Ref  boundaries: {} ({:?})", reference.boundaries.len(), reference.boundaries);

    let boundary_count_diff = (boundaries.len() as i32 - reference.boundaries.len() as i32).abs();
    check(
        "Boundary count",
        boundary_count_diff <= 2,
        &format!("rust={}, ref={}, diff={}", boundaries.len(), reference.boundaries.len(), boundary_count_diff),
    );

    // Predict labels
    let ll_shape = label_logits.size();
    let seq_len = ll_shape[1] as usize;
    let num_labels = ll_shape[2] as usize;
    let logits_data: Vec<f32> = label_logits.view([-1]).try_into()?;

    let label_names = ["INTRO", "VERSE", "PRE", "CHORUS", "BRIDGE", "SOLO", "OUTRO", "OTHER"];
    let mut labels = Vec::new();
    for i in 0..boundaries.len().saturating_sub(1) {
        let start_frame = ((boundaries[i] * frame_rate) as usize).min(seq_len - 1);
        let end_frame = ((boundaries[i + 1] * frame_rate) as usize).min(seq_len);
        let mut avg = vec![0.0f32; num_labels];
        for frame in start_frame..end_frame {
            for j in 0..num_labels {
                let idx = frame * num_labels + j;
                if idx < logits_data.len() {
                    avg[j] += logits_data[idx];
                }
            }
        }
        let label_idx = avg.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(7);
        labels.push(label_names[label_idx]);
    }

    println!("  Rust labels: {:?}", labels);
    println!("  Ref  labels: {:?}", reference.labels);

    let labels_match = labels.len() == reference.labels.len()
        && labels.iter().zip(reference.labels.iter()).all(|(a, b)| a == b);
    check("Labels match", labels_match, &format!("rust={:?} vs ref={:?}", labels, reference.labels));

    // --- Summary ---
    println!("\n=== Verification Summary ===");
    println!("  Audio:      {:.2}s (sr={})", duration, sr);
    println!("  Embeddings: {:?}", emb_shape);
    println!("  Boundaries: {}", boundaries.len());
    println!("  Segments:   {}", labels.len());
    println!("  Labels:     {:?}", labels);

    Ok(())
}

fn load_and_normalize_wav(path: &Path, target_sr: u32) -> Result<(Tensor, u32, f64)> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open: {}", path.display()))?;

    let spec = reader.spec();
    let file_sr = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Mono
    let mono: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()
    } else {
        samples
    };

    // Resample if needed
    let audio = if file_sr != target_sr {
        resample_simple(&mono, file_sr, target_sr)
    } else {
        mono
    };

    let num_samples = audio.len();
    let duration = num_samples as f64 / target_sr as f64;

    // Normalize: zero-mean, unit-variance
    let sum: f64 = audio.iter().map(|x| *x as f64).sum();
    let mean = sum / num_samples as f64;
    let var: f64 = audio.iter().map(|x| { let d = *x as f64 - mean; d * d }).sum::<f64>() / num_samples as f64;
    let std = var.sqrt().max(1e-8);

    let normalized: Vec<f32> = audio.iter().map(|x| ((*x as f64 - mean) / std) as f32).collect();
    let tensor = Tensor::from_slice(&normalized).reshape(&[1, num_samples as i64]);

    Ok((tensor, target_sr, duration))
}

/// Simple linear interpolation resampler.
fn resample_simple(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    let ratio = from_sr as f64 / to_sr as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

fn check(name: &str, passed: bool, detail: &str) {
    if passed {
        println!("  [PASS] {} ({})", name, detail);
    } else {
        println!("  [FAIL] {} ({})", name, detail);
    }
}
