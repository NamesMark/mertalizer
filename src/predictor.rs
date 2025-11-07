use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tch::{CModule, Device, IValue, Tensor};
use tracing::{debug, info, instrument};

#[derive(Debug, Clone, Default)]
pub struct PredictOptions {
    pub threshold: Option<f64>,
    pub smooth: Option<bool>,
    pub smoothing_window: Option<u32>,
    pub min_gap_seconds: Option<f64>,
    pub min_segment_seconds: Option<f64>,
    pub position_bias: Option<bool>,
    pub prominence: Option<f64>,
    pub verbose: bool,
}

const LABEL_NAMES: [&str; 8] = [
    "INTRO", "VERSE", "PRE", "CHORUS", "BRIDGE", "SOLO", "OUTRO", "OTHER",
];

pub struct MusicStructurePredictor {
    model: CModule,
    device: Device,
    project_root: PathBuf,
    python_bin: PathBuf,
}

impl MusicStructurePredictor {
    pub fn new(project_root: PathBuf, model_path: PathBuf, python_bin: PathBuf) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!(
                "Model not found at {}",
                model_path.display()
            ));
        }

        // Check for CUDA availability
        let device = if tch::Cuda::is_available() {
            info!("🎮 CUDA available, using GPU");
            Device::Cuda(0)
        } else {
            info!("💻 CUDA not available, using CPU");
            Device::Cpu
        };

        info!("📦 Loading TorchScript model from {}", model_path.display());
        let model = CModule::load_on_device(&model_path, device)
            .context("Failed to load TorchScript model")?;

        info!("✓ Model loaded successfully on device: {:?}", device);

        Ok(Self {
            model,
            device,
            project_root,
            python_bin,
        })
    }

    pub fn model_path(&self) -> &Path {
        Path::new("models/mertalizer_traced.pt")
    }

    pub fn python_bin(&self) -> &Path {
        &self.python_bin
    }

    pub fn script_path(&self) -> &Path {
        Path::new("ml/inference/cli.py")
    }

    #[instrument(skip_all, fields(audio = %audio_path.display()))]
    pub async fn predict(&self, audio_path: &Path, options: &PredictOptions) -> Result<Value> {
        debug!("🎵 Processing audio file");

        // Extract embeddings using Python (MERT model is complex to export)
        let (embeddings, sr, duration, beats) = self
            .extract_embeddings_python(audio_path)
            .await
            .context("Failed to extract embeddings")?;

        debug!("🧠 Running native Rust inference");
        let (boundary_logits, label_logits) = self
            .run_inference(&embeddings)
            .context("Model inference failed")?;

        // Post-process results
        let threshold = options.threshold.unwrap_or(0.5);
        let boundaries = self.detect_boundaries(&boundary_logits, duration, threshold)?;
        
        let labels = self.predict_labels(&label_logits, &boundaries, duration)?;
        
        let segments = self.build_segments(&boundaries, &labels)?;

        // Build response
        let result = json!({
            "track_id": audio_path.file_stem().unwrap().to_str().unwrap(),
            "sr": sr,
            "duration": duration,
            "boundaries": boundaries,
            "labels": labels,
            "segments": segments,
            "version": "rust-tch@2025-11-07",
            "beats": beats,
            "threshold": threshold,
            "smooth": options.smooth.unwrap_or(true),
            "smoothing_window": options.smoothing_window.unwrap_or(1),
            "min_gap_seconds": options.min_gap_seconds.unwrap_or(3.0),
            "min_segment_seconds": options.min_segment_seconds.unwrap_or(1.5),
            "position_bias": options.position_bias.unwrap_or(true),
        });

        debug!("✓ Prediction complete");
        Ok(result)
    }

    async fn extract_embeddings_python(
        &self,
        audio_path: &Path,
    ) -> Result<(Tensor, u32, f64, Vec<f64>)> {
        use tokio::process::Command;

        // Create temp file for embeddings
        let temp_output = tempfile::Builder::new()
            .prefix("mertalizer_emb")
            .suffix(".json")
            .tempfile()
            .context("Failed to create temp file")?;
        let output_path = temp_output.path().to_path_buf();

        // Call Python script to extract embeddings
        let script_path = self.project_root
            .join("ml")
            .join("inference")
            .join("extract_embeddings.py");
        
        let mut command = Command::new(&self.python_bin);
        command
            .current_dir(&self.project_root)
            .env("PYTHONPATH", self.project_root.join("ml").display().to_string())
            .arg(&script_path)
            .arg(audio_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--model-type")
            .arg("mert");

        debug!("Extracting embeddings with Python (this may take a moment for MERT loading)");
        let output = command
            .output()
            .await
            .context("Failed to run Python embedding extraction")?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        debug!("Python exit code: {:?}", output.status.code());
        debug!("Python stdout: {}", stdout);
        debug!("Python stderr: {}", stderr);
        
        if !output.status.success() {
            return Err(anyhow!(
                "Python embedding extraction failed with exit code {}\nstderr: {}\nstdout: {}",
                output.status.code().unwrap_or(-1),
                stderr,
                stdout
            ));
        }

        // Load embeddings from JSON
        debug!("Reading embeddings from: {}", output_path.display());
        let data = tokio::fs::read_to_string(&output_path)
            .await
            .with_context(|| format!("Failed to read embedding output from {} - file might have been deleted", output_path.display()))?;
        
        debug!("Read {} bytes of JSON data", data.len());
        let parsed: Value = serde_json::from_str(&data)
            .with_context(|| format!("Invalid JSON in embedding output. Data length: {}", data.len()))?;

        debug!("Parsing embeddings array");
        let emb_vec: Vec<Vec<f32>> = serde_json::from_value(parsed["embeddings"].clone())
            .context("Invalid embeddings format in JSON")?;
        debug!("Extracted embeddings array with {} frames", emb_vec.len());
        let sr: u32 = parsed["sr"]
            .as_u64()
            .ok_or_else(|| anyhow!("Missing 'sr' field in embedding output"))? as u32;
        let duration: f64 = parsed["duration"]
            .as_f64()
            .ok_or_else(|| anyhow!("Missing 'duration' field in embedding output"))?;
        let beats: Vec<f64> = serde_json::from_value(parsed["beats"].clone())
            .unwrap_or_default();

        // Convert to tensor
        let emb_flat: Vec<f32> = emb_vec.into_iter().flatten().collect();
        let batch_size = 1;
        let seq_len = emb_flat.len() / 768;
        let tensor = Tensor::from_slice(&emb_flat)
            .reshape(&[batch_size as i64, seq_len as i64, 768])
            .to_device(self.device);

        // Keep temp_output alive until we're done with it
        drop(temp_output);

        Ok((tensor, sr, duration, beats))
    }

    fn run_inference(&self, embeddings: &Tensor) -> Result<(Tensor, Tensor)> {
        // Run model inference
        let outputs = self
            .model
            .forward_is(&[IValue::Tensor(embeddings.shallow_clone())])
            .context("Model forward pass failed")?;

        // Parse outputs
        let tuple = match outputs {
            IValue::Tuple(t) => t,
            _ => return Err(anyhow!("Expected tuple output from model")),
        };

        if tuple.len() != 2 {
            return Err(anyhow!(
                "Expected 2 outputs (boundary, label), got {}",
                tuple.len()
            ));
        }

        let boundary_logits = match &tuple[0] {
            IValue::Tensor(t) => t.shallow_clone(),
            _ => return Err(anyhow!("Boundary output is not a tensor")),
        };

        let label_logits = match &tuple[1] {
            IValue::Tensor(t) => t.shallow_clone(),
            _ => return Err(anyhow!("Label output is not a tensor")),
        };

        Ok((boundary_logits, label_logits))
    }

    fn detect_boundaries(
        &self,
        logits: &Tensor,
        duration: f64,
        threshold: f64,
    ) -> Result<Vec<f64>> {
        // Apply sigmoid to get probabilities
        let probs = logits.sigmoid();

        // Convert to Vec for processing
        let probs_vec: Vec<f32> = probs
            .view([-1])
            .try_into()
            .context("Failed to convert probabilities to Vec")?;

        // Find frames above threshold
        let mut boundaries = vec![0.0];
        let frame_rate = probs_vec.len() as f64 / duration;

        for (i, &prob) in probs_vec.iter().enumerate() {
            if prob > threshold as f32 && i > 0 {
                let time = i as f64 / frame_rate;
                // Add if far enough from last boundary
                if boundaries.is_empty() || time - boundaries.last().unwrap() > 1.0 {
                    boundaries.push(time);
                }
            }
        }

        // Ensure we have start and end
        if boundaries.is_empty() || boundaries[0] > 0.01 {
            boundaries.insert(0, 0.0);
        }
        if boundaries.is_empty() || (duration - boundaries.last().unwrap()).abs() > 0.01 {
            boundaries.push(duration);
        }

        // Deduplicate close boundaries
        boundaries.dedup_by(|a, b| (*a - *b).abs() < 0.1);

        Ok(boundaries)
    }

    fn predict_labels(
        &self,
        logits: &Tensor,
        boundaries: &[f64],
        duration: f64,
    ) -> Result<Vec<String>> {
        let shape = logits.size();
        let seq_len = shape[1] as usize;
        let num_labels = shape[2] as usize;

        // Convert logits to Vec
        let logits_data: Vec<f32> = logits
            .view([-1])
            .try_into()
            .context("Failed to convert logits")?;

        let mut labels = Vec::new();

        for i in 0..boundaries.len().saturating_sub(1) {
            let start_time = boundaries[i];
            let end_time = boundaries[i + 1];

            // Convert times to frame indices
            let frame_rate = seq_len as f64 / duration;
            let start_frame = ((start_time * frame_rate) as usize).min(seq_len - 1);
            let end_frame = ((end_time * frame_rate) as usize).min(seq_len);

            // Average logits over segment
            let mut avg_logits = vec![0.0f32; num_labels];
            let mut count = 0;

            for frame in start_frame..end_frame {
                for label_idx in 0..num_labels {
                    let idx = frame * num_labels + label_idx;
                    if idx < logits_data.len() {
                        avg_logits[label_idx] += logits_data[idx];
                        count += 1;
                    }
                }
            }

            // Get predicted label
            let label_idx = if count > 0 {
                avg_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(7)
            } else {
                7 // OTHER
            };

            labels.push(LABEL_NAMES[label_idx].to_string());
        }

        Ok(labels)
    }

    fn build_segments(&self, boundaries: &[f64], labels: &[String]) -> Result<Vec<Value>> {
        let mut segments = Vec::new();

        for i in 0..boundaries.len().saturating_sub(1) {
            segments.push(json!({
                "start": boundaries[i],
                "end": boundaries[i + 1],
                "label": labels.get(i).unwrap_or(&"OTHER".to_string()),
            }));
        }

        Ok(segments)
    }
}
