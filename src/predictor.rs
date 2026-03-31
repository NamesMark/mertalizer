use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tch::{CModule, Device, IValue, Kind, Tensor};
use tracing::{debug, info, instrument};

/// Audio features extracted via Python/librosa sidecar.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AudioFeatures {
    pub beats: Vec<f64>,
    pub bpm: f64,
    pub key: String,
    pub energy_mean: f64,
    pub energy_max: f64,
    pub brightness: f64,
    pub onset_density: f64,
}

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

/// Feature extractor config loaded from mert_encoder_config.json
#[derive(Debug, Deserialize)]
struct MertConfig {
    sampling_rate: u32,
    hidden_size: usize,
    #[allow(dead_code)]
    num_hidden_layers: usize,
    #[allow(dead_code)]
    n_last_layers: usize,
}

pub struct MusicStructurePredictor {
    /// TorchScript MERT encoder (audio -> embeddings)
    mert_encoder: CModule,
    /// TorchScript boundary+label heads (embeddings -> logits)
    head_model: CModule,
    device: Device,
    project_root: PathBuf,
    python_bin: PathBuf,
    mert_config: MertConfig,
}

impl MusicStructurePredictor {
    pub fn new(project_root: PathBuf, model_path: PathBuf, python_bin: PathBuf) -> Result<Self> {
        let mert_path = project_root.join("models").join("mert_encoder.pt");
        if !mert_path.exists() {
            return Err(anyhow!(
                "MERT encoder not found at {}. Run: PYTHONPATH=ml python ml/export/export_mert_onnx.py",
                mert_path.display()
            ));
        }
        if !model_path.exists() {
            return Err(anyhow!(
                "Head model not found at {}",
                model_path.display()
            ));
        }

        // Load MERT config
        let config_path = project_root.join("models").join("mert_encoder_config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mert_config: MertConfig = serde_json::from_str(&config_str)
            .context("Failed to parse mert_encoder_config.json")?;

        let device = if tch::Cuda::is_available() {
            info!("CUDA available, using GPU");
            Device::Cuda(0)
        } else {
            info!("CUDA not available, using CPU");
            Device::Cpu
        };

        info!("Loading MERT encoder from {}", mert_path.display());
        let mert_encoder = CModule::load_on_device(&mert_path, device)
            .context("Failed to load MERT encoder")?;

        info!("Loading head model from {}", model_path.display());
        let head_model = CModule::load_on_device(&model_path, device)
            .context("Failed to load head model")?;

        info!("Models loaded (MERT hidden_size={}, sr={})",
              mert_config.hidden_size, mert_config.sampling_rate);

        Ok(Self {
            mert_encoder,
            head_model,
            device,
            project_root,
            python_bin,
            mert_config,
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
        debug!("Processing audio file");

        // Step 1: Load audio and normalize (replaces Python feature extractor)
        let (audio_tensor, sr, duration) = self
            .load_and_normalize_audio(audio_path)
            .context("Failed to load audio")?;

        // Step 2: Run MERT encoder in Rust (replaces Python subprocess!)
        debug!("Running MERT encoder (native Rust)");
        let embeddings = self
            .run_mert_encoder(&audio_tensor)
            .context("MERT encoder failed")?;

        // Step 3: Get audio features from Python sidecar (librosa)
        let features = self
            .get_audio_features_python(audio_path, sr)
            .await
            .unwrap_or_default();

        // Step 4: Run boundary+label heads
        debug!("Running boundary/label heads");
        let (boundary_logits, label_logits) = self
            .run_head_model(&embeddings)
            .context("Head model inference failed")?;

        // Step 5: Post-process
        let threshold = options.threshold.unwrap_or(0.1);
        let boundaries = self.detect_boundaries(&boundary_logits, duration, threshold)?;
        let labels = self.predict_labels(&label_logits, &boundaries, duration)?;
        let segments = self.build_segments(&boundaries, &labels)?;

        let result = json!({
            "track_id": audio_path.file_stem().unwrap().to_str().unwrap(),
            "sr": sr,
            "duration": duration,
            "boundaries": boundaries,
            "labels": labels,
            "segments": segments,
            "version": "rust-native@2025-03-31",
            "beats": features.beats,
            "bpm": features.bpm,
            "key": features.key,
            "energy_mean": features.energy_mean,
            "energy_max": features.energy_max,
            "brightness": features.brightness,
            "onset_density": features.onset_density,
            "threshold": threshold,
            "smooth": options.smooth.unwrap_or(true),
            "smoothing_window": options.smoothing_window.unwrap_or(1),
            "min_gap_seconds": options.min_gap_seconds.unwrap_or(3.0),
            "min_segment_seconds": options.min_segment_seconds.unwrap_or(1.5),
            "position_bias": options.position_bias.unwrap_or(true),
        });

        debug!("Prediction complete");
        Ok(result)
    }

    /// Load audio file, resample to MERT's expected rate, normalize to zero-mean unit-variance.
    fn load_and_normalize_audio(&self, audio_path: &Path) -> Result<(Tensor, u32, f64)> {
        let target_sr = self.mert_config.sampling_rate;

        // Use librosa via Python for audio loading (handles mp3, flac, etc.)
        // For WAV files, we could use hound directly, but librosa handles resampling too
        let audio_samples = self.load_audio_samples(audio_path, target_sr)?;
        let num_samples = audio_samples.len();
        let duration = num_samples as f64 / target_sr as f64;

        debug!("Audio: {} samples, {:.2}s @ {}Hz", num_samples, duration, target_sr);

        // Normalize: zero-mean, unit-variance (same as Wav2Vec2FeatureExtractor)
        let sum: f64 = audio_samples.iter().map(|x| *x as f64).sum();
        let mean = sum / num_samples as f64;
        let variance: f64 = audio_samples
            .iter()
            .map(|x| {
                let diff = *x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / num_samples as f64;
        let std = variance.sqrt().max(1e-8);

        let normalized: Vec<f32> = audio_samples
            .iter()
            .map(|x| ((*x as f64 - mean) / std) as f32)
            .collect();

        let tensor = Tensor::from_slice(&normalized)
            .reshape(&[1, num_samples as i64])
            .to_device(self.device);

        Ok((tensor, target_sr, duration))
    }

    /// Load audio samples from file as f32 mono at target sample rate.
    fn load_audio_samples(&self, audio_path: &Path, target_sr: u32) -> Result<Vec<f32>> {
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "wav" => self.load_wav(audio_path, target_sr),
            _ => {
                // For non-WAV formats, fall back to Python/librosa
                // This is a thin wrapper, not full embedding extraction
                self.load_audio_python(audio_path, target_sr)
            }
        }
    }

    /// Load WAV file natively in Rust using hound.
    fn load_wav(&self, audio_path: &Path, target_sr: u32) -> Result<Vec<f32>> {
        let reader = hound::WavReader::open(audio_path)
            .with_context(|| format!("Failed to open WAV: {}", audio_path.display()))?;

        let spec = reader.spec();
        let file_sr = spec.sample_rate;

        // Read all samples as f32
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Failed to read float WAV samples")?,
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1i64 << (bits - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("Failed to read int WAV samples")?
                    .into_iter()
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
        };

        // Convert to mono if stereo
        let mono = if spec.channels > 1 {
            let ch = spec.channels as usize;
            samples
                .chunks(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                .collect()
        } else {
            samples
        };

        // Resample if needed
        if file_sr != target_sr {
            self.resample(&mono, file_sr, target_sr)
        } else {
            Ok(mono)
        }
    }

    /// Resample audio using rubato.
    fn resample(&self, samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
        use rubato::{FftFixedInOut, Resampler};

        let chunk_size = 1024;
        let mut resampler = FftFixedInOut::<f32>::new(
            from_sr as usize,
            to_sr as usize,
            chunk_size,
            1, // mono
        )
        .context("Failed to create resampler")?;

        let input_frames = resampler.input_frames_next();
        let mut output = Vec::new();

        // Process in chunks
        let mut pos = 0;
        while pos + input_frames <= samples.len() {
            let chunk = &samples[pos..pos + input_frames];
            let input_buf: Vec<&[f32]> = vec![chunk];
            let result = resampler
                .process(&input_buf, None)
                .context("Resampling failed")?;
            output.extend_from_slice(&result[0]);
            pos += input_frames;
        }

        // Handle remaining samples by padding with zeros
        if pos < samples.len() {
            let mut chunk: Vec<f32> = samples[pos..].to_vec();
            chunk.resize(input_frames, 0.0);
            let input_buf: Vec<&[f32]> = vec![&chunk];
            let result = resampler
                .process(&input_buf, None)
                .context("Resampling final chunk failed")?;
            let remaining_output =
                ((samples.len() - pos) as f64 * to_sr as f64 / from_sr as f64) as usize;
            output.extend_from_slice(&result[0][..remaining_output.min(result[0].len())]);
        }

        Ok(output)
    }

    /// Load audio via Python/librosa for non-WAV formats.
    fn load_audio_python(&self, audio_path: &Path, target_sr: u32) -> Result<Vec<f32>> {
        use std::process::Command;

        let output = Command::new(&self.python_bin)
            .current_dir(&self.project_root)
            .arg("-c")
            .arg(format!(
                "import librosa, json, sys; \
                 audio, sr = librosa.load('{}', sr={}, mono=True); \
                 print(json.dumps(audio.tolist()))",
                audio_path.display(),
                target_sr
            ))
            .output()
            .context("Failed to run Python for audio loading")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Python audio loading failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let samples: Vec<f32> =
            serde_json::from_str(&stdout).context("Failed to parse audio samples from Python")?;

        Ok(samples)
    }

    /// Run the MERT encoder on normalized audio.
    fn run_mert_encoder(&self, audio_tensor: &Tensor) -> Result<Tensor> {
        let output = self
            .mert_encoder
            .forward_is(&[IValue::Tensor(audio_tensor.shallow_clone())])
            .context("MERT encoder forward pass failed")?;

        match output {
            IValue::Tensor(t) => Ok(t),
            _ => Err(anyhow!("Expected tensor output from MERT encoder")),
        }
    }

    /// Run the boundary+label head model on embeddings.
    fn run_head_model(&self, embeddings: &Tensor) -> Result<(Tensor, Tensor)> {
        let outputs = self
            .head_model
            .forward_is(&[IValue::Tensor(embeddings.shallow_clone())])
            .context("Head model forward pass failed")?;

        let tuple = match outputs {
            IValue::Tuple(t) => t,
            _ => return Err(anyhow!("Expected tuple output from head model")),
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

    /// Get beats and audio features from Python/librosa sidecar.
    async fn get_audio_features_python(
        &self,
        audio_path: &Path,
        sr: u32,
    ) -> Result<AudioFeatures> {
        use tokio::process::Command;

        let script = format!(
            r#"
import librosa, json, numpy as np
audio, sr = librosa.load('{}', sr={}, mono=True)
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=512, units='time')
bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
key_idx = int(np.argmax(np.mean(chroma, axis=1)))
keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
rms = librosa.feature.rms(y=audio, hop_length=512)[0]
centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512, units='time')
onset_density = len(onsets) / (len(audio) / sr) if len(audio) > 0 else 0.0
print(json.dumps({{
    'beats': beats.tolist(),
    'bpm': bpm,
    'key': keys[key_idx],
    'energy_mean': float(np.mean(rms)),
    'energy_max': float(np.max(rms)),
    'brightness': float(np.mean(centroid)),
    'onset_density': onset_density,
}}))
"#,
            audio_path.display(),
            sr
        );

        let output = Command::new(&self.python_bin)
            .current_dir(&self.project_root)
            .arg("-c")
            .arg(&script)
            .output()
            .await
            .context("Failed to run Python for audio features")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            debug!("Python audio features stderr: {}", stderr);
            return Err(anyhow!("Python audio features failed"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parsed: Value =
            serde_json::from_str(&stdout).context("Failed to parse audio features JSON")?;

        Ok(AudioFeatures {
            beats: serde_json::from_value(parsed["beats"].clone()).unwrap_or_default(),
            bpm: parsed["bpm"].as_f64().unwrap_or(0.0),
            key: parsed["key"].as_str().unwrap_or("?").to_string(),
            energy_mean: parsed["energy_mean"].as_f64().unwrap_or(0.0),
            energy_max: parsed["energy_max"].as_f64().unwrap_or(0.0),
            brightness: parsed["brightness"].as_f64().unwrap_or(0.0),
            onset_density: parsed["onset_density"].as_f64().unwrap_or(0.0),
        })
    }

    fn detect_boundaries(
        &self,
        logits: &Tensor,
        duration: f64,
        threshold: f64,
    ) -> Result<Vec<f64>> {
        let probs = logits.sigmoid();
        let mut probs_vec: Vec<f32> = probs
            .view([-1])
            .try_into()
            .context("Failed to convert probabilities to Vec")?;

        // Apply smoothing (moving average, window=5)
        let window = 5usize;
        if probs_vec.len() > window {
            let original = probs_vec.clone();
            let half = window / 2;
            for i in half..probs_vec.len().saturating_sub(half) {
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(original.len());
                let sum: f32 = original[start..end].iter().sum();
                probs_vec[i] = sum / (end - start) as f32;
            }
        }

        let frame_rate = probs_vec.len() as f64 / duration;
        let min_gap_frames = (3.0 * frame_rate) as usize; // 3 second min gap

        // Peak detection with min_gap
        let mut boundaries = vec![0.0];
        let mut last_peak_idx = 0usize;

        for (i, &prob) in probs_vec.iter().enumerate() {
            if prob > threshold as f32 && i > 0 && i - last_peak_idx >= min_gap_frames {
                // Check it's a local peak (higher than neighbors)
                let is_peak = (i == 0 || probs_vec[i] >= probs_vec[i - 1])
                    && (i + 1 >= probs_vec.len() || probs_vec[i] >= probs_vec[i + 1]);
                if is_peak {
                    let time = i as f64 / frame_rate;
                    boundaries.push(time);
                    last_peak_idx = i;
                }
            }
        }

        if boundaries.is_empty() || boundaries[0] > 0.01 {
            boundaries.insert(0, 0.0);
        }
        if boundaries.is_empty() || (duration - boundaries.last().unwrap()).abs() > 0.01 {
            boundaries.push(duration);
        }

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

        let logits_data: Vec<f32> = logits
            .view([-1])
            .try_into()
            .context("Failed to convert logits")?;

        let mut labels = Vec::new();

        for i in 0..boundaries.len().saturating_sub(1) {
            let start_time = boundaries[i];
            let end_time = boundaries[i + 1];

            let frame_rate = seq_len as f64 / duration;
            let start_frame = ((start_time * frame_rate) as usize).min(seq_len - 1);
            let end_frame = ((end_time * frame_rate) as usize).min(seq_len);

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
