use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tempfile::TempPath;
use tokio::process::Command;
use tracing::{debug, instrument};

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

pub struct MusicStructurePredictor {
    project_root: PathBuf,
    model_path: PathBuf,
    python_bin: PathBuf,
    script_path: PathBuf,
}

impl MusicStructurePredictor {
    pub fn new(project_root: PathBuf, model_path: PathBuf, python_bin: PathBuf) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!(
                "Model checkpoint not found at {}",
                model_path.display()
            ));
        }

        let script_path = project_root.join("src").join("inference").join("cli.py");
        if !script_path.exists() {
            return Err(anyhow!(
                "Inference CLI not found at {}",
                script_path.display()
            ));
        }

        Ok(Self {
            project_root,
            model_path,
            python_bin,
            script_path,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn python_bin(&self) -> &Path {
        &self.python_bin
    }

    pub fn script_path(&self) -> &Path {
        &self.script_path
    }

    #[instrument(skip_all, fields(audio = %audio_path.display()))]
    pub async fn predict(&self, audio_path: &Path, options: &PredictOptions) -> Result<Value> {
        let output_temp = tempfile::Builder::new()
            .prefix("mertalizer_pred")
            .suffix(".json")
            .tempfile()
            .context("Failed to create temporary output file")?;
        let output_path: TempPath = output_temp.into_temp_path();
        let output_path_buf = output_path.to_path_buf();

        let mut command = Command::new(&self.python_bin);
        command
            .current_dir(&self.project_root)
            .arg(&self.script_path)
            .arg(audio_path)
            .arg("--model")
            .arg(&self.model_path)
            .arg("--output")
            .arg(&output_path_buf)
            .arg("--format")
            .arg("json");

        if let Some(threshold) = options.threshold {
            command.arg("--threshold").arg(threshold.to_string());
        }
        if let Some(min_gap) = options.min_gap_seconds {
            command.arg("--min-gap").arg(min_gap.to_string());
        }
        if let Some(min_segment) = options.min_segment_seconds {
            command.arg("--min-segment").arg(min_segment.to_string());
        }
        if let Some(window) = options.smoothing_window {
            command
                .arg("--smoothing-window")
                .arg(window.max(1).to_string());
        }
        if let Some(smooth) = options.smooth {
            if !smooth {
                command.arg("--no-smooth");
            }
        }
        if let Some(position_bias) = options.position_bias {
            if !position_bias {
                command.arg("--no-position-bias");
            }
        }
        if let Some(prominence) = options.prominence {
            command.arg("--prominence").arg(prominence.to_string());
        }
        if options.verbose {
            command.arg("--verbose");
        }

        // Ensure Python can import project modules
        let src_path = self.project_root.join("src");
        let existing_pythonpath = std::env::var("PYTHONPATH").unwrap_or_default();
        let python_path = if existing_pythonpath.is_empty() {
            src_path.display().to_string()
        } else {
            format!("{}:{}", src_path.display(), existing_pythonpath)
        };
        command.env("PYTHONPATH", python_path);

        debug!("Invoking Python CLI for inference");
        let output = command
            .output()
            .await
            .context("Failed to run Python inference CLI")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(anyhow!(
                "Python CLI failed (status {}): {}{}{}",
                output.status,
                stderr,
                if stderr.is_empty() { "" } else { " " },
                stdout
            ));
        }

        let data = tokio::fs::read_to_string(&output_path_buf)
            .await
            .context("Failed to read prediction output")?;
        let json: Value = serde_json::from_str(&data).context("Invalid JSON result")?;

        output_path
            .close()
            .context("Failed to clean up temporary output file")?;

        Ok(json)
    }
}
