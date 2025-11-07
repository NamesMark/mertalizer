use anyhow::{anyhow, Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::task;
use tracing::warn;

#[derive(Debug, Serialize, Deserialize)]
pub struct HistorySummary {
    pub history_id: String,
    pub track_id: Option<String>,
    pub timestamp: Option<String>,
    pub duration: Option<f64>,
    pub segment_count: Option<usize>,
    pub audio_url: Option<String>,
    pub has_audio: bool,
}

#[derive(Clone)]
pub struct HistoryStore {
    root: PathBuf,
    audio_dir: PathBuf,
}

impl HistoryStore {
    pub fn new(root: PathBuf) -> Self {
        let audio_dir = root.join("audio");
        Self { root, audio_dir }
    }

    pub async fn save_result(
        &self,
        result: &mut Value,
        audio_source: Option<&Path>,
    ) -> Result<String> {
        if !result.is_object() {
            return Err(anyhow!("Prediction result must be a JSON object"));
        }

        fs::create_dir_all(&self.root)
            .await
            .context("Failed to create history directory")?;

        let timestamp = ensure_timestamp(result);
        let track_id = result
            .get("track_id")
            .and_then(|v| v.as_str())
            .unwrap_or("track");

        let track_slug = slugify(track_id);
        let compact_ts = timestamp
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>();
        let history_id = format!("{}_{}", compact_ts, track_slug);
        let history_path = self.root.join(format!("{history_id}.json"));

        let mut audio_url: Option<String> = None;
        if let Some(source) = audio_source {
            if source.exists() {
                fs::create_dir_all(&self.audio_dir)
                    .await
                    .context("Failed to create audio history directory")?;

                let extension = source
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("wav");
                let dest = self.audio_dir.join(format!("{history_id}.{extension}"));

                fs::copy(source, &dest)
                    .await
                    .with_context(|| format!("Failed to copy audio to {}", dest.display()))?;

                let relative = dest.strip_prefix(&self.root).unwrap_or(&dest).to_owned();
                let relative_str = relative.to_string_lossy().to_string();

                result
                    .as_object_mut()
                    .unwrap()
                    .insert("history_audio_path".into(), Value::String(relative_str));
                audio_url = Some(format!("/api/history/{history_id}/audio"));
            }
        }

        {
            let obj = result.as_object_mut().unwrap();
            obj.insert("history_id".into(), Value::String(history_id.clone()));
            if let Some(url) = audio_url.clone() {
                obj.insert("audio_url".into(), Value::String(url));
            }
        }

        let data = serde_json::to_string_pretty(result).context("Failed to serialize result")?;
        fs::write(&history_path, data)
            .await
            .with_context(|| format!("Failed to write history file {}", history_path.display()))?;

        Ok(history_id)
    }

    pub async fn list(&self, limit: usize) -> Result<Vec<HistorySummary>> {
        let root = self.root.clone();
        let summaries = task::spawn_blocking(move || -> Result<Vec<HistorySummary>> {
            if !root.exists() {
                return Ok(Vec::new());
            }

            let mut files = std::fs::read_dir(&root)
                .with_context(|| format!("Failed to read {}", root.display()))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
                .collect::<Vec<_>>();

            files.sort();
            files.reverse();

            let mut summaries = Vec::new();
            for path in files.into_iter().take(limit) {
                match std::fs::read_to_string(&path) {
                    Ok(contents) => match serde_json::from_str::<Value>(&contents) {
                        Ok(value) => {
                            if let Some(summary) = summarize_history(&value) {
                                summaries.push(summary);
                            }
                        }
                        Err(err) => {
                            warn!("Failed to parse history {}: {}", path.display(), err);
                        }
                    },
                    Err(err) => {
                        warn!("Failed to read history {}: {}", path.display(), err);
                    }
                }
            }

            Ok(summaries)
        })
        .await??;

        Ok(summaries)
    }

    pub async fn load(&self, history_id: &str) -> Result<Option<Value>> {
        let history_path = self.root.join(format!("{history_id}.json"));
        if !history_path.exists() {
            return Ok(None);
        }

        let data = fs::read_to_string(&history_path)
            .await
            .with_context(|| format!("Failed to read {}", history_path.display()))?;
        let mut value: Value = serde_json::from_str(&data).context("Invalid history JSON")?;

        if value.get("audio_url").and_then(|v| v.as_str()).is_none() {
            if self.audio_file_exists(history_id).await? {
                value.as_object_mut().map(|obj| {
                    obj.insert(
                        "audio_url".into(),
                        Value::String(format!("/api/history/{history_id}/audio")),
                    );
                });
            }
        }

        Ok(Some(value))
    }

    pub async fn audio_file_path(&self, history_id: &str) -> Result<Option<PathBuf>> {
        let audio_dir = self.audio_dir.clone();
        let history_id = history_id.to_string();
        let result = task::spawn_blocking(move || -> Result<Option<PathBuf>> {
            if !audio_dir.exists() {
                return Ok(None);
            }

            for entry in std::fs::read_dir(&audio_dir)? {
                let path = entry?.path();
                if path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(|stem| stem == history_id)
                    .unwrap_or(false)
                {
                    return Ok(Some(path));
                }
            }

            Ok(None)
        })
        .await??;

        Ok(result)
    }

    async fn audio_file_exists(&self, history_id: &str) -> Result<bool> {
        Ok(self.audio_file_path(history_id).await?.is_some())
    }
}

fn ensure_timestamp(result: &mut Value) -> String {
    if let Some(ts) = result
        .get("timestamp")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    {
        return ts;
    }

    let now = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    result
        .as_object_mut()
        .map(|obj| obj.insert("timestamp".into(), Value::String(now.clone())));
    now
}

fn slugify(input: &str) -> String {
    let cleaned = input
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
                ch.to_ascii_lowercase()
            } else if ch.is_ascii_whitespace() || ch == '-' {
                '-'
            } else {
                '-'
            }
        })
        .collect::<String>();

    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "track".into()
    } else {
        trimmed.to_string()
    }
}

fn summarize_history(value: &Value) -> Option<HistorySummary> {
    let history_id = value.get("history_id")?.as_str()?.to_string();
    let track_id = value
        .get("track_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let timestamp = value
        .get("timestamp")
        .and_then(|v| v.as_str())
        .map(String::from);
    let duration = value.get("duration").and_then(|v| v.as_f64());
    let segments = value.get("segments").and_then(|v| v.as_array());
    let audio_url = value
        .get("audio_url")
        .and_then(|v| v.as_str())
        .map(String::from);
    let has_audio = value
        .get("history_audio_path")
        .and_then(|v| v.as_str())
        .is_some()
        || audio_url.is_some();

    Some(HistorySummary {
        history_id,
        track_id,
        timestamp,
        duration,
        segment_count: segments.map(|s| s.len()),
        audio_url,
        has_audio,
    })
}
