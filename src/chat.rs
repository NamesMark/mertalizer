//! Chat module: LLM-powered music feedback using analysis context.
//!
//! Stores chat messages alongside history entries and streams responses
//! from the Anthropic Claude API via SSE.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::fs;
use tracing::{debug, error};

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "user" or "assistant"
    pub content: String,
}

/// Manages chat persistence and LLM interaction.
#[derive(Clone)]
pub struct ChatService {
    history_dir: PathBuf,
    api_key: Option<String>,
    model: String,
}

impl ChatService {
    pub fn new(history_dir: PathBuf) -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
        let model =
            std::env::var("CHAT_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());

        if api_key.is_some() {
            tracing::info!("Chat enabled (model={})", model);
        } else {
            tracing::warn!("ANTHROPIC_API_KEY not set — chat feature disabled");
        }

        Self {
            history_dir,
            api_key,
            model,
        }
    }

    pub fn is_available(&self) -> bool {
        self.api_key.is_some()
    }

    /// Load chat messages for a history entry.
    pub async fn load_chat(&self, history_id: &str) -> Result<Vec<ChatMessage>> {
        let path = self.chat_path(history_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let data = fs::read_to_string(&path).await?;
        let messages: Vec<ChatMessage> = serde_json::from_str(&data)?;
        Ok(messages)
    }

    /// Save chat messages for a history entry.
    pub async fn save_chat(&self, history_id: &str, messages: &[ChatMessage]) -> Result<()> {
        let path = self.chat_path(history_id);
        let data = serde_json::to_string_pretty(messages)?;
        fs::write(&path, data).await?;
        Ok(())
    }

    /// Delete chat for a history entry.
    pub async fn delete_chat(&self, history_id: &str) -> Result<()> {
        let path = self.chat_path(history_id);
        if path.exists() {
            fs::remove_file(&path).await?;
        }
        Ok(())
    }

    /// Stream a chat response from the LLM. Returns the full response text.
    /// The `on_chunk` callback is called for each text chunk as it arrives.
    pub async fn send_message(
        &self,
        history_id: &str,
        user_message: &str,
        analysis: &Value,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("ANTHROPIC_API_KEY not set"))?;

        // Load existing chat history
        let mut messages = self.load_chat(history_id).await?;

        // Build system prompt with analysis context
        let system_prompt = build_system_prompt(analysis);

        // Add user message
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_message.to_string(),
        });

        // Build API request
        let api_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let body = json!({
            "model": self.model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": api_messages,
            "stream": true,
        });

        debug!("Calling Anthropic API (model={})", self.model);

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to call Anthropic API")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Anthropic API error ({}): {}", status, text));
        }

        // Stream the response
        let mut full_response = String::new();
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Stream read error")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }
                let data = &line[6..];
                if data == "[DONE]" {
                    continue;
                }

                if let Ok(event) = serde_json::from_str::<Value>(data) {
                    // Extract text from content_block_delta events
                    if event.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                        if let Some(text) = event
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            full_response.push_str(text);
                            let _ = tx.send(text.to_string()).await;
                        }
                    }
                }
            }
        }

        // Save assistant response to history
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: full_response.clone(),
        });
        self.save_chat(history_id, &messages).await?;

        Ok(full_response)
    }

    fn chat_path(&self, history_id: &str) -> PathBuf {
        self.history_dir.join(format!("{}_chat.json", history_id))
    }
}

/// Build a system prompt that includes the full analysis context.
fn build_system_prompt(analysis: &Value) -> String {
    let track_id = analysis
        .get("track_id")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");
    let duration = analysis.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let bpm = analysis.get("bpm").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let key = analysis
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let energy = analysis
        .get("energy_mean")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let brightness = analysis
        .get("brightness")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let onset_density = analysis
        .get("onset_density")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let segments = analysis
        .get("segments")
        .and_then(|v| v.as_array())
        .map(|segs| {
            segs.iter()
                .enumerate()
                .map(|(i, seg)| {
                    let label = seg.get("label").and_then(|v| v.as_str()).unwrap_or("?");
                    let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let end = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    format!(
                        "  {}. {} ({:.1}s - {:.1}s, {:.1}s)",
                        i + 1,
                        label,
                        start,
                        end,
                        end - start
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| "  No segments detected".to_string());

    let labels: Vec<&str> = analysis
        .get("labels")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let unique_labels: std::collections::HashSet<&str> = labels.iter().copied().collect();

    format!(
        r#"You are a music production and structure analysis expert. You have deep knowledge of songwriting, arrangement, mixing concepts, and music theory.

A song has been analyzed using a music-native AI model (MERT). Here are the analysis results:

SONG ANALYSIS:
- Track: {track_id}
- Duration: {duration:.1}s ({:.0}:{:02.0})
- BPM: {bpm:.0}
- Estimated Key: {key}
- Average Energy (RMS): {energy:.4}
- Brightness (Spectral Centroid): {brightness:.0} Hz
- Onset Density: {onset_density:.1} onsets/sec

DETECTED STRUCTURE:
{segments}

LABELS FOUND: {unique}
TOTAL SECTIONS: {n_sections}

Based on this analysis, help the user understand and improve their song. You can:
- Point out missing sections (e.g., no bridge, no pre-chorus, no intro/outro)
- Compare to common structures (ABABCB, AABA, verse-chorus, etc.)
- Suggest arrangement improvements
- Discuss energy flow and dynamics across sections
- Offer songwriting and production advice for specific sections
- Note if sections seem too long or too short

Be concise, specific, and actionable. Reference the actual section timings when relevant."#,
        duration / 60.0,
        duration % 60.0,
        unique = unique_labels
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .join(", "),
        n_sections = labels.len(),
    )
}
