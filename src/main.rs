mod history;
mod predictor;

use crate::history::{HistoryStore, HistorySummary};
use crate::predictor::{MusicStructurePredictor, PredictOptions};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{Html, Json, Response},
    routing::{get, post},
    Router,
};
use mime_guess::MimeGuess;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempPath;
use tokio::fs;
use tokio_util::io::ReaderStream;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing::{error, info};

#[derive(Debug, Serialize, Deserialize)]
struct PredictionResponse {
    track_id: String,
    sr: u32,
    duration: f64,
    boundaries: Vec<f64>,
    labels: Vec<String>,
    segments: Vec<Segment>,
    version: String,
    #[serde(default)]
    beats: Option<Vec<f64>>,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    smooth: Option<bool>,
    #[serde(default)]
    smoothing_window: Option<u32>,
    #[serde(default)]
    min_gap_seconds: Option<f64>,
    #[serde(default)]
    min_segment_seconds: Option<f64>,
    #[serde(default)]
    position_bias: Option<bool>,
    #[serde(default)]
    debug: Option<Value>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    history_id: Option<String>,
    #[serde(default)]
    audio_url: Option<String>,
    #[serde(default)]
    history_audio_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Segment {
    start: f64,
    end: f64,
    label: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ErrorResponse {
    error: String,
    detail: Option<String>,
}

#[derive(Clone)]
struct AppState {
    predictor: Arc<MusicStructurePredictor>,
    history: Arc<HistoryStore>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PredictionQuery {
    #[serde(default)]
    track_id: Option<String>,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    smooth: Option<bool>,
    #[serde(default)]
    smoothing_window: Option<u32>,
    #[serde(default, alias = "min_gap_seconds")]
    min_gap: Option<f64>,
    #[serde(default, alias = "min_segment_seconds")]
    min_segment: Option<f64>,
    #[serde(default)]
    position_bias: Option<bool>,
    #[serde(default)]
    prominence: Option<f64>,
}

#[derive(Default, Clone)]
struct PredictionOverrides {
    track_id: Option<String>,
    threshold: Option<f64>,
    smooth: Option<bool>,
    smoothing_window: Option<u32>,
    min_gap: Option<f64>,
    min_segment: Option<f64>,
    position_bias: Option<bool>,
    prominence: Option<f64>,
}

impl PredictionOverrides {
    fn apply_field(&mut self, name: &str, value: &str) {
        match name {
            "track_id" => self.track_id = Some(value.to_string()),
            "threshold" => self.threshold = parse_f64(value),
            "smooth" => self.smooth = parse_bool(value),
            "smoothing_window" => self.smoothing_window = parse_u32(value),
            "min_gap" | "min_gap_seconds" => self.min_gap = parse_f64(value),
            "min_segment" | "min_segment_seconds" => self.min_segment = parse_f64(value),
            "position_bias" => self.position_bias = parse_bool(value),
            "prominence" => self.prominence = parse_f64(value),
            _ => {}
        }
    }

    fn merge(self, query: PredictionQuery) -> (Option<String>, PredictOptions) {
        let track_id = self.track_id.or(query.track_id);
        let options = PredictOptions {
            threshold: self.threshold.or(query.threshold),
            smooth: self.smooth.or(query.smooth),
            smoothing_window: self.smoothing_window.or(query.smoothing_window),
            min_gap_seconds: self.min_gap.or(query.min_gap),
            min_segment_seconds: self.min_segment.or(query.min_segment),
            position_bias: self.position_bias.or(query.position_bias),
            prominence: self.prominence.or(query.prominence),
            verbose: true,
        };
        (track_id, options)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/mertalizer_traced.pt".to_string());

    // Try to use venv Python first, fallback to system Python
    let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| {
        let venv_python = PathBuf::from("venv/bin/python");
        if venv_python.exists() {
            venv_python.display().to_string()
        } else {
            "python3".to_string()
        }
    });
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let predictor = Arc::new(MusicStructurePredictor::new(
        project_root.clone(),
        PathBuf::from(&model_path),
        PathBuf::from(python_bin),
    )?);

    let history_dir = project_root.join("data").join("history");
    let history = Arc::new(HistoryStore::new(history_dir));

    let state = AppState { predictor, history };

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/health", get(health_handler))
        .route("/api/history", get(history_handler))
        .route("/api/history/:id", get(history_entry_handler))
        .route("/api/history/:id/audio", get(history_audio_handler))
        .route("/api/predict", post(predict_handler))
        .route("/api/upload", post(upload_handler))
        .nest_service("/static", ServeDir::new("static"))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state);

    let address = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&address).await?;
    info!("Server running on http://{}", address);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../templates/index.html"))
}

async fn health_handler(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "model_path": state.predictor.model_path().display().to_string(),
        "python_bin": state.predictor.python_bin().display().to_string(),
        "cli_path": state.predictor.script_path().display().to_string()
    }))
}

async fn predict_handler(
    State(state): State<AppState>,
    Query(query): Query<PredictionQuery>,
    mut multipart: Multipart,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut audio_data = Vec::new();
    let mut filename = None;
    let mut overrides = PredictionOverrides::default();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        error_response(
            StatusCode::BAD_REQUEST,
            "Failed to read multipart field",
            Some(e.to_string()),
        )
    })? {
        let field_name = field.name().map(|n| n.to_string());
        if let Some(name) = field_name.as_deref() {
            if name == "audio_file" {
                filename = field.file_name().map(|f| f.to_string());
                let bytes = field.bytes().await.map_err(|e| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "Failed to read audio data",
                        Some(e.to_string()),
                    )
                })?;
                audio_data = bytes.to_vec();
            } else {
                let value = field.text().await.map_err(|e| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "Failed to parse multipart field",
                        Some(e.to_string()),
                    )
                })?;
                overrides.apply_field(name, value.trim());
            }
        }
    }

    if audio_data.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "No audio file provided",
            None,
        ));
    }

    let original_name = filename.unwrap_or_else(|| "upload.wav".to_string());
    let extension = infer_extension(&original_name);
    let temp_audio = tempfile::Builder::new()
        .prefix("mertalizer_audio")
        .suffix(&format!(".{}", extension))
        .tempfile()
        .map_err(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to create temporary file",
                Some(e.to_string()),
            )
        })?;
    let temp_audio_path: TempPath = temp_audio.into_temp_path();
    let audio_path_buf = temp_audio_path.to_path_buf();
    fs::write(&audio_path_buf, &audio_data).await.map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to write uploaded audio",
            Some(e.to_string()),
        )
    })?;

    let (track_override, options) = overrides.merge(query);

    let mut result = state
        .predictor
        .predict(audio_path_buf.as_path(), &options)
        .await
        .map_err(|e| {
            error!("Prediction failed for {}: {}", original_name, e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Prediction failed",
                Some(e.to_string()),
            )
        })?;

    if let Some(track_id) = track_override {
        if let Some(obj) = result.as_object_mut() {
            obj.insert("track_id".into(), Value::String(track_id));
        }
    }

    state
        .history
        .save_result(&mut result, Some(audio_path_buf.as_path()))
        .await
        .map_err(|e| {
            error!("Failed to persist history: {}", e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to save history",
                Some(e.to_string()),
            )
        })?;

    if let Err(e) = temp_audio_path.close() {
        error!("Failed to remove temporary file: {}", e);
    }

    let prediction: PredictionResponse = serde_json::from_value(result).map_err(|e| {
        error!("Invalid prediction payload: {}", e);
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Invalid prediction payload",
            Some(e.to_string()),
        )
    })?;

    Ok(Json(prediction))
}

async fn upload_handler(
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    let mut audio_data = Vec::new();
    let mut filename = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        error_response(
            StatusCode::BAD_REQUEST,
            "Failed to read multipart field",
            Some(e.to_string()),
        )
    })? {
        if field.name() == Some("audio_file") {
            filename = field.file_name().map(|f| f.to_string());
            audio_data = field
                .bytes()
                .await
                .map_err(|e| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "Failed to read file data",
                        Some(e.to_string()),
                    )
                })?
                .to_vec();
        }
    }

    if audio_data.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "No audio file provided",
            None,
        ));
    }

    let name = filename.unwrap_or_else(|| "upload.wav".to_string());
    let uploads_dir = Path::new("uploads");
    fs::create_dir_all(uploads_dir).await.map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to prepare uploads directory",
            Some(e.to_string()),
        )
    })?;

    let file_path = uploads_dir.join(&name);
    fs::write(&file_path, audio_data).await.map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to save uploaded file",
            Some(e.to_string()),
        )
    })?;

    Ok(Json(json!({
        "message": "File uploaded successfully",
        "filename": name,
        "path": file_path.display().to_string()
    })))
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    limit: Option<u32>,
}

async fn history_handler(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    let limit = query.limit.unwrap_or(20).clamp(1, 100) as usize;
    let entries: Vec<HistorySummary> = state.history.list(limit).await.map_err(|e| {
        error!("Failed to list history: {}", e);
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to load history",
            Some(e.to_string()),
        )
    })?;

    Ok(Json(json!({ "entries": entries })))
}

async fn history_entry_handler(
    State(state): State<AppState>,
    AxumPath(history_id): AxumPath<String>,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    match state.history.load(&history_id).await {
        Ok(Some(entry)) => Ok(Json(entry)),
        Ok(None) => Err(error_response(
            StatusCode::NOT_FOUND,
            "History entry not found",
            None,
        )),
        Err(e) => {
            error!("Failed to load history {}: {}", history_id, e);
            Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to load history entry",
                Some(e.to_string()),
            ))
        }
    }
}

async fn history_audio_handler(
    State(state): State<AppState>,
    AxumPath(history_id): AxumPath<String>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let audio_path = state
        .history
        .audio_file_path(&history_id)
        .await
        .map_err(|e| {
            error!("Failed to locate audio for {}: {}", history_id, e);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to locate history audio",
                Some(e.to_string()),
            )
        })?;

    let audio_path = match audio_path {
        Some(path) => path,
        None => {
            return Err(error_response(
                StatusCode::NOT_FOUND,
                "Audio not available for entry",
                None,
            ))
        }
    };

    let file = fs::File::open(&audio_path).await.map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to open audio file",
            Some(e.to_string()),
        )
    })?;

    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    let mime = MimeGuess::from_path(&audio_path)
        .first_or_octet_stream()
        .to_string();
    let filename = audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("history_audio");

    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&mime).unwrap_or(HeaderValue::from_static("audio/mpeg")),
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("inline; filename=\"{}\"", filename))
            .unwrap_or(HeaderValue::from_static("inline")),
    );

    Ok(response)
}

fn error_response(
    status: StatusCode,
    message: &str,
    detail: Option<String>,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: message.to_string(),
            detail,
        }),
    )
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok()
}

fn infer_extension(filename: &str) -> String {
    let lower = filename.to_ascii_lowercase();
    for ext in ["wav", "mp3", "flac", "m4a", "aac", "ogg"] {
        if lower.ends_with(ext) {
            return ext.to_string();
        }
    }
    "wav".to_string()
}
