use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{Html, Json, Response},
    routing::{get, post},
    Router,
};
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[derive(Debug, Serialize, Deserialize)]
struct PredictionRequest {
    track_id: Option<String>,
    threshold: Option<f64>,
}

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
    model_path: String,
    python_api_url: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Get configuration from environment
    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "models/best_model.ckpt".to_string());
    let python_api_url =
        std::env::var("PYTHON_API_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());

    let state = AppState {
        model_path,
        python_api_url,
    };

    // Create router
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/health", get(health_handler))
        .route("/api/history", get(history_handler))
        .route("/api/history/:id", get(history_entry_handler))
        .route("/api/history/:id/audio", get(history_audio_handler))
        .route("/api/predict", post(predict_handler))
        .route("/api/upload", post(upload_handler))
        .nest_service("/static", ServeDir::new("web_dashboard/static"))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    tracing::info!("Server running on http://0.0.0.0:{}", port);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../../web_dashboard/templates/index.html"))
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "model_path": state.model_path,
        "python_api_url": state.python_api_url
    }))
}

async fn predict_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut audio_data = Vec::new();
    let mut filename = String::new();

    // Extract file from multipart form
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to read multipart field".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })? {
        if field.name() == Some("audio_file") {
            filename = field.file_name().unwrap_or("unknown").to_string();
            audio_data = field
                .bytes()
                .await
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "Failed to read file data".to_string(),
                            detail: Some(e.to_string()),
                        }),
                    )
                })?
                .to_vec();
        }
    }

    if audio_data.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No audio file provided".to_string(),
                detail: None,
            }),
        ));
    }

    // Save temporary file
    let temp_path = format!("/tmp/{}", filename);
    fs::write(&temp_path, audio_data).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to save temporary file".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    // Call Python API
    let prediction = call_python_api(&state.python_api_url, &temp_path)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Prediction failed".to_string(),
                    detail: Some(e.to_string()),
                }),
            )
        })?;

    // Clean up temporary file
    let _ = fs::remove_file(&temp_path).await;

    Ok(Json(prediction))
}

async fn upload_handler(
    State(_state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let mut audio_data = Vec::new();
    let mut filename = String::new();

    // Extract file from multipart form
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to read multipart field".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })? {
        if field.name() == Some("audio_file") {
            filename = field.file_name().unwrap_or("unknown").to_string();
            audio_data = field
                .bytes()
                .await
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "Failed to read file data".to_string(),
                            detail: Some(e.to_string()),
                        }),
                    )
                })?
                .to_vec();
        }
    }

    if audio_data.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No audio file provided".to_string(),
                detail: None,
            }),
        ));
    }

    // Save to uploads directory
    let uploads_dir = "uploads";
    fs::create_dir_all(uploads_dir).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to create uploads directory".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    let file_path = format!("{}/{}", uploads_dir, filename);
    fs::write(&file_path, audio_data).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to save file".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(serde_json::json!({
        "message": "File uploaded successfully",
        "filename": filename,
        "path": file_path
    })))
}

async fn call_python_api(api_url: &str, file_path: &str) -> Result<PredictionResponse, String> {
    let client = reqwest::Client::new();

    // Create form data
    let file_content = tokio::fs::read(file_path)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;
    let file_name = std::path::Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.wav");

    let form = multipart::Form::new().part(
        "audio_file",
        multipart::Part::bytes(file_content).file_name(file_name.to_string()),
    );

    // Make request
    let response = client
        .post(&format!("{}/predict", api_url))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("API error: {}", error_text));
    }

    let prediction: PredictionResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    Ok(prediction)
}

async fn proxy_python_get(
    api_url: &str,
    path: &str,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    let client = reqwest::Client::new();
    let url = format!("{}{}", api_url.trim_end_matches('/'), path);

    let response = client.get(&url).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: "Failed to contact Python API".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    if !response.status().is_success() {
        let status_code = response.status().as_u16();
        let detail = response.text().await.unwrap_or_default();
        return Err((
            StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(ErrorResponse {
                error: "Upstream API error".to_string(),
                detail: Some(detail),
            }),
        ));
    }

    let json: Value = response.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: "Invalid JSON from API".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(json))
}

async fn proxy_python_audio(
    api_url: &str,
    path: &str,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let client = reqwest::Client::new();
    let url = format!("{}{}", api_url.trim_end_matches('/'), path);

    let response = client.get(&url).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: "Failed to contact Python API".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(ErrorResponse {
                error: "Upstream API error".to_string(),
                detail: Some(detail),
            }),
        ));
    }

    let headers = response.headers().clone();
    let bytes = response.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse {
                error: "Invalid audio payload from API".to_string(),
                detail: Some(e.to_string()),
            }),
        )
    })?;

    let mut resp = Response::new(Body::from(bytes));
    *resp.status_mut() = StatusCode::OK;

    for key in [
        header::CONTENT_TYPE,
        header::CONTENT_LENGTH,
        header::CONTENT_DISPOSITION,
    ] {
        if let Some(value) = headers.get(key.as_str()) {
            if let Ok(value_str) = value.to_str() {
                if let Ok(header_value) = HeaderValue::from_str(value_str) {
                    resp.headers_mut().insert(key, header_value);
                }
            }
        }
    }

    Ok(resp)
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    limit: Option<u32>,
}
async fn history_handler(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    let limit = query.limit.unwrap_or(20);

    proxy_python_get(&state.python_api_url, &format!("/history?limit={}", limit)).await
}

async fn history_entry_handler(
    State(state): State<AppState>,
    Path(history_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    proxy_python_get(&state.python_api_url, &format!("/history/{}", history_id)).await
}

async fn history_audio_handler(
    State(state): State<AppState>,
    Path(history_id): Path<String>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    proxy_python_audio(
        &state.python_api_url,
        &format!("/history/{}/audio", history_id),
    )
    .await
}
