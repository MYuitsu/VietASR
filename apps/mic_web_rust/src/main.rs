use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;
use tokio::fs;
use tokio::process::Command;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct AppState {
    config: Arc<AppConfig>,
}

#[derive(Clone, Debug)]
struct AppConfig {
    host: String,
    port: u16,
    repo_root: PathBuf,
    python_bin: String,
    ffmpeg_bin: String,
    checkpoint: PathBuf,
    tokens: PathBuf,
    beam_size: u32,
    method: String,
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
}

#[derive(Deserialize, Serialize)]
struct TranscribeResponse {
    transcript: String,
}

#[derive(Deserialize, Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() {
    let config = AppConfig::from_env().expect("failed to load app config");
    let static_dir = config.repo_root.join("apps/mic_web_rust/static");
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("invalid bind address");

    let state = AppState {
        config: Arc::new(config),
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/transcribe", post(transcribe))
        .nest_service("/", ServeDir::new(static_dir))
        .with_state(state);

    println!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind listener");
    axum::serve(listener, app).await.expect("server error");
}

impl AppConfig {
    fn from_env() -> Result<Self, String> {
        let repo_root = env::var("VIETASR_REPO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/home/nguyenthaiduy277/VietASR-1"));

        let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(3000);

        let python_bin = env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
        let ffmpeg_bin = env::var("FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string());

        let checkpoint = env::var("VIETASR_CHECKPOINT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_root.join("hf_models/viet_iter3_pseudo_label/exp/epoch-12.pt"));

        let tokens = env::var("VIETASR_TOKENS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                repo_root.join("hf_models/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt")
            });

        let beam_size = env::var("VIETASR_BEAM_SIZE")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(4);

        let method =
            env::var("VIETASR_METHOD").unwrap_or_else(|_| "modified_beam_search".to_string());

        Ok(Self {
            host,
            port,
            repo_root,
            python_bin,
            ffmpeg_bin,
            checkpoint,
            tokens,
            beam_size,
            method,
        })
    }
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { ok: true })
}

async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<TranscribeResponse>, AppError> {
    let temp_dir = tempdir().map_err(internal)?;
    let input_path = temp_dir.path().join("input.webm");
    let wav_path = temp_dir.path().join("input.wav");

    let mut found_audio = false;
    while let Some(field) = multipart.next_field().await.map_err(bad_request)? {
        if field.name() != Some("audio") {
            continue;
        }

        let bytes = field.bytes().await.map_err(bad_request)?;
        fs::write(&input_path, bytes).await.map_err(internal)?;
        found_audio = true;
        break;
    }

    if !found_audio {
        return Err(AppError::bad_request("Missing multipart field 'audio'"));
    }

    convert_to_wav(&state.config, &input_path, &wav_path).await?;
    let transcript = run_asr(&state.config, &wav_path).await?;

    Ok(Json(TranscribeResponse { transcript }))
}

async fn convert_to_wav(config: &AppConfig, input: &Path, output: &Path) -> Result<(), AppError> {
    let status = Command::new(&config.ffmpeg_bin)
        .arg("-y")
        .arg("-i")
        .arg(input)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg(output)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .await
        .map_err(internal)?;

    if !status.success() {
        return Err(AppError::internal("ffmpeg failed to convert uploaded audio"));
    }

    Ok(())
}

async fn run_asr(config: &AppConfig, wav_path: &Path) -> Result<String, AppError> {
    let helper = config.repo_root.join("apps/mic_web_rust/transcribe_once.py");
    let py_path = format!(
        "{}:{}:{}",
        config.repo_root.display(),
        config.repo_root.join("ASR/zipformer").display(),
        config.repo_root.join("icefall").display()
    );

    let output = Command::new(&config.python_bin)
        .arg(&helper)
        .arg("--repo-root")
        .arg(&config.repo_root)
        .arg("--checkpoint")
        .arg(&config.checkpoint)
        .arg("--tokens")
        .arg(&config.tokens)
        .arg("--method")
        .arg(&config.method)
        .arg("--beam-size")
        .arg(config.beam_size.to_string())
        .arg("--sound-file")
        .arg(wav_path)
        .env("PYTHONPATH", py_path)
        .output()
        .await
        .map_err(internal)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::internal(format!(
            "ASR helper failed: {}",
            stderr.trim()
        )));
    }

    let response: TranscribeResponse =
        serde_json::from_slice(&output.stdout).map_err(internal)?;
    Ok(response.transcript)
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}

fn bad_request<E: std::fmt::Display>(error: E) -> AppError {
    AppError::bad_request(error.to_string())
}

fn internal<E: std::fmt::Display>(error: E) -> AppError {
    AppError::internal(error.to_string())
}
