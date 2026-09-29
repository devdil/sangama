use super::{
    CONTEXT_LIMIT, Manifest, ShardSpec, check_hash, config, device, engine::Engine, load_manifest,
    model::ShardedModel, weights, wire::*,
};
use crate::{
    protocol::{url, validate_address},
    server,
};
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::net::TcpListener;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Info {
    pub model_id: String,
    pub model_hash: String,
    pub shard: ShardSpec,
    pub device: String,
    #[serde(default = "candle_engine")]
    pub engine: String,
    pub precision: String,
    pub pid: u32,
    #[serde(default)]
    pub busy: bool,
    #[serde(default)]
    pub memory: Option<crate::resources::Memory>,
}

fn candle_engine() -> String {
    "candle".into()
}

struct Session {
    id: String,
    position: usize,
    used: Instant,
}
struct Resident {
    model: Engine,
    session: Option<Session>,
}
#[derive(Clone)]
struct Worker {
    resident: Arc<Mutex<Resident>>,
    info: Info,
    manifest: Manifest,
    token: String,
    http: reqwest::Client,
    allow_next: Vec<SocketAddr>,
}

/// Which engine a worker runs, and for llama.cpp the approved GGUF file in the model directory.
pub struct EngineChoice<'a> {
    pub name: &'a str,
    pub gguf: Option<&'a str>,
}

pub async fn serve(
    dir: &Path,
    index: usize,
    backend: &str,
    engine: EngineChoice<'_>,
    listen: SocketAddr,
    token: String,
    allow_next: Vec<SocketAddr>,
) -> Result<()> {
    crate::security::loopback(listen)?;
    for address in &allow_next {
        crate::security::loopback(*address)?;
    }
    server::validate_token(&token)?;
    let (manifest, hash) = load_manifest(dir)?;
    let spec = manifest
        .shards
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("shard index not in manifest"))?
        .clone();
    let (model, precision, memory) = match engine.name {
        "candle" => {
            ensure!(engine.gguf.is_none(), "--gguf requires --engine llamacpp");
            // Create the device first: it reports a missing build feature clearly, and a CUDA
            // context's own memory is then excluded from the measured budget.
            let device = device(backend)?;
            let memory =
                crate::resources::check(spec.file_bytes, spec.end - spec.start, None, backend)?;
            let file = dir.join(&spec.file);
            check_hash(&file, &spec.sha256)?;
            let cfg = config(dir)?;
            let model = ShardedModel::new(&cfg, weights(&file, &device)?, spec.start, spec.end)?;
            (Engine::Candle(Box::new(model)), "f32".to_string(), memory)
        }
        "llamacpp" => open_llamacpp(dir, &spec, &hash, backend, engine.gguf)?,
        other => anyhow::bail!("unknown engine {other}"),
    };
    let state = Worker {
        resident: Arc::new(Mutex::new(Resident {
            model,
            session: None,
        })),
        info: Info {
            model_id: manifest.model_id.clone(),
            model_hash: hash,
            shard: spec,
            device: backend.into(),
            engine: engine.name.into(),
            precision,
            pid: std::process::id(),
            busy: false,
            memory: Some(memory),
        },
        manifest,
        token: token.clone(),
        http: server::client()?,
        allow_next,
    };
    let (engine_name, precision) = (state.info.engine.clone(), state.info.precision.clone());
    let app = Router::new()
        .route("/v1/qwen/info", get(info))
        .route("/v1/qwen/forward", post(forward))
        .route("/v1/qwen/reset", post(reset))
        .route("/v1/qwen/reserve", post(reserve))
        .layer(DefaultBodyLimit::max(MAX_FRAME_BYTES))
        .layer(middleware::from_fn_with_state(token, server::authenticate))
        .with_state(state);
    let listener = TcpListener::bind(listen).await?;
    tracing::info!(%listen, shard = index, device = backend, engine = %engine_name, %precision, "Qwen shard loaded; ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(feature = "llamacpp")]
fn open_llamacpp(
    dir: &Path,
    spec: &ShardSpec,
    hash: &str,
    backend: &str,
    gguf: Option<&str>,
) -> Result<(Engine, String, crate::resources::Memory)> {
    use anyhow::Context;
    use sangama_llama_stage::{Options, Stage, gpu_memory};
    let gguf = super::approved_gguf(
        dir,
        gguf.context("--engine llamacpp requires --gguf")?,
        hash,
    )?;
    let gpu = backend != "cpu";
    let available = if gpu {
        let (free, _total, name) =
            gpu_memory().context("this build of llama.cpp has no GPU backend")?;
        let prefix = match backend {
            "metal" => "MTL",
            "cuda" => "CUDA",
            "vulkan" => "Vulkan",
            "rocm" => "ROCm",
            _ => anyhow::bail!("unsupported llama.cpp device {backend}"),
        };
        ensure!(
            name.starts_with(prefix),
            "requested {backend} but llama.cpp was built for {name}"
        );
        free
    } else {
        crate::resources::available().context("cannot measure host memory")?
    };
    // Until each worker has its own slice of the GGUF, budget for the whole file.
    let layers = (spec.end - spec.start) as u64;
    let required = gguf.file_bytes + layers * 2 * 4096 * 2 * 64 * 4 + 384 * 1024 * 1024;
    let memory = crate::resources::check_required(required, available, None)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let stage = Stage::open(
        &dir.join(&gguf.file),
        spec.start,
        spec.end,
        &Options {
            gpu,
            context: CONTEXT_LIMIT,
            threads,
        },
    )?;
    ensure!(
        stage.architecture() == "qwen2"
            && stage.layers() == 24
            && stage.hidden_size() == 896
            && stage.vocab_size() == 151936,
        "GGUF does not match the pinned Qwen2.5-0.5B architecture"
    );
    ensure!(
        stage.precision() == gguf.precision,
        "GGUF precision {} differs from gguf.json ({})",
        stage.precision(),
        gguf.precision
    );
    Ok((Engine::LlamaCpp(stage), gguf.precision, memory))
}

#[cfg(not(feature = "llamacpp"))]
fn open_llamacpp(
    _: &Path,
    _: &ShardSpec,
    _: &str,
    _: &str,
    _: Option<&str>,
) -> Result<(Engine, String, crate::resources::Memory)> {
    anyhow::bail!("rebuild with --features llamacpp (or llamacpp-metal, -cuda, -vulkan, -hip)")
}

async fn info(State(state): State<Worker>) -> Json<Info> {
    let mut info = state.info;
    info.busy = state
        .resident
        .try_lock()
        .map(|r| {
            r.session
                .as_ref()
                .is_some_and(|s| s.used.elapsed() < Duration::from_secs(60))
        })
        .unwrap_or(true);
    Json(info)
}

#[derive(Deserialize)]
struct Reset {
    session: String,
}
async fn reserve(State(state): State<Worker>, Json(request): Json<Reset>) -> Response {
    if uuid::Uuid::parse_str(&request.session).is_err() {
        return error(StatusCode::BAD_REQUEST, "invalid session");
    }
    match state.resident.try_lock() {
        Ok(mut resident) => {
            if resident
                .session
                .as_ref()
                .is_some_and(|s| s.used.elapsed() > Duration::from_secs(60))
            {
                resident.model.clear();
                resident.session = None;
            }
            if let Some(session) = &mut resident.session {
                if session.id != request.session {
                    return error(StatusCode::CONFLICT, "worker reserved by another session");
                }
                session.used = Instant::now();
            } else {
                resident.session = Some(Session {
                    id: request.session,
                    position: 0,
                    used: Instant::now(),
                });
            }
            Json(serde_json::json!({"reserved":true,"lease_seconds":60})).into_response()
        }
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "worker busy"),
    }
}
async fn reset(State(state): State<Worker>, Json(request): Json<Reset>) -> Response {
    match state.resident.try_lock() {
        Ok(mut resident) => {
            if resident
                .session
                .as_ref()
                .is_some_and(|s| s.id != request.session)
            {
                return error(StatusCode::CONFLICT, "another session owns this worker");
            }
            resident.model.clear();
            resident.session = None;
            Json(serde_json::json!({"reset":true})).into_response()
        }
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "worker busy"),
    }
}

fn error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (
        status,
        Json(serde_json::json!({"error":message.to_string()})),
    )
        .into_response()
}

fn validate(frame: &Frame, state: &Worker) -> Result<()> {
    let h = &frame.header;
    ensure!(
        h.protocol == 1 && h.model_hash == state.info.model_hash,
        "protocol/model mismatch"
    );
    ensure!(
        uuid::Uuid::parse_str(&h.session).is_ok(),
        "invalid session id"
    );
    ensure!(
        (1..=512).contains(&h.seq_len) && h.position <= CONTEXT_LIMIT - h.seq_len,
        "context/sequence limit exceeded"
    );
    ensure!(!h.route.is_empty() && h.route.len() <= 8, "invalid route");
    let index = state.info.shard.index;
    ensure!(
        h.route.len() == state.manifest.shards.len() - index,
        "incomplete route"
    );
    for (offset, endpoint) in h.route.iter().enumerate() {
        ensure!(endpoint.shard == index + offset, "route gap/overlap");
        validate_address(endpoint.address)?;
        crate::security::loopback(endpoint.address)?;
    }
    if let Some(next) = h.route.get(1) {
        crate::security::next_hop(next.address, &state.allow_next)?;
    }
    ensure!(h.trace.len() == index, "incorrect trace length");
    if index == 0 {
        ensure!(
            h.kind == Kind::Tokens && h.tokens.len() == h.seq_len && frame.values.is_empty(),
            "first shard requires tokens"
        );
        ensure!(
            h.tokens.iter().all(|t| *t < 151936),
            "token outside vocabulary"
        );
    } else {
        ensure!(
            h.kind == Kind::Hidden && h.tokens.is_empty() && frame.values.len() == h.seq_len * 896,
            "hidden tensor shape mismatch"
        );
    }
    Ok(())
}

async fn forward(State(state): State<Worker>, bytes: Bytes) -> Response {
    let frame = match Frame::decode(&bytes).and_then(|f| {
        validate(&f, &state)?;
        Ok(f)
    }) {
        Ok(frame) => frame,
        Err(err) => return error(StatusCode::BAD_REQUEST, err),
    };
    let worker = state.clone();
    let session_id = frame.header.session.clone();
    let calculation = tokio::task::spawn_blocking(move || -> Result<Frame> {
        let mut resident = worker
            .resident
            .try_lock()
            .map_err(|_| anyhow::anyhow!("worker busy"))?;
        if resident
            .session
            .as_ref()
            .is_some_and(|s| s.used.elapsed() > Duration::from_secs(60))
        {
            resident.model.clear();
            resident.session = None;
        }
        let h = &frame.header;
        if let Some(session) = &resident.session {
            ensure!(
                session.id == h.session && session.position == h.position,
                "session mismatch or out-of-order position; reset required"
            );
        } else {
            ensure!(h.position == 0, "new session must begin at position zero");
        }
        let started = Instant::now();
        let mut values =
            match resident
                .model
                .forward(&h.tokens, &frame.values, h.seq_len, h.position)
            {
                Ok(values) => values,
                Err(error) => {
                    resident.model.clear();
                    resident.session = None;
                    return Err(error);
                }
            };
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        resident.session = Some(Session {
            id: h.session.clone(),
            position: h.position + h.seq_len,
            used: Instant::now(),
        });
        let mut header = frame.header;
        header.tokens.clear();
        header.route.remove(0);
        header.trace.push(Trace {
            shard: worker.info.shard.index,
            start: worker.info.shard.start,
            end: worker.info.shard.end,
            forward_ms: elapsed,
        });
        header.kind = if header.route.is_empty() {
            Kind::Logits
        } else {
            Kind::Hidden
        };
        if header.route.is_empty() && header.sample {
            ensure!(
                values.len() == 151936 && values.iter().all(|v| v.is_finite()),
                "invalid final logits"
            );
            let mut best = 0;
            for i in 1..values.len() {
                if values[i] > values[best] {
                    best = i;
                }
            }
            header.tokens = vec![best as u32];
            header.kind = Kind::Sampled;
            values.clear();
        }
        Ok(Frame { header, values })
    })
    .await;
    let result: Result<Frame> = async {
        let frame = calculation??;
        if let Some(next) = frame.header.route.first() {
            let response = state
                .http
                .post(url(next.address, "/v1/qwen/forward"))
                .bearer_auth(&state.token)
                .header("content-type", "application/octet-stream")
                .body(frame.encode()?)
                .send()
                .await?;
            let output = super::wire::response(response).await?;
            ensure!(
                output.valid_output()
                    && output.header.sample == frame.header.sample
                    && output.header.model_hash == frame.header.model_hash
                    && output.header.session == frame.header.session
                    && output.header.position == frame.header.position
                    && output.header.seq_len == frame.header.seq_len
                    && output.header.trace.len() == state.manifest.shards.len(),
                "invalid downstream response"
            );
            Ok(output)
        } else {
            Ok(frame)
        }
    }
    .await;
    match result.and_then(|frame| frame.encode()) {
        Ok(bytes) => ([("content-type", "application/octet-stream")], bytes).into_response(),
        Err(err) => {
            // A partial chain cannot safely retry a position: discard this local session.
            if let Ok(mut resident) = state.resident.try_lock()
                && resident
                    .session
                    .as_ref()
                    .is_some_and(|s| s.id == session_id)
            {
                resident.model.clear();
                resident.session = None;
            }
            error(StatusCode::SERVICE_UNAVAILABLE, err)
        }
    }
}
