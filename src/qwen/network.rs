use super::{
    CONTEXT_LIMIT, Manifest, ShardSpec, check_hash, config, device, engine::Engine, load_manifest,
    model::ShardedModel, weights, wire::*,
};
use crate::{
    protocol::{url, validate_address},
    server,
};
use anyhow::{Context, Result, ensure};
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
    /// SHA-256 of the GGUF a llama.cpp worker loaded. Quantizing is not reproducible across
    /// machines, so a route must not mix different GGUF files of the same precision.
    #[serde(default)]
    pub weights_sha256: Option<String>,
}

/// Test-only: `SANGAMA_SIMULATE_MS_PER_LAYER_TOKEN` adds this many milliseconds per layer and
/// position to every forward, so a fast container can stand in for a slower device.
fn simulated_ms_per_layer_token() -> Option<f64> {
    static VALUE: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("SANGAMA_SIMULATE_MS_PER_LAYER_TOKEN")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1000.0)
    })
}

fn candle_engine() -> String {
    "candle".into()
}

struct Session {
    id: String,
    position: usize,
    used: Instant,
    /// The state before the latest speculative batch, and that batch's inputs.
    draft: Option<Draft>,
}
struct Draft {
    position: usize,
    state: Vec<u8>,
    tokens: Vec<u32>,
    values: Vec<f32>,
    seq_len: usize,
}
struct Resident {
    model: Engine,
    session: Option<Session>,
}
/// A detached frame waiting to be passed to the next stage.
struct Outgoing {
    next: SocketAddr,
    bytes: Vec<u8>,
    sent: Header,
}
/// A detached result kept by the last stage until the client collects it.
struct Delivered {
    session: String,
    position: usize,
    bytes: Vec<u8>,
}
#[derive(Clone)]
struct Worker {
    resident: Arc<Mutex<Resident>>,
    results: Arc<tokio::sync::watch::Sender<Option<Arc<Delivered>>>>,
    /// Detached frames leave in the order this stage computed them, one at a time, so a
    /// pipelined prompt's chunks cannot overtake each other.
    outbox: tokio::sync::mpsc::UnboundedSender<Outgoing>,
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
    /// Cap on the memory this worker may use, e.g. to act like an average laptop.
    pub memory_budget_mib: Option<u64>,
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
    let (model, precision, weights_sha256, memory) = match engine.name {
        "candle" => {
            ensure!(engine.gguf.is_none(), "--gguf requires --engine llamacpp");
            ensure!(
                manifest.sliced.is_none(),
                "this model is published as GGUF slices; use --engine llamacpp"
            );
            // Create the device first: it reports a missing build feature clearly, and a CUDA
            // context's own memory is then excluded from the measured budget.
            let device = device(backend)?;
            let memory = crate::resources::check(
                spec.file_bytes,
                spec.end - spec.start,
                engine.memory_budget_mib,
                backend,
            )?;
            let file = dir.join(&spec.file);
            check_hash(&file, &spec.sha256)?;
            let cfg = config(dir)?;
            let model = weights(&file, &device)
                .and_then(|vb| Ok(ShardedModel::new(&cfg, vb, spec.start, spec.end)?))
                .map_err(|error| {
                    if error.to_string().contains("UNSUPPORTED_PTX_VERSION") {
                        error.context(
                            "Sangama was built with a CUDA toolkit newer than this driver supports; \
                             rebuild with a toolkit no newer than the CUDA version nvidia-smi reports",
                        )
                    } else {
                        error
                    }
                })?;
            (
                Engine::Candle(Box::new(model)),
                "f32".to_string(),
                None,
                memory,
            )
        }
        "llamacpp" => open_llamacpp(dir, &manifest, &spec, &hash, backend, &engine)?,
        other => anyhow::bail!("unknown engine {other}"),
    };
    let (outbox, mut outgoing) = tokio::sync::mpsc::unbounded_channel::<Outgoing>();
    let state = Worker {
        outbox,
        resident: Arc::new(Mutex::new(Resident {
            model,
            session: None,
        })),
        results: Arc::new(tokio::sync::watch::channel(None).0),
        info: Info {
            model_id: manifest.model_id.clone(),
            model_hash: hash,
            shard: spec,
            device: backend.into(),
            engine: engine.name.into(),
            weights_sha256,
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
    {
        let state = state.clone();
        tokio::spawn(async move {
            while let Some(Outgoing { next, bytes, sent }) = outgoing.recv().await {
                let passed: Result<()> = async {
                    let response = state
                        .http
                        .post(url(next, "/v1/qwen/forward"))
                        .bearer_auth(&state.token)
                        .header("content-type", "application/octet-stream")
                        .body(bytes)
                        .send()
                        .await?;
                    let reply = super::wire::response(response).await?;
                    ensure!(reply.accepts(&sent), "invalid downstream acknowledgement");
                    Ok(())
                }
                .await;
                if let Err(error) = passed {
                    tracing::warn!(%error, session = %sent.session, "passing a detached frame on failed");
                    discard(&state, &sent.session);
                }
            }
        });
    }
    let (engine_name, precision) = (state.info.engine.clone(), state.info.precision.clone());
    let app = Router::new()
        .route("/v1/qwen/info", get(info))
        .route("/v1/qwen/forward", post(forward))
        .route("/v1/qwen/result", post(result))
        .route("/v1/qwen/reset", post(reset))
        .route("/v1/qwen/reserve", post(reserve))
        .layer(DefaultBodyLimit::max(MAX_FRAME_BYTES))
        .layer(middleware::from_fn_with_state(token, server::authenticate))
        .with_state(state);
    let listener = TcpListener::bind(listen).await?;
    if let Some(ms) = simulated_ms_per_layer_token() {
        tracing::warn!(
            ms_per_layer_token = ms,
            "device simulation: adding artificial compute delay; not for real use"
        );
    }
    tracing::info!(%listen, shard = index, device = backend, engine = %engine_name, %precision, "Qwen shard loaded; ready");
    axum::serve(server::nodelay(listener), app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(feature = "llamacpp")]
fn open_llamacpp(
    dir: &Path,
    manifest: &Manifest,
    spec: &ShardSpec,
    hash: &str,
    backend: &str,
    engine: &EngineChoice<'_>,
) -> Result<(Engine, String, Option<String>, crate::resources::Memory)> {
    use anyhow::Context;
    let gguf = engine.gguf;
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
    let memory = crate::resources::check_required(required, available, engine.memory_budget_mib)?;
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
        stage.architecture() == manifest.architecture()
            && stage.layers() == manifest.layers()
            && stage.hidden_size() == manifest.hidden_size()
            && stage.vocab_size() == manifest.vocab_size(),
        "GGUF architecture or shape does not match the manifest"
    );
    ensure!(
        stage.precision() == gguf.precision,
        "GGUF precision {} differs from gguf.json ({})",
        stage.precision(),
        gguf.precision
    );
    Ok((
        Engine::LlamaCpp(stage),
        gguf.precision,
        Some(gguf.sha256),
        memory,
    ))
}

#[cfg(not(feature = "llamacpp"))]
fn open_llamacpp(
    _: &Path,
    _: &Manifest,
    _: &ShardSpec,
    _: &str,
    _: &str,
    _: &EngineChoice<'_>,
) -> Result<(Engine, String, Option<String>, crate::resources::Memory)> {
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
                    draft: None,
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
    ensure!(
        !h.route.is_empty() && h.route.len() <= super::MAX_SHARDS,
        "invalid route"
    );
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
            h.tokens
                .iter()
                .all(|t| (*t as usize) < state.manifest.vocab_size()),
            "token outside vocabulary"
        );
    } else {
        ensure!(
            h.kind == Kind::Hidden
                && h.tokens.is_empty()
                && frame.values.len() == h.seq_len * state.manifest.hidden_size(),
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
    let (position, seq_len) = (frame.header.position, frame.header.seq_len);
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
        let resident = &mut *resident;
        if let Some(session) = &mut resident.session {
            ensure!(
                session.id == h.session,
                "session mismatch or out-of-order position; reset required"
            );
            let previous = session.draft.take();
            if session.position != h.position {
                // The client accepted only part of the last speculative batch: restore the state
                // from before it and replay the accepted inputs, which this stage kept.
                let d = previous
                    .filter(|d| d.position < h.position && h.position < d.position + d.seq_len)
                    .context("session mismatch or out-of-order position; reset required")?;
                let keep = h.position - d.position;
                let width = d.values.len() / d.seq_len;
                let replayed = resident.model.load_state(&d.state).and_then(|()| {
                    resident.model.forward(
                        &d.tokens[..d.tokens.len().min(keep)],
                        &d.values[..(keep * width).min(d.values.len())],
                        keep,
                        d.position,
                    )
                });
                if let Err(error) = replayed {
                    resident.model.clear();
                    resident.session = None;
                    return Err(error);
                }
                session.position = h.position;
            }
        } else {
            ensure!(h.position == 0, "new session must begin at position zero");
        }
        let draft = if h.speculative {
            Some(Draft {
                position: h.position,
                state: resident.model.save_state()?,
                tokens: h.tokens.clone(),
                values: frame.values.clone(),
                seq_len: h.seq_len,
            })
        } else {
            None
        };
        let started = Instant::now();
        let started_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0);
        let last = h.route.len() == 1;
        let computed = if last && h.sample && h.speculative {
            resident
                .model
                .greedy(&h.tokens, &frame.values, h.seq_len, h.position)
                .map(|ids| (Some(ids), vec![]))
        } else {
            resident
                .model
                .forward(&h.tokens, &frame.values, h.seq_len, h.position)
                .map(|values| (None, values))
        };
        let (drafted, mut values) = match computed {
            Ok(result) => result,
            Err(error) => {
                resident.model.clear();
                resident.session = None;
                return Err(error);
            }
        };
        if let Some(ms) = simulated_ms_per_layer_token() {
            let layers = (worker.info.shard.end - worker.info.shard.start) as f64;
            std::thread::sleep(Duration::from_secs_f64(
                ms * layers * h.seq_len as f64 / 1000.0,
            ));
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        resident.session = Some(Session {
            id: h.session.clone(),
            position: h.position + h.seq_len,
            used: Instant::now(),
            draft,
        });
        let mut header = frame.header;
        header.tokens.clear();
        header.route.remove(0);
        header.trace.push(Trace {
            shard: worker.info.shard.index,
            start: worker.info.shard.start,
            end: worker.info.shard.end,
            forward_ms: elapsed,
            started_ms,
        });
        header.kind = if header.route.is_empty() {
            Kind::Logits
        } else {
            Kind::Hidden
        };
        if let Some(ids) = drafted {
            header.tokens = ids;
            header.kind = Kind::Sampled;
        } else if header.route.is_empty() && header.sample {
            ensure!(
                values.len() == worker.manifest.vocab_size()
                    && values.iter().all(|v| v.is_finite()),
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
        if frame.header.detached {
            return detach(&state, frame);
        }
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
                output.valid_output(state.manifest.vocab_size())
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
            tracing::warn!(error = %err, session = %session_id, position, seq_len, "forward failed; session discarded");
            discard(&state, &session_id);
            error(StatusCode::SERVICE_UNAVAILABLE, err)
        }
    }
}

/// A partial chain cannot safely retry a position: discard this local session.
fn discard(state: &Worker, session: &str) {
    if let Ok(mut resident) = state.resident.try_lock()
        && resident.session.as_ref().is_some_and(|s| s.id == session)
    {
        resident.model.clear();
        resident.session = None;
    }
}

/// Queues a detached frame for the next stage, or keeps it at the last stage
/// for the client, and returns this stage's acknowledgement.
fn detach(state: &Worker, frame: Frame) -> Result<Frame> {
    let bytes = frame.encode()?;
    let mut ack = Frame {
        header: frame.header.clone(),
        values: vec![],
    };
    ack.header.kind = Kind::Accepted;
    ack.header.tokens.clear();
    ack.header.route.clear();
    let sent = frame.header;
    match sent.route.first() {
        Some(next) => {
            let next = next.address;
            state
                .outbox
                .send(Outgoing { next, bytes, sent })
                .map_err(|_| anyhow::anyhow!("outbox closed"))?;
        }
        None => {
            state.results.send_replace(Some(Arc::new(Delivered {
                session: sent.session,
                position: sent.position,
                bytes,
            })));
        }
    }
    Ok(ack)
}

#[derive(Deserialize)]
struct Collect {
    session: String,
    position: usize,
}
/// Holds the request open until the last stage has the result for this session and position.
async fn result(State(state): State<Worker>, Json(request): Json<Collect>) -> Response {
    let mut results = state.results.subscribe();
    let wanted = |d: &Option<Arc<Delivered>>| {
        d.as_ref()
            .is_some_and(|d| d.session == request.session && d.position == request.position)
    };
    let collected = tokio::time::timeout(super::RESULT_WAIT, async {
        results
            .wait_for(wanted)
            .await
            .ok()
            .and_then(|d| d.as_ref().map(|d| d.bytes.clone()))
    })
    .await;
    match collected {
        Ok(Some(bytes)) => ([("content-type", "application/octet-stream")], bytes).into_response(),
        _ => error(
            StatusCode::GATEWAY_TIMEOUT,
            "no result from the last stage; a stage or hop on the route failed",
        ),
    }
}
