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
    collections::HashMap,
    net::SocketAddr,
    path::Path,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
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
    /// True when every slot holds a live session.
    #[serde(default)]
    pub busy: bool,
    /// Sessions the worker serves at once.
    #[serde(default = "one_slot")]
    pub slots: usize,
    #[serde(default)]
    pub memory: Option<crate::resources::Memory>,
    /// SHA-256 of the GGUF a llama.cpp worker loaded. Quantizing is not reproducible across
    /// machines, so a route must not mix different GGUF files of the same precision.
    #[serde(default)]
    pub weights_sha256: Option<String>,
    /// SHA-256 of the MTP head a final stage drafts with, if it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp_sha256: Option<String>,
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

fn one_slot() -> usize {
    1
}

/// A session idle this long loses its slot.
const SESSION_IDLE: Duration = Duration::from_secs(60);

/// Positions one device call can hold: the engine's batch size.
const MAX_BATCH_POSITIONS: usize = 512;

/// Most tokens a client may ask the final stage to draft per step.
const MAX_DRAFTS: usize = 8;

/// Drafts the MTP head is less sure of than this are not proposed; `SANGAMA_MTP_P_MIN`.
fn mtp_p_min() -> f32 {
    static VALUE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("SANGAMA_MTP_P_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f32| (0.0..1.0).contains(v))
            .unwrap_or(0.0)
    })
}

/// Most frames one batch takes: the live sessions shared evenly between the route's stages,
/// so each stage has a batch in hand. `SANGAMA_BATCH_MAX` overrides it for measurements.
fn batch_limit(sessions: usize, stages: usize) -> usize {
    static FIXED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let fixed = *FIXED.get_or_init(|| {
        std::env::var("SANGAMA_BATCH_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v >= 1)
    });
    fixed.unwrap_or_else(|| sessions.max(1).div_ceil(stages.max(1)))
}

/// Frames of different sessions share a device call unless `SANGAMA_BATCH=0`, which runs them
/// one at a time, e.g. to compare the two.
fn batching() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("SANGAMA_BATCH").map_or(true, |v| v != "0"))
}

fn argmax(values: &[f32]) -> usize {
    (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

struct Session {
    /// The engine sequence that holds this session's cache.
    slot: usize,
    position: usize,
    used: Instant,
    /// The state before the latest speculative batch, and that batch's inputs.
    draft: Option<Draft>,
}
struct Draft {
    position: usize,
    seq_len: usize,
    /// The state before the batch and its inputs, kept only when the engine cannot rewind
    /// this many positions on its own.
    saved: Option<Saved>,
}
struct Saved {
    state: Vec<u8>,
    tokens: Vec<u32>,
    values: Vec<f32>,
}
struct Resident {
    model: Engine,
    sessions: HashMap<String, Session>,
    load: Load,
}
/// How much of the recent time the engine spent computing.
struct Load {
    since: Instant,
    busy: Duration,
    level: f64,
}
impl Load {
    fn new() -> Self {
        Load {
            since: Instant::now(),
            busy: Duration::ZERO,
            level: 0.0,
        }
    }
    /// Counts `busy` computing time and, every fifth of a second, averages the busy share of
    /// the time since into the level.
    fn add(&mut self, busy: Duration) {
        self.busy += busy;
        let window = self.since.elapsed();
        if window >= Duration::from_millis(200) {
            let share = (self.busy.as_secs_f64() / window.as_secs_f64()).min(1.0);
            self.level = 0.5 * self.level + 0.5 * share;
            self.since = Instant::now();
            self.busy = Duration::ZERO;
        }
    }
}
/// Drafts to propose when the client asked for `wanted`. Checking a draft costs every stage a
/// position. That pays while a pass is mostly network and fixed cost, and loses once the
/// device is busy, where a position spent on another request's token is never wasted.
/// `SANGAMA_MTP_LOAD=0` always drafts what the client asked for.
fn drafts_under_load(wanted: usize, level: f64) -> usize {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("SANGAMA_MTP_LOAD").map_or(true, |v| v != "0")) {
        return wanted;
    }
    if level > 0.85 {
        0
    } else if level > 0.65 {
        wanted.min(2)
    } else {
        wanted
    }
}
impl Resident {
    /// Frees the slots of sessions idle past their expiry; returns their ids.
    fn expire(&mut self) -> Vec<String> {
        let stale: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.used.elapsed() > SESSION_IDLE)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &stale {
            self.end(id);
        }
        stale
    }
    /// Starts a session at position zero in a free slot, clearing whatever it held.
    fn start(&mut self, id: &str) -> Result<&mut Session> {
        let slot = (0..self.model.slots())
            .find(|slot| self.sessions.values().all(|s| s.slot != *slot))
            .context("all of this worker's slots are in use")?;
        self.model.clear(slot);
        Ok(self.sessions.entry(id.to_string()).or_insert(Session {
            slot,
            position: 0,
            used: Instant::now(),
            draft: None,
        }))
    }
    fn end(&mut self, id: &str) -> bool {
        match self.sessions.remove(id) {
            Some(session) => {
                self.model.clear(session.slot);
                true
            }
            None => false,
        }
    }
}
/// A frame waiting for the engine, and where its result goes.
struct Job {
    frame: Frame,
    received_ms: f64,
    reply: tokio::sync::oneshot::Sender<Result<Frame>>,
}
/// A detached frame waiting to be passed to the next stage.
struct Outgoing {
    next: SocketAddr,
    bytes: Vec<u8>,
    sent: Header,
}
/// A detached result kept by the last stage until the client collects it.
struct Delivered {
    position: usize,
    bytes: Vec<u8>,
}
#[derive(Clone)]
struct Worker {
    /// The engine and its sessions. Frames wait here for their turn on the device.
    resident: Arc<Mutex<Resident>>,
    /// Frames that arrived while the engine was busy; they run together when it is free.
    queue: Arc<Mutex<Vec<Job>>>,
    /// Live sessions, readable without waiting for the engine.
    active: Arc<AtomicUsize>,
    /// The last stage's latest result for each session, until its client collects it.
    results: Arc<Mutex<HashMap<String, Arc<Delivered>>>>,
    /// Bumped whenever a result arrives, to wake waiting collectors.
    delivered: Arc<tokio::sync::watch::Sender<u64>>,
    /// One queue per session: a session's detached frames leave in the order this stage
    /// computed them, so a pipelined prompt's chunks cannot overtake each other, while
    /// different sessions send in parallel.
    outboxes: Arc<Mutex<HashMap<String, tokio::sync::mpsc::UnboundedSender<Outgoing>>>>,
    info: Info,
    manifest: Manifest,
    token: String,
    http: reqwest::Client,
    allow_next: Vec<SocketAddr>,
}

/// A loaded engine with its precision, the SHA-256 of its GGUF and MTP head, and its memory.
type Opened = (
    Engine,
    String,
    Option<String>,
    Option<String>,
    crate::resources::Memory,
);

/// Which engine a worker runs, and for llama.cpp the approved GGUF file in the model directory.
pub struct EngineChoice<'a> {
    pub name: &'a str,
    pub gguf: Option<&'a str>,
    /// Cap on the memory this worker may use, e.g. to act like an average laptop.
    pub memory_budget_mib: Option<u64>,
    /// Sessions served at once, each with its own cache. Only llama.cpp supports more than one.
    pub slots: usize,
    /// MTP-only GGUF in the model directory; the final stage then drafts tokens (llama.cpp).
    pub mtp_gguf: Option<&'a str>,
    /// Drafted positions a session can be rewound without saving state (llama.cpp, models
    /// with recurrent layers). Each costs one more recurrent state per slot.
    pub rollback: usize,
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
    let (model, precision, weights_sha256, mtp_sha256, memory) = match engine.name {
        "candle" => {
            ensure!(engine.gguf.is_none(), "--gguf requires --engine llamacpp");
            ensure!(
                engine.slots == 1,
                "--slots above 1 requires --engine llamacpp"
            );
            ensure!(
                engine.mtp_gguf.is_none(),
                "--mtp-gguf requires --engine llamacpp"
            );
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
                None,
                memory,
            )
        }
        "llamacpp" => open_llamacpp(dir, &manifest, &spec, &hash, backend, &engine)?,
        other => anyhow::bail!("unknown engine {other}"),
    };
    let slots = model.slots();
    let state = Worker {
        outboxes: Arc::default(),
        resident: Arc::new(Mutex::new(Resident {
            model,
            sessions: HashMap::new(),
            load: Load::new(),
        })),
        queue: Arc::default(),
        active: Arc::default(),
        results: Arc::default(),
        delivered: Arc::new(tokio::sync::watch::channel(0).0),
        info: Info {
            model_id: manifest.model_id.clone(),
            model_hash: hash,
            shard: spec,
            device: backend.into(),
            engine: engine.name.into(),
            weights_sha256,
            mtp_sha256,
            precision,
            pid: std::process::id(),
            busy: false,
            slots,
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
) -> Result<Opened> {
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
    let slots = engine.slots as u64;
    let mtp = match engine.mtp_gguf {
        Some(name) => {
            ensure!(
                Path::new(name).components().count() == 1 && name.ends_with(".gguf"),
                "--mtp-gguf must be a .gguf file name inside the model directory"
            );
            ensure!(
                spec.index + 1 == manifest.shards.len(),
                "only the final stage drafts with an MTP head"
            );
            let path = dir.join(name);
            let bytes = std::fs::metadata(&path)
                .with_context(|| format!("MTP head {} not found", path.display()))?
                .len();
            Some((path, bytes))
        }
        None => None,
    };
    let required = gguf.file_bytes
        + mtp.as_ref().map_or(0, |(_, bytes)| *bytes)
        + slots * layers * 2 * CONTEXT_LIMIT as u64 * 2 * 64 * 4
        + 384 * 1024 * 1024;
    let memory = crate::resources::check_required(required, available, engine.memory_budget_mib)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let options = Options {
        gpu,
        context: CONTEXT_LIMIT,
        slots: engine.slots,
        rollback: engine.rollback,
        threads,
    };
    let mut stage = Stage::open(&dir.join(&gguf.file), spec.start, spec.end, &options)?;
    let mtp_sha256 = match &mtp {
        Some((path, _)) => {
            stage.attach_mtp(path, &options)?;
            Some(super::sha256(path)?)
        }
        None => None,
    };
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
        mtp_sha256,
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
) -> Result<Opened> {
    anyhow::bail!("rebuild with --features llamacpp (or llamacpp-metal, -cuda, -vulkan, -hip)")
}

async fn info(State(state): State<Worker>) -> Json<Info> {
    let mut info = state.info;
    info.busy = state.active.load(Ordering::Relaxed) >= info.slots;
    Json(info)
}

impl Worker {
    /// Waits for the engine. Frames of different sessions take turns on the device.
    fn engine(&self) -> MutexGuard<'_, Resident> {
        self.resident.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Drops what this stage kept for ended sessions: queued frames and uncollected results.
    fn forget(&self, ended: &[String]) {
        if ended.is_empty() {
            return;
        }
        let mut outboxes = self.outboxes.lock().unwrap_or_else(|e| e.into_inner());
        let mut results = self.results.lock().unwrap_or_else(|e| e.into_inner());
        for id in ended {
            outboxes.remove(id);
            results.remove(id);
        }
    }
    /// Runs `f` on the engine off the async runtime, then clears up sessions that ended.
    async fn with_engine<T: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Resident) -> (T, Vec<String>) + Send + 'static,
    ) -> Result<T> {
        let worker = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut resident = worker.engine();
            let mut ended = resident.expire();
            let (value, more) = f(&mut resident);
            ended.extend(more);
            worker
                .active
                .store(resident.sessions.len(), Ordering::Relaxed);
            drop(resident);
            worker.forget(&ended);
            value
        })
        .await
        .context("engine task failed")
    }
}

#[derive(Deserialize)]
struct Reset {
    session: String,
}
async fn reserve(State(state): State<Worker>, Json(request): Json<Reset>) -> Response {
    if uuid::Uuid::parse_str(&request.session).is_err() {
        return error(StatusCode::BAD_REQUEST, "invalid session");
    }
    let reserved = state
        .with_engine(move |resident| {
            let reserved = match resident.sessions.get_mut(&request.session) {
                Some(session) => {
                    session.used = Instant::now();
                    true
                }
                None => resident.start(&request.session).is_ok(),
            };
            (reserved, vec![])
        })
        .await;
    match reserved {
        Ok(true) => Json(serde_json::json!({"reserved":true,"lease_seconds":60})).into_response(),
        Ok(false) => error(
            StatusCode::CONFLICT,
            "all of this worker's slots are in use",
        ),
        Err(err) => error(StatusCode::SERVICE_UNAVAILABLE, err),
    }
}
async fn reset(State(state): State<Worker>, Json(request): Json<Reset>) -> Response {
    let reset = state
        .with_engine(move |resident| {
            resident.end(&request.session);
            ((), vec![request.session])
        })
        .await;
    match reset {
        Ok(()) => Json(serde_json::json!({"reset":true})).into_response(),
        Err(err) => error(StatusCode::SERVICE_UNAVAILABLE, err),
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
    ensure!(
        h.drafts.is_empty() && h.mtp_drafts <= MAX_DRAFTS,
        "invalid draft request"
    );
    if !h.inputs.is_empty() {
        ensure!(
            h.inputs.len() == h.seq_len
                && h.inputs
                    .iter()
                    .all(|t| (*t as usize) < state.manifest.vocab_size())
                && (index > 0 || h.inputs == h.tokens),
            "invalid input tokens"
        );
    }
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

fn unix_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

async fn forward(State(state): State<Worker>, bytes: Bytes) -> Response {
    let received_ms = unix_ms();
    let frame = match Frame::decode(&bytes).and_then(|f| {
        validate(&f, &state)?;
        Ok(f)
    }) {
        Ok(frame) => frame,
        Err(err) => return error(StatusCode::BAD_REQUEST, err),
    };
    let session_id = frame.header.session.clone();
    let (position, seq_len) = (frame.header.position, frame.header.seq_len);
    // Frames that arrive while the engine is busy wait in the queue and run together.
    let (reply, computed) = tokio::sync::oneshot::channel();
    state
        .queue
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Job {
            frame,
            received_ms,
            reply,
        });
    let worker = state.clone();
    tokio::task::spawn_blocking(move || worker.drain());
    let calculation = computed.await.context("engine task failed");
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
            discard(&state, &session_id).await;
            error(StatusCode::SERVICE_UNAVAILABLE, err)
        }
    }
}

/// A partial chain cannot safely retry a position: discard this local session.
async fn discard(state: &Worker, session: &str) {
    let session = session.to_string();
    let _ = state
        .with_engine(move |resident| {
            resident.end(&session);
            ((), vec![session])
        })
        .await;
}

impl Worker {
    /// Runs every queued frame, one per session at a time, until the queue is empty. Whichever
    /// caller gets the engine first serves the frames of all that are waiting.
    fn drain(&self) {
        let mut resident = self.engine();
        let mut ended = resident.expire();
        loop {
            // Take one frame per session, up to a share of the live sessions. If every waiting
            // frame ran as one batch, the sessions would move through the route as a single
            // wave and only one stage would work at a time. Smaller batches, run back to
            // back, keep every stage busy.
            let limit = batch_limit(resident.sessions.len(), self.manifest.shards.len());
            let jobs = {
                let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                let mut seen = std::collections::HashSet::new();
                let (mut jobs, mut rest) = (Vec::new(), Vec::new());
                // A prompt is many positions and would hold up every token that shares its
                // turn, so a turn takes one prompt frame at most.
                let mut prompt = false;
                for job in queue.drain(..) {
                    let long = job.frame.header.seq_len > 1 + MAX_DRAFTS;
                    if jobs.len() < limit
                        && !(long && prompt)
                        && !seen.contains(&job.frame.header.session)
                    {
                        seen.insert(job.frame.header.session.clone());
                        prompt |= long;
                        jobs.push(job);
                    } else {
                        // A session's later frames wait for its earlier ones.
                        seen.insert(job.frame.header.session.clone());
                        rest.push(job);
                    }
                }
                *queue = rest;
                jobs
            };
            if jobs.is_empty() {
                break;
            }
            let sessions: Vec<String> = jobs
                .iter()
                .map(|j| j.frame.header.session.clone())
                .collect();
            let (frames, replies): (Vec<_>, Vec<_>) = jobs
                .into_iter()
                .map(|j| ((j.frame, j.received_ms), j.reply))
                .unzip();
            let started = Instant::now();
            let results = compute_batch(self, &mut resident, frames);
            resident.load.add(started.elapsed());
            for ((result, reply), session) in results.into_iter().zip(replies).zip(sessions) {
                if result.is_err() {
                    // A failed step leaves this stage's cache in an unknown state.
                    resident.end(&session);
                    ended.push(session);
                }
                let _ = reply.send(result);
            }
        }
        self.active
            .store(resident.sessions.len(), Ordering::Relaxed);
        drop(resident);
        self.forget(&ended);
    }
}

/// What a stage computed for one frame.
enum Output {
    /// Hidden states of every position, or the final stage's logits of the last position.
    Values(Vec<f32>),
    /// The final stage's greedy token after every position.
    Tokens(Vec<u32>),
}

/// Finds or starts the frame's session, rewinds rejected drafts, and notes how to undo this
/// frame if it carries drafts. Returns the session's slot and that note.
fn prepare(resident: &mut Resident, frame: &Frame) -> Result<(usize, Option<Draft>)> {
    let h = &frame.header;
    let session = match resident.sessions.get_mut(&h.session) {
        Some(session) => session,
        None => {
            ensure!(h.position == 0, "new session must begin at position zero");
            resident.start(&h.session)?
        }
    };
    let slot = session.slot;
    let previous = session.draft.take();
    if session.position != h.position {
        // The client accepted only part of the last speculative batch. Rewind to the end of
        // the accepted inputs; without engine support, restore the state from before the
        // batch and replay them.
        let d = previous
            .filter(|d| d.position < h.position && h.position < d.position + d.seq_len)
            .context("session mismatch or out-of-order position; reset required")?;
        match d.saved {
            None => resident.model.rollback(slot, h.position)?,
            Some(saved) => {
                let keep = h.position - d.position;
                let width = saved.values.len() / d.seq_len;
                resident.model.load_state(slot, &saved.state)?;
                resident.model.forward(
                    slot,
                    &saved.tokens[..saved.tokens.len().min(keep)],
                    &saved.values[..(keep * width).min(saved.values.len())],
                    keep,
                    d.position,
                )?;
            }
        }
    }
    let draft = if h.speculative {
        // Every position after the first may be rejected.
        let saved = if resident.model.rollback_depth() >= h.seq_len - 1 {
            None
        } else {
            Some(Saved {
                state: resident.model.save_state(slot)?,
                tokens: h.tokens.clone(),
                values: frame.values.clone(),
            })
        };
        Some(Draft {
            position: h.position,
            seq_len: h.seq_len,
            saved,
        })
    } else {
        None
    };
    Ok((slot, draft))
}

/// Runs one frame of each session through this stage's layers. Frames of different sessions
/// share one device call when the engine can batch; each result is in the order given.
fn compute_batch(
    worker: &Worker,
    resident: &mut Resident,
    frames: Vec<(Frame, f64)>,
) -> Vec<Result<Frame>> {
    let last = worker.info.shard.index + 1 == worker.manifest.shards.len();
    let mut results: Vec<Option<Result<Frame>>> = frames.iter().map(|_| None).collect();
    // Index into `frames`, slot, undo note.
    let mut ready: Vec<(usize, usize, Option<Draft>)> = Vec::new();
    for (index, (frame, _)) in frames.iter().enumerate() {
        match prepare(resident, frame) {
            Ok((slot, draft)) => ready.push((index, slot, draft)),
            Err(error) => results[index] = Some(Err(error)),
        }
    }
    let started = Instant::now();
    let started_ms = unix_ms();
    // The final stage batches only frames that want a sampled token: a batch returns tokens.
    let together: Vec<usize> = if resident.model.batches() && batching() {
        ready
            .iter()
            .enumerate()
            .filter(|(_, (index, _, _))| !last || frames[*index].0.header.sample)
            .map(|(k, _)| k)
            .collect()
    } else {
        vec![]
    };
    // Output of each ready frame, and the drafts the final stage proposes after it.
    let mut outputs: Vec<Option<Result<Output>>> = ready.iter().map(|_| None).collect();
    let mut drafts: Vec<Vec<u32>> = ready.iter().map(|_| vec![]).collect();
    // The engine merges sequences into one step only when their slots are consecutive and
    // increasing, so frames go in slot order.
    let mut together = together;
    together.sort_by_key(|&k| ready[k].1);
    // A device call holds at most MAX_BATCH_POSITIONS positions, so frames go in groups.
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut size = 0;
    // Frames that carry drafts also stay in runs of adjacent slots. With one shared attention
    // buffer the engine would merge any slots into a step, but its rollback states then come
    // out wrong: drafted sessions in slots with gaps between them lost their own context.
    let mut previous: Option<(usize, bool)> = None;
    for &k in &together {
        let h = &frames[ready[k].0].0.header;
        let (n, slot) = (h.seq_len, ready[k].1);
        let gap = previous
            .is_some_and(|(before, drafted)| (drafted || h.speculative) && slot != before + 1);
        previous = Some((slot, h.speculative));
        if groups.is_empty() || gap || size + n > MAX_BATCH_POSITIONS {
            groups.push(vec![]);
            size = 0;
        }
        groups.last_mut().expect("a group was just added").push(k);
        size += n;
    }
    let width = worker.manifest.hidden_size();
    for group in groups.into_iter().filter(|g| g.len() > 1) {
        let items: Vec<(usize, usize, usize)> = group
            .iter()
            .map(|&k| {
                let h = &frames[ready[k].0].0.header;
                (ready[k].1, h.seq_len, h.position)
            })
            .collect();
        let tokens: Vec<u32> = group
            .iter()
            .flat_map(|&k| frames[ready[k].0].0.header.tokens.iter().copied())
            .collect();
        let values: Vec<f32> = group
            .iter()
            .flat_map(|&k| frames[ready[k].0].0.values.iter().copied())
            .collect();
        match resident.model.forward_many(&items, &tokens, &values) {
            Ok(many) => {
                let mut row = 0;
                let mut computed: Vec<(usize, usize, Output)> = vec![];
                for (&k, item) in group.iter().zip(&items) {
                    let n = item.1;
                    let h = &frames[ready[k].0].0.header;
                    let output = match &many {
                        super::engine::Many::Hidden(all) => {
                            Output::Values(all[row * width..(row + n) * width].to_vec())
                        }
                        super::engine::Many::Tokens(ids) => {
                            let ids = &ids[row..row + n];
                            // A frame without drafts wants only the token after its last position.
                            Output::Tokens(if h.speculative {
                                ids.to_vec()
                            } else {
                                vec![ids[n - 1]]
                            })
                        }
                    };
                    computed.push((k, row, output));
                    row += n;
                }
                // The MTP head reads this call's hidden states, so it runs before the next.
                let headers: Vec<(&Header, usize, usize, &Output)> = computed
                    .iter()
                    .map(|(k, row, output)| {
                        (&frames[ready[*k].0].0.header, ready[*k].1, *row, output)
                    })
                    .collect();
                match draft_after(resident, last, &headers) {
                    Ok(mut proposed) => {
                        for (n, (k, _, output)) in computed.into_iter().enumerate() {
                            drafts[k] = std::mem::take(&mut proposed[n]);
                            outputs[k] = Some(Ok(output));
                        }
                    }
                    Err(error) => {
                        let message = error.to_string();
                        for &k in &group {
                            outputs[k] = Some(Err(anyhow::anyhow!("{message}")));
                        }
                    }
                }
            }
            Err(error) => {
                let message = error.to_string();
                for &k in &group {
                    outputs[k] = Some(Err(anyhow::anyhow!("{message}")));
                }
            }
        }
    }
    for (k, (index, slot, _)) in ready.iter().enumerate() {
        if outputs[k].is_some() {
            continue;
        }
        let (frame, _) = &frames[*index];
        let h = &frame.header;
        let computed = if last && h.sample && h.speculative {
            resident
                .model
                .greedy(*slot, &h.tokens, &frame.values, h.seq_len, h.position)
                .map(Output::Tokens)
        } else {
            resident
                .model
                .forward(*slot, &h.tokens, &frame.values, h.seq_len, h.position)
                .map(Output::Values)
        };
        outputs[k] = Some(computed.and_then(|output| {
            drafts[k] = draft_after(resident, last, &[(h, *slot, 0, &output)])?.remove(0);
            Ok(output)
        }));
    }
    if let Some(ms) = simulated_ms_per_layer_token() {
        let layers = (worker.info.shard.end - worker.info.shard.start) as f64;
        let positions: usize = ready
            .iter()
            .map(|(i, _, _)| frames[*i].0.header.seq_len)
            .sum();
        std::thread::sleep(Duration::from_secs_f64(
            ms * layers * positions as f64 / 1000.0,
        ));
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    let batch_frames = ready.len();
    let batch_positions: usize = ready
        .iter()
        .map(|(i, _, _)| frames[*i].0.header.seq_len)
        .sum();
    let mut frames: Vec<Option<(Frame, f64)>> = frames.into_iter().map(Some).collect();
    for (k, (index, _, draft)) in ready.into_iter().enumerate() {
        let (frame, received_ms) = frames[index].take().expect("each frame is finished once");
        let output = outputs[k].take().expect("every ready frame was computed");
        results[index] = Some(output.and_then(|output| {
            finish(
                worker,
                resident,
                frame,
                output,
                std::mem::take(&mut drafts[k]),
                draft,
                Trace {
                    shard: worker.info.shard.index,
                    start: worker.info.shard.start,
                    end: worker.info.shard.end,
                    forward_ms: elapsed,
                    started_ms,
                    received_ms,
                    batch_frames,
                    batch_positions,
                },
            )
        }));
    }
    results
        .into_iter()
        .map(|r| r.expect("every frame has a result"))
        .collect()
}

/// The final stage's MTP head sees every kept position of each frame, then drafts after the
/// token the model chose: the drafts it verified up to the first rejection, then its own
/// token. It reads the hidden states of the device call the frames shared; each entry is a
/// frame's header, slot, first row in that call and output, in increasing slot order.
fn draft_after(
    resident: &mut Resident,
    last: bool,
    frames: &[(&Header, usize, usize, &Output)],
) -> Result<Vec<Vec<u32>>> {
    let mut proposed: Vec<Vec<u32>> = frames.iter().map(|_| vec![]).collect();
    if !last || !resident.model.has_mtp() {
        return Ok(proposed);
    }
    let level = resident.load.level;
    let (mut steps, mut owners) = (vec![], vec![]);
    for (n, (h, slot, row, output)) in frames.iter().enumerate() {
        if h.inputs.is_empty() {
            continue;
        }
        let (keep, next) = match output {
            Output::Tokens(ids) if h.speculative => {
                let accepted = h.inputs[1..]
                    .iter()
                    .zip(ids)
                    .take_while(|(draft, model)| draft == model)
                    .count();
                (accepted + 1, ids[accepted])
            }
            Output::Tokens(ids) => (h.seq_len, ids[0]),
            Output::Values(values) => (h.seq_len, argmax(values) as u32),
        };
        let wanted = if h.sample {
            drafts_under_load(h.mtp_drafts, level)
        } else {
            0
        };
        let room = CONTEXT_LIMIT.saturating_sub(h.position + keep);
        steps.push(super::engine::MtpStep {
            slot: *slot,
            row: *row,
            inputs: &h.inputs[..keep],
            position: h.position,
            next,
            n_draft: wanted.min(room),
        });
        owners.push(n);
    }
    if steps.is_empty() {
        return Ok(proposed);
    }
    let drafted = resident.model.mtp_step_many(&steps, mtp_p_min())?;
    for (n, drafts) in owners.into_iter().zip(drafted) {
        proposed[n] = drafts;
    }
    Ok(proposed)
}

/// Records the session's progress and turns a computed frame into the one to pass on.
fn finish(
    worker: &Worker,
    resident: &mut Resident,
    frame: Frame,
    output: Output,
    drafts: Vec<u32>,
    draft: Option<Draft>,
    trace: Trace,
) -> Result<Frame> {
    let h = &frame.header;
    let session = resident
        .sessions
        .get_mut(&h.session)
        .context("session ended during its forward")?;
    session.position = h.position + h.seq_len;
    session.used = Instant::now();
    session.draft = draft;
    let mut header = frame.header;
    header.tokens.clear();
    header.route.remove(0);
    header.trace.push(trace);
    header.kind = if header.route.is_empty() {
        Kind::Logits
    } else {
        Kind::Hidden
    };
    if header.route.is_empty() {
        header.inputs.clear();
        header.drafts = drafts;
    }
    let mut values = vec![];
    match output {
        Output::Tokens(ids) => {
            header.tokens = ids;
            header.kind = Kind::Sampled;
        }
        Output::Values(computed) if header.route.is_empty() && header.sample => {
            ensure!(
                computed.len() == worker.manifest.vocab_size()
                    && computed.iter().all(|v| v.is_finite()),
                "invalid final logits"
            );
            header.tokens = vec![argmax(&computed) as u32];
            header.kind = Kind::Sampled;
        }
        Output::Values(computed) => values = computed,
    }
    Ok(Frame { header, values })
}

/// Starts a session's outbox: its frames go to the next stage one at a time, in order.
fn send_in_order(state: Worker) -> tokio::sync::mpsc::UnboundedSender<Outgoing> {
    let (outbox, mut outgoing) = tokio::sync::mpsc::unbounded_channel::<Outgoing>();
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
                discard(&state, &sent.session).await;
            }
        }
    });
    outbox
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
            let mut outboxes = state.outboxes.lock().unwrap_or_else(|e| e.into_inner());
            let outbox = outboxes
                .entry(sent.session.clone())
                .or_insert_with(|| send_in_order(state.clone()));
            outbox
                .send(Outgoing { next, bytes, sent })
                .map_err(|_| anyhow::anyhow!("outbox closed"))?;
        }
        None => {
            state
                .results
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    sent.session,
                    Arc::new(Delivered {
                        position: sent.position,
                        bytes,
                    }),
                );
            state.delivered.send_modify(|n| *n += 1);
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
    let mut delivered = state.delivered.subscribe();
    let collected = tokio::time::timeout(super::RESULT_WAIT, async {
        loop {
            delivered.borrow_and_update();
            let found = state
                .results
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&request.session)
                .filter(|d| d.position == request.position)
                .map(|d| d.bytes.clone());
            if found.is_some() {
                return found;
            }
            delivered.changed().await.ok()?;
        }
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
