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
        })),
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
    let calculation = state
        .with_engine(move |resident| {
            let session = frame.header.session.clone();
            match compute(&worker, resident, frame) {
                Ok(frame) => (Ok(frame), vec![]),
                Err(error) => {
                    // A failed step leaves this stage's cache in an unknown state.
                    resident.end(&session);
                    (Err(error), vec![session])
                }
            }
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

/// Runs one frame of a session through this stage's layers, starting the session if new.
fn compute(worker: &Worker, resident: &mut Resident, frame: Frame) -> Result<Frame> {
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
    let started = Instant::now();
    let started_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0);
    let last = h.route.len() == 1;
    let (drafted, mut values) = if last && h.sample && h.speculative {
        let ids = resident
            .model
            .greedy(slot, &h.tokens, &frame.values, h.seq_len, h.position)?;
        (Some(ids), vec![])
    } else {
        let values =
            resident
                .model
                .forward(slot, &h.tokens, &frame.values, h.seq_len, h.position)?;
        (None, values)
    };
    // The final stage's MTP head sees every kept position, then drafts after the token the
    // model chose: the drafts it verified up to the first rejection, then its own token.
    let drafts = if last && !h.inputs.is_empty() && resident.model.has_mtp() {
        let (keep, next) = match &drafted {
            Some(ids) => {
                let accepted = h.inputs[1..]
                    .iter()
                    .zip(ids)
                    .take_while(|(draft, model)| draft == model)
                    .count();
                (accepted + 1, ids[accepted])
            }
            None => (h.seq_len, argmax(&values) as u32),
        };
        let wanted = if h.sample { h.mtp_drafts } else { 0 };
        let room = CONTEXT_LIMIT.saturating_sub(h.position + keep);
        resident.model.mtp_step(
            slot,
            &h.inputs[..keep],
            h.position,
            next,
            wanted.min(room),
            mtp_p_min(),
        )?
    } else {
        vec![]
    };
    if let Some(ms) = simulated_ms_per_layer_token() {
        let layers = (worker.info.shard.end - worker.info.shard.start) as f64;
        std::thread::sleep(Duration::from_secs_f64(
            ms * layers * h.seq_len as f64 / 1000.0,
        ));
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
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
    if header.route.is_empty() {
        header.inputs.clear();
        header.drafts = drafts;
    }
    if let Some(ids) = drafted {
        header.tokens = ids;
        header.kind = Kind::Sampled;
    } else if header.route.is_empty() && header.sample {
        ensure!(
            values.len() == worker.manifest.vocab_size() && values.iter().all(|v| v.is_finite()),
            "invalid final logits"
        );
        header.tokens = vec![argmax(&values) as u32];
        header.kind = Kind::Sampled;
        values.clear();
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
