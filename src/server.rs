use crate::{kernel::Shard, planner::plan, protocol::*};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::de::DeserializeOwned;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    sync::{RwLock, Semaphore},
    task::JoinHandle,
};

type ApiResult<T> = std::result::Result<Json<T>, ApiError>;

pub struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({"error": self.1}))).into_response()
    }
}
fn bad(e: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, e.to_string())
}
fn unavailable(e: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::SERVICE_UNAVAILABLE, e.to_string())
}

pub fn validate_token(token: &str) -> Result<()> {
    ensure!(
        (16..=256).contains(&token.len()) && token.bytes().all(|b| b.is_ascii_graphic()),
        "set P2P_TOKEN to 16..=256 printable non-space ASCII characters"
    );
    Ok(())
}

pub(crate) async fn authenticate(
    State(token): State<String>,
    request: Request,
    next: Next,
) -> Response {
    let provided = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());
    if !provided
        .is_some_and(|value| crate::security::token_matches(&format!("Bearer {token}"), value))
    {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "invalid bearer token"})),
        )
            .into_response();
    }
    next.run(request).await
}

/// Sets TCP_NODELAY on accepted sockets. Without it, a response written in parts waits for the
/// peer's delayed ACK (about 40 ms on Linux) on every request; measured as ~80 ms per hop on a
/// 20-stage route.
pub fn nodelay(
    listener: TcpListener,
) -> axum::serve::TapIo<TcpListener, fn(&mut tokio::net::TcpStream)> {
    use axum::serve::ListenerExt;
    listener.tap_io(|tcp| {
        let _ = tcp.set_nodelay(true);
    })
}

pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(30))
        .build()?)
}

pub async fn decode<T: DeserializeOwned>(mut response: reqwest::Response) -> Result<T> {
    let status = response.status();
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            data.len() + chunk.len() <= 256 * 1024,
            "peer response exceeds size limit"
        );
        data.extend_from_slice(&chunk);
    }
    ensure!(
        status.is_success(),
        "peer returned {status}: {}",
        String::from_utf8_lossy(&data)
    );
    Ok(serde_json::from_slice(&data)?)
}

pub struct Service {
    pub address: SocketAddr,
    task: JoinHandle<()>,
    heartbeat: Option<JoinHandle<()>>,
}
impl Drop for Service {
    fn drop(&mut self) {
        if let Some(task) = &self.heartbeat {
            task.abort();
        }
        self.task.abort();
    }
}

struct Entry {
    info: WorkerInfo,
    seen: Instant,
    probe_ms: f64,
}
#[derive(Clone)]
struct Coordinator {
    entries: Arc<RwLock<HashMap<String, Entry>>>,
    client: reqwest::Client,
    token: String,
    slots: Arc<Semaphore>,
}

impl Coordinator {
    async fn peers(&self) -> Vec<Peer> {
        let mut entries = self.entries.write().await;
        entries.retain(|_, entry| entry.seen.elapsed() < Duration::from_secs(LEASE_SECONDS));
        let mut peers: Vec<_> = entries
            .values()
            .map(|entry| Peer {
                worker: entry.info.clone(),
                age_seconds: entry.seen.elapsed().as_secs_f64(),
                probe_ms: entry.probe_ms,
            })
            .collect();
        peers.sort_by(|a, b| a.worker.id.cmp(&b.worker.id));
        peers
    }
}

pub async fn coordinator(listener: TcpListener, token: String) -> Result<Service> {
    validate_token(&token)?;
    let address = listener.local_addr()?;
    let state = Coordinator {
        entries: Arc::default(),
        client: client()?,
        token: token.clone(),
        slots: Arc::new(Semaphore::new(8)),
    };
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(serde_json::json!({"status":"ok", "protocol":PROTOCOL_VERSION})) }),
        )
        .route("/v1/workers", get(peers).post(register))
        .route("/v1/plan", post(make_plan))
        .route("/v1/run", post(run))
        .layer(DefaultBodyLimit::max(64 * 1024))
        .layer(middleware::from_fn_with_state(token, authenticate))
        .with_state(state);
    let task = tokio::spawn(async move {
        if let Err(error) = axum::serve(nodelay(listener), app).await {
            tracing::error!(%error, "coordinator stopped");
        }
    });
    Ok(Service {
        address,
        task,
        heartbeat: None,
    })
}

async fn peers(State(state): State<Coordinator>) -> Json<Vec<Peer>> {
    Json(state.peers().await)
}

async fn register(
    State(state): State<Coordinator>,
    Json(claim): Json<WorkerInfo>,
) -> ApiResult<WorkerInfo> {
    claim.validate().map_err(bad)?;
    let started = Instant::now();
    let response = state
        .client
        .get(url(claim.address, "/v1/info"))
        .bearer_auth(&state.token)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map_err(unavailable)?;
    let info: WorkerInfo = decode(response).await.map_err(unavailable)?;
    info.validate().map_err(bad)?;
    if info.id != claim.id
        || info.address != claim.address
        || info.model != claim.model
        || info.start != claim.start
        || info.end != claim.end
    {
        return Err(bad("registration did not match the worker probe"));
    }
    let mut entries = state.entries.write().await;
    entries.retain(|_, e| e.seen.elapsed() < Duration::from_secs(LEASE_SECONDS));
    if let Some(existing) = entries.get(&info.id)
        && existing.info.address != info.address
    {
        return Err(bad("worker id already belongs to another live address"));
    }
    if entries.len() >= 256 && !entries.contains_key(&info.id) {
        return Err(unavailable("registry is full"));
    }
    entries.insert(
        info.id.clone(),
        Entry {
            info: info.clone(),
            seen: Instant::now(),
            probe_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
    );
    Ok(Json(info))
}

async fn make_plan(
    State(state): State<Coordinator>,
    Json(model): Json<ModelSpec>,
) -> ApiResult<Vec<WorkerInfo>> {
    model.validate().map_err(bad)?;
    Ok(Json(
        plan(&model, &state.peers().await).map_err(unavailable)?,
    ))
}

async fn run(
    State(state): State<Coordinator>,
    Json(request): Json<RunRequest>,
) -> ApiResult<RunResponse> {
    request.validate().map_err(bad)?;
    let _permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| unavailable("coordinator busy; retry later"))?;
    let route = plan(&request.model, &state.peers().await).map_err(unavailable)?;
    let first = route[0].address;
    let response = state
        .client
        .post(url(first, "/v1/execute"))
        .bearer_auth(&state.token)
        .json(&ExecuteRequest {
            model: request.model.clone(),
            activation: request.activation,
            route,
        })
        .send()
        .await
        .map_err(unavailable)?;
    let output: RunResponse = decode(response).await.map_err(unavailable)?;
    validate_output(&request.model, &output).map_err(unavailable)?;
    Ok(Json(output))
}

fn validate_output(model: &ModelSpec, output: &RunResponse) -> Result<()> {
    ensure!(output.model_id == model.id(), "response model mismatch");
    ensure!(
        output.activation.len() == model.width && output.activation.iter().all(|x| x.is_finite()),
        "invalid peer output"
    );
    ensure!(output.hops.len() <= MAX_HOPS, "too many response hops");
    Ok(())
}

#[derive(Clone)]
struct Worker {
    shard: Arc<Shard>,
    info: WorkerInfo,
    client: reqwest::Client,
    token: String,
    slots: Arc<Semaphore>,
}

pub struct WorkerConfig {
    pub id: String,
    pub advertise: Option<SocketAddr>,
    pub coordinator: SocketAddr,
    pub token: String,
    pub shard: Shard,
    pub delay_ms: u64,
}

pub async fn worker(listener: TcpListener, config: WorkerConfig) -> Result<Service> {
    validate_token(&config.token)?;
    validate_address(config.coordinator)?;
    let address = config.advertise.unwrap_or(listener.local_addr()?);
    let info = WorkerInfo {
        protocol: PROTOCOL_VERSION,
        id: config.id,
        address,
        model: config.shard.model.clone(),
        start: config.shard.start,
        end: config.shard.end,
        weight_bytes: config.shard.weight_bytes(),
        estimated_compute_ms: config.shard.calibrate_ms()?,
        simulated_delay_ms: config.delay_ms,
    };
    info.validate()?;
    let token = config.token;
    let state = Worker {
        shard: Arc::new(config.shard),
        info: info.clone(),
        client: client()?,
        token: token.clone(),
        slots: Arc::new(Semaphore::new(1)),
    };
    let app = Router::new()
        .route("/v1/info", get(worker_info))
        .route("/v1/execute", post(execute))
        .layer(DefaultBodyLimit::max(64 * 1024))
        .layer(middleware::from_fn_with_state(token.clone(), authenticate))
        .with_state(state);
    let task = tokio::spawn(async move {
        if let Err(error) = axum::serve(nodelay(listener), app).await {
            tracing::error!(%error, "worker stopped");
        }
    });
    let mut service = Service {
        address,
        task,
        heartbeat: None,
    };
    let http = client()?;
    let endpoint = url(config.coordinator, "/v1/workers");
    let response = http
        .post(&endpoint)
        .bearer_auth(&token)
        .json(&info)
        .send()
        .await
        .context("coordinator registration failed")?;
    let _: WorkerInfo = decode(response).await?;
    tracing::info!(worker = %info.id, %address, layers = %format!("{}..{}", info.start, info.end), bytes = info.weight_bytes, "worker registered");
    service.heartbeat = Some(tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            let result = async {
                let response = http
                    .post(&endpoint)
                    .bearer_auth(&token)
                    .json(&info)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await?;
                decode::<WorkerInfo>(response).await
            }
            .await;
            if let Err(error) = result {
                tracing::warn!(worker = %info.id, %error, "heartbeat failed; retrying");
            }
        }
    }));
    Ok(service)
}

async fn worker_info(State(state): State<Worker>) -> Json<WorkerInfo> {
    Json(state.info)
}

async fn execute(
    State(state): State<Worker>,
    Json(request): Json<ExecuteRequest>,
) -> ApiResult<RunResponse> {
    RunRequest {
        model: request.model.clone(),
        activation: request.activation.clone(),
    }
    .validate()
    .map_err(bad)?;
    if request.route.is_empty() || request.route.len() > MAX_HOPS {
        return Err(bad("invalid route length"));
    }
    let first = &request.route[0];
    if first.id != state.info.id
        || first.address != state.info.address
        || first.start != state.info.start
        || first.end != state.info.end
        || request.model != state.info.model
    {
        return Err(bad("route does not match this worker's resident shard"));
    }
    let mut next_layer = state.info.start;
    for hop in &request.route {
        hop.validate().map_err(bad)?;
        if hop.model != request.model || hop.start != next_layer {
            return Err(bad("route has a gap, overlap, or model mismatch"));
        }
        next_layer = hop.end;
    }
    if next_layer != request.model.layers {
        return Err(bad("route must reach the final layer"));
    }
    let permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| unavailable("worker busy; retry later"))?;
    if state.info.simulated_delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(state.info.simulated_delay_ms)).await;
    }
    let shard = state.shard.clone();
    let (activation, compute_ms) = tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let output = shard.forward(request.activation)?;
        Ok::<_, anyhow::Error>((output, started.elapsed().as_secs_f64() * 1000.0))
    })
    .await
    .map_err(unavailable)?
    .map_err(unavailable)?;
    drop(permit);
    let hop = Hop {
        worker_id: state.info.id.clone(),
        start: state.info.start,
        end: state.info.end,
        compute_ms,
        simulated_delay_ms: state.info.simulated_delay_ms,
    };
    let tail: Vec<_> = request.route.into_iter().skip(1).collect();
    if let Some(next) = tail.first() {
        let response = state
            .client
            .post(url(next.address, "/v1/execute"))
            .bearer_auth(&state.token)
            .json(&ExecuteRequest {
                model: request.model.clone(),
                activation,
                route: tail,
            })
            .send()
            .await
            .map_err(unavailable)?;
        let mut output: RunResponse = decode(response).await.map_err(unavailable)?;
        validate_output(&request.model, &output).map_err(unavailable)?;
        output.hops.insert(0, hop);
        Ok(Json(output))
    } else {
        Ok(Json(RunResponse {
            model_id: request.model.id(),
            activation,
            hops: vec![hop],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn expired_leases_are_removed_before_planning() {
        let model = ModelSpec {
            layers: 1,
            width: 8,
        };
        let info = WorkerInfo {
            protocol: PROTOCOL_VERSION,
            id: "expired".into(),
            address: "127.0.0.1:1234".parse().unwrap(),
            model,
            start: 0,
            end: 1,
            weight_bytes: 256,
            estimated_compute_ms: 1.0,
            simulated_delay_ms: 0,
        };
        let state = Coordinator {
            entries: Arc::default(),
            client: client().unwrap(),
            token: "test-token-long-enough".into(),
            slots: Arc::new(Semaphore::new(1)),
        };
        state.entries.write().await.insert(
            info.id.clone(),
            Entry {
                info,
                seen: Instant::now() - Duration::from_secs(LEASE_SECONDS + 1),
                probe_ms: 1.0,
            },
        );
        assert!(state.peers().await.is_empty());
        assert!(state.entries.read().await.is_empty());
    }
}
