//! Local control panel. Browser access requires an ephemeral capability, never the peer token.
use crate::qwen::runner::{self, Options};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::{net::TcpListener, sync::Mutex};

#[derive(Clone)]
struct App {
    key: String,
    origin: String,
    host: String,
    model_dir: PathBuf,
    device: String,
    peers: Vec<SocketAddr>,
    peer_token: Option<String>,
    job: Arc<Mutex<Value>>,
    dht: Option<crate::dht::Handle>,
}

pub async fn serve(
    listen: SocketAddr,
    model_dir: PathBuf,
    device: String,
    peers: Vec<SocketAddr>,
    peer_token: Option<String>,
    dht: crate::dht::Handle,
) -> Result<()> {
    crate::security::loopback(listen)?;
    for peer in &peers {
        crate::security::loopback(*peer)?;
    }
    anyhow::ensure!(
        peers.is_empty() || peer_token.is_some(),
        "configured peers require --token-file"
    );
    let key = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let host = listen.to_string();
    let origin = format!("http://{host}");
    let app = App {
        key,
        origin,
        host,
        model_dir,
        device,
        peers,
        peer_token,
        job: Arc::new(Mutex::new(json!({"phase":"idle"}))),
        dht: Some(dht),
    };
    let listener = TcpListener::bind(listen).await?;
    // Fragment is not sent in HTTP requests. Do not share this local control URL.
    println!("Open this private local control URL in your browser:");
    println!("{}#{}", app.origin, app.key);
    axum::serve(listener, router(app))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn router(app: App) -> Router {
    Router::new()
        .route(
            "/",
            get(|| async {
                (
                    [("content-type", "text/html; charset=utf-8")],
                    include_str!("../ui/index.html"),
                )
            }),
        )
        .route(
            "/app.css",
            get(|| async {
                (
                    [("content-type", "text/css")],
                    include_str!("../ui/app.css"),
                )
            }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [("content-type", "text/javascript")],
                    include_str!("../ui/app.js"),
                )
            }),
        )
        .route("/api/dht", get(dht_status).post(dht_command))
        .route("/api/status", get(status))
        .route("/api/run", post(run))
        .layer(DefaultBodyLimit::max(20 * 1024))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .with_state(app)
}

async fn guard(State(app): State<App>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let host_ok = headers.get("host").and_then(|x| x.to_str().ok()) == Some(&app.host);
    let origin_ok = headers
        .get("origin")
        .is_none_or(|x| x == app.origin.as_str());
    let cross_site = headers
        .get("sec-fetch-site")
        .is_some_and(|x| x == "cross-site");
    let auth = !request.uri().path().starts_with("/api/")
        || headers
            .get("x-ui-key")
            .and_then(|x| x.to_str().ok())
            .is_some_and(|x| crate::security::token_matches(&app.key, x));
    let mut response = if !host_ok || !origin_ok || cross_site {
        (
            StatusCode::FORBIDDEN,
            Json(json!({"error":"origin or host rejected"})),
        )
            .into_response()
    } else if !auth {
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error":"Open the private URL printed by the UI command."})),
        )
            .into_response()
    } else {
        next.run(request).await
    };
    for (name, value) in [
        (
            "content-security-policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        ),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("cache-control", "no-store"),
        ("x-frame-options", "DENY"),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    response
}

async fn status(State(app): State<App>) -> Json<Value> {
    let manifest_hash = crate::qwen::load_manifest(&app.model_dir)
        .ok()
        .map(|(_, hash)| hash);
    Json(
        json!({"manifest_hash":manifest_hash,"device":app.device,"model_ready":app.model_dir.join("manifest.json").is_file(),
        "peers":app.peers,"peer_enabled":!app.peers.is_empty(),"job":app.job.lock().await.clone()}),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    prompt: String,
    max_tokens: usize,
    mode: String,
}
async fn run(State(app): State<App>, Json(input): Json<Run>) -> Response {
    if input.prompt.trim().is_empty()
        || input.prompt.len() > 16 * 1024
        || !(1..=128).contains(&input.max_tokens)
        || !["local", "peers"].contains(&input.mode.as_str())
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"Provide a prompt, 1–128 tokens, and a valid mode."})),
        )
            .into_response();
    }
    if input.mode == "peers" && app.peers.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"Start the UI with --peers and --token-file first."})),
        )
            .into_response();
    }
    let mut job = app.job.lock().await;
    if job["phase"] == "running" {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error":"A test is already running."})),
        )
            .into_response();
    }
    *job = json!({"phase":"running","mode":input.mode,"prompt":input.prompt});
    drop(job);
    tokio::spawn(async move {
        let options = Options {
            model_dir: app.model_dir,
            device: app.device,
            prompt: input.prompt,
            max_tokens: input.max_tokens,
            peers: if input.mode == "peers" {
                app.peers
            } else {
                vec![]
            },
            token: app
                .peer_token
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        };
        // Keep a failed inference task from leaving the UI permanently busy.
        let outcome = tokio::spawn(runner::run(options)).await;
        let value = match outcome {
            Ok(Ok(report)) => json!({"phase":"complete","report":report}),
            Ok(Err(error)) => json!({"phase":"error","error":error.to_string()}),
            Err(_) => {
                json!({"phase":"error","error":"Inference task stopped unexpectedly. Check the terminal."})
            }
        };
        *app.job.lock().await = value;
    });
    (StatusCode::ACCEPTED, Json(json!({"accepted":true}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn rejects_unauthenticated_cross_origin_and_rebinding_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let host = listener.local_addr().unwrap().to_string();
        let origin = format!("http://{host}");
        let app = App {
            key: "test-ui-secret".into(),
            origin: origin.clone(),
            host,
            model_dir: PathBuf::from("missing"),
            device: "cpu".into(),
            peers: vec![],
            peer_token: None,
            job: Arc::new(Mutex::new(json!({"phase":"idle"}))),
            dht: None,
        };
        let task = tokio::spawn(async move {
            axum::serve(listener, router(app)).await.unwrap();
        });
        let client = crate::server::client().unwrap();
        let endpoint = format!("{origin}/api/status");
        assert_eq!(
            client.get(&endpoint).send().await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            client
                .get(&endpoint)
                .header("x-ui-key", "test-ui-secret")
                .header("origin", "https://evil.invalid")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            client
                .get(&endpoint)
                .header("x-ui-key", "test-ui-secret")
                .header("host", "evil.invalid")
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN
        );
        let good = client
            .get(&endpoint)
            .header("x-ui-key", "test-ui-secret")
            .send()
            .await
            .unwrap();
        assert!(good.status().is_success());
        assert_eq!(good.headers()["cache-control"], "no-store");
        let response = client
            .post(format!("{origin}/api/run"))
            .header("x-ui-key", "test-ui-secret")
            .json(&json!({"mode":"peers","prompt":"test","max_tokens":1}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        task.abort();
    }
}

async fn dht_status(State(app): State<App>) -> Json<Value> {
    match app.dht {
        Some(dht) => Json(json!(dht.snapshot().await)),
        None => Json(json!({"error":"DHT unavailable"})),
    }
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum DhtCommand {
    Join { address: String },
    Find { model_hash: String },
    Publish { start: usize, end: usize },
}
async fn dht_command(State(app): State<App>, Json(command): Json<DhtCommand>) -> Response {
    let Some(dht) = app.dht else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let result = match command {
        DhtCommand::Join { address } => dht.join(&address).await,
        DhtCommand::Find { model_hash } => dht.find(model_hash).await,
        DhtCommand::Publish { start, end } => {
            // Advertise only a manifest present locally, with an actual shard hash verified.
            let dir = app.model_dir.clone();
            let checked = tokio::task::spawn_blocking(move || -> Result<String> {
                let (manifest, hash) = crate::qwen::load_manifest(&dir)?;
                let shard = manifest
                    .shards
                    .iter()
                    .find(|s| s.start == start && s.end == end)
                    .ok_or_else(|| anyhow::anyhow!("select a shard range in the local manifest"))?;
                crate::qwen::check_hash(&dir.join(&shard.file), &shard.sha256)?;
                Ok(hash)
            })
            .await;
            match checked {
                Ok(Ok(model_hash)) => {
                    dht.publish(crate::dht::Offer {
                        model_hash,
                        start,
                        end,
                    })
                    .await
                }
                Ok(Err(error)) => Err(error),
                Err(error) => Err(error.into()),
            }
        }
    };
    match result {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"accepted":true}))).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":error.to_string()})),
        )
            .into_response(),
    }
}
