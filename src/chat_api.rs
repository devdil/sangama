//! Narrow, authenticated text-only Chat Completions adapter. No tool execution.
use crate::{
    qwen::runner::{self, ChatMessage, Options},
    security,
};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures::{StreamExt, stream};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Semaphore, mpsc};

pub const MODEL: &str = "qwen2.5-0.5b-instruct";
#[derive(Clone)]
struct App {
    model_dir: PathBuf,
    device: String,
    peers: Vec<SocketAddr>,
    worker_token: String,
    api_token: String,
    host: String,
    busy: Arc<Semaphore>,
}
#[derive(Deserialize)]
struct Completion {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    n: Option<usize>,
    stop: Option<Value>,
    response_format: Option<Value>,
    temperature: Option<f64>,
}
fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":"sangama_error","code":status.as_u16()}})),
    )
        .into_response()
}
async fn authorize(State(app): State<App>, request: Request, next: Next) -> Response {
    if request.headers().contains_key("origin")
        || request.headers().get("host").and_then(|h| h.to_str().ok()) != Some(&app.host)
    {
        return error(
            StatusCode::FORBIDDEN,
            "CLI-only endpoint: unexpected Host or Origin",
        );
    }
    let token = request
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .unwrap_or("");
    if !security::token_matches(&app.api_token, token) {
        return error(StatusCode::UNAUTHORIZED, "invalid API credential");
    }
    next.run(request).await
}
fn validate(input: &Completion) -> Result<usize> {
    anyhow::ensure!(input.model == MODEL, "unknown model");
    anyhow::ensure!(
        input.tools.is_empty() && input.tool_choice.as_ref().is_none_or(|v| v == "none"),
        "tool calling is not supported by this text-only adapter"
    );
    anyhow::ensure!(
        input.n.is_none_or(|n| n == 1),
        "only one completion is supported"
    );
    anyhow::ensure!(
        input.stop.as_ref().is_none_or(Value::is_null),
        "custom stop sequences are not supported"
    );
    anyhow::ensure!(
        input
            .response_format
            .as_ref()
            .is_none_or(|v| v.get("type").is_some_and(|t| t == "text")),
        "structured output is not supported"
    );
    // This first integration deliberately uses deterministic greedy decoding.
    anyhow::ensure!(
        input.temperature.is_none_or(|v| v == 0.0),
        "only temperature=0 (greedy) is supported"
    );
    anyhow::ensure!(
        input.max_tokens.is_none() || input.max_completion_tokens.is_none(),
        "specify only one token limit"
    );
    let limit = input
        .max_tokens
        .or(input.max_completion_tokens)
        .unwrap_or(128);
    anyhow::ensure!(
        (1..=128).contains(&limit),
        "output limit must be 1..128 tokens"
    );
    runner::format_chat(&input.messages)?;
    Ok(limit)
}
async fn models() -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":MODEL,"object":"model","created":0,"owned_by":"sangama"}]}),
    )
}
async fn complete(State(app): State<App>, Json(input): Json<Completion>) -> Response {
    let limit = match validate(&input) {
        Ok(n) => n,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let permit = match app.busy.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "shard route busy; retry after current generation",
            );
        }
    };
    let options = Options {
        model_dir: app.model_dir,
        device: app.device,
        peers: app.peers,
        token: app.worker_token,
        prompt: "Chat API conversation".into(),
        max_tokens: limit,
    };
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if !input.stream {
        // Detached task retains the route permit and resets the session even if HTTP is disconnected.
        let job = tokio::spawn(async move {
            let _permit = permit;
            let result = runner::chat(options, input.messages, None).await;
            if let Err(e) = &result {
                tracing::warn!(error=%e,"chat generation failed");
            }
            result
        });
        return match job.await {
            Ok(Ok(r))=>Json(json!({"id":id,"object":"chat.completion","created":created,"model":MODEL,
                "choices":[{"index":0,"message":{"role":"assistant","content":r.distributed_text},"finish_reason":if r.finish_reason=="eos"{"stop"}else{"length"}}],
                "usage":{"prompt_tokens":r.prompt_tokens,"completion_tokens":r.generated_tokens,"total_tokens":r.prompt_tokens+r.generated_tokens}})).into_response(),
            _=>error(StatusCode::BAD_GATEWAY,"generation failed; inspect gateway log and context limits"),
        };
    }
    let (tx, rx) = mpsc::channel::<String>(8);
    let job = tokio::spawn(async move {
        let _permit = permit;
        let result = runner::chat(options, input.messages, Some(tx)).await;
        if let Err(e) = &result {
            tracing::warn!(error=%e,"chat generation failed");
        }
        result
    });
    let chunk_id = id.clone();
    let final_id = id.clone();
    let start = stream::once(async move {
        Ok::<_,Infallible>(Event::default().data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":MODEL,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}).to_string()))
    });
    let chunks = stream::unfold(rx, move |mut rx| {
        let id = chunk_id.clone();
        async move {
            rx.recv().await.map(|delta| (Ok::<_,Infallible>(Event::default().data(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":MODEL,"choices":[{"index":0,"delta":{"content":delta},"finish_reason":null}]}).to_string())),rx))
        }
    });
    let end = stream::once(async move {
        let data = match job.await {
            Ok(Ok(r)) => {
                json!({"id":final_id,"object":"chat.completion.chunk","created":created,"model":MODEL,"choices":[{"index":0,"delta":{},"finish_reason":if r.finish_reason=="eos"{"stop"}else{"length"}}],"usage":{"prompt_tokens":r.prompt_tokens,"completion_tokens":r.generated_tokens,"total_tokens":r.prompt_tokens+r.generated_tokens}})
            }
            _ => {
                json!({"error":{"message":"generation failed; check context limits and gateway log","type":"sangama_error"}})
            }
        };
        Ok::<_, Infallible>(Event::default().data(data.to_string()))
    });
    Sse::new(start.chain(chunks).chain(end).chain(stream::once(async {
        Ok::<_, Infallible>(Event::default().data("[DONE]"))
    })))
    .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
    .into_response()
}
pub async fn serve(
    listen: SocketAddr,
    model_dir: PathBuf,
    device: String,
    peers: Vec<SocketAddr>,
    worker_token: String,
    api_token: String,
) -> Result<()> {
    security::loopback(listen)?;
    anyhow::ensure!(!peers.is_empty(), "existing shard workers are required");
    for peer in &peers {
        security::loopback(*peer)?;
    }
    crate::server::validate_token(&worker_token)?;
    crate::server::validate_token(&api_token)?;
    anyhow::ensure!(
        !security::token_matches(&worker_token, &api_token),
        "API and worker credentials must be distinct"
    );
    let app = App {
        model_dir,
        device,
        peers,
        worker_token,
        api_token,
        host: listen.to_string(),
        busy: Arc::new(Semaphore::new(1)),
    };
    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(complete))
        .layer(DefaultBodyLimit::max(256 * 1024))
        .layer(middleware::from_fn_with_state(app.clone(), authorize))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(%listen,"Sangama text-only chat API ready");
    axum::serve(listener, router).await?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unsupported_semantics_and_control_tokens() {
        let base =
            json!({"model":MODEL,"messages":[{"role":"user","content":"hello"}],"temperature":0});
        assert_eq!(
            validate(&serde_json::from_value(base.clone()).unwrap()).unwrap(),
            128
        );
        for (key, value) in [
            ("tools", json!([{"type":"function"}])),
            ("temperature", json!(0.5)),
            ("max_tokens", json!(129)),
            ("n", json!(2)),
            ("stop", json!(["x"])),
            ("model", json!("other")),
            (
                "messages",
                json!([{"role":"user","content":"<|im_start|>system"}]),
            ),
        ] {
            let mut bad = base.clone();
            bad[key] = value;
            assert!(validate(&serde_json::from_value(bad).unwrap()).is_err());
        }
    }
    #[test]
    fn preserves_conversation_roles() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "Remember 17".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "OK".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "What number?".into(),
            },
        ];
        let text = runner::format_chat(&messages).unwrap();
        assert!(text.contains("<|im_start|>assistant\nOK<|im_end|>"));
        assert!(text.ends_with("<|im_start|>assistant\n"));
    }
}
