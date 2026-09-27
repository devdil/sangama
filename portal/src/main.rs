use anyhow::{Context, Result, ensure};
use axum::{
    Form, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{env, sync::Arc};
use tokio::sync::Semaphore;
use tokio_postgres::{Client, NoTls};

#[derive(Clone)]
struct App {
    db: Arc<Client>,
    origin: String,
    host: String,
    capacity: Arc<Semaphore>,
}
fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
fn hash(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}
fn page(title: &str, body: &str) -> Html<String> {
    Html(format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{} · Sangama</title><link rel="stylesheet" href="/style.css"></head><body><div class="sheet"><header><a class="wordmark" href="/">संगम <span>Sangama</span></a><p>Computers working together.</p></header><nav aria-label="Main"><a href="/">Network directory</a><a href="/join">Register a device</a><a href="/about">How it works</a></nav><main>{}</main><footer>Sangama · Experimental peer-to-peer inference<br>Registration is an introduction. Device ownership and availability are not yet verified.</footer></div></body></html>"#,
        escape(title),
        body
    ))
}
fn failure(status: StatusCode, message: &str) -> Response {
    (status,page("Request unsuccessful",&format!("<h1>Request unsuccessful</h1><p>{}</p><p><a href=\"/join\">Return to registration</a></p>",escape(message)))).into_response()
}
async fn guard(State(app): State<App>, request: Request, next: Next) -> Response {
    if request.headers().get("host").and_then(|h| h.to_str().ok()) != Some(app.host.as_str()) {
        return failure(StatusCode::BAD_REQUEST, "Unexpected host.");
    }
    if request.method() == axum::http::Method::POST
        && request
            .headers()
            .get("origin")
            .and_then(|h| h.to_str().ok())
            != Some(app.origin.as_str())
    {
        return failure(
            StatusCode::FORBIDDEN,
            "Please submit the form from this website.",
        );
    }
    let Ok(_permit) = app.capacity.try_acquire() else {
        return failure(
            StatusCode::TOO_MANY_REQUESTS,
            "The portal is busy. Please try again shortly.",
        );
    };
    let mut response = next.run(request).await;
    for (key, value) in [
        (
            "content-security-policy",
            "default-src 'none'; style-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        ),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("cache-control", "no-store"),
        (
            "permissions-policy",
            "camera=(), microphone=(), geolocation=()",
        ),
    ] {
        response
            .headers_mut()
            .insert(key, HeaderValue::from_static(value));
    }
    response
}
async fn index(State(app): State<App>) -> Response {
    let rows=match app.db.query("SELECT name, platform, memory_gib FROM registrations ORDER BY created_at DESC LIMIT 100",&[]).await {Ok(r)=>r,Err(_)=>return failure(StatusCode::SERVICE_UNAVAILABLE,"The directory is temporarily unavailable.")};
    let count = match app
        .db
        .query_one("SELECT count(*) FROM registrations", &[])
        .await
    {
        Ok(r) => r.get::<_, i64>(0),
        Err(_) => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "The directory is temporarily unavailable.",
            );
        }
    };
    let mut body = format!(
        r#"<p class="eyebrow">THE SANGAMA NETWORK</p><h1>A place for your computer<br>to be part of something larger.</h1><p class="intro">Register a device, meet the network, and help explore running language models across computers.</p><p><a class="button" href="/join">Register your device →</a></p><section class="notice"><strong>Early access</strong><p>We are testing with invited peers. Registration does not connect your device or share its resources automatically.</p></section><h2>Device directory <span class="count">{count} registered</span></h2><p>Declared specifications. Listed devices may be offline. Showing the latest 100 registrations.</p><div class="table-wrap"><table><thead><tr><th>Device</th><th>Platform</th><th>Memory</th><th>Status</th></tr></thead><tbody>"#
    );
    if rows.is_empty() {
        body.push_str("<tr><td colspan=\"4\" class=\"empty\">No devices registered yet. Yours can be the first.</td></tr>");
    }
    for row in rows {
        body.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{} GiB</td><td>Registered · unverified</td></tr>",
            escape(row.get::<_, &str>(0)),
            escape(row.get::<_, &str>(1)),
            row.get::<_, i32>(2)
        ));
    }
    body.push_str("</tbody></table></div><h2>Start small. Measure honestly.</h2><p>Sangama currently splits one supported Qwen model between workers. Larger models, automatic placement, and open Internet participation are still being developed.</p>");
    page("Network directory", &body).into_response()
}
async fn join() -> Html<String> {
    page(
        "Register a device",
        r#"<p class="eyebrow">JOIN EARLY ACCESS</p><h1>Introduce your device.</h1><p>You need a single-use invitation from the network operator. Choose a public device name; it, your platform, and memory size appear in the directory. Do not include your name or address unless you want them public.</p><form action="/join" method="post"><label for="name">Public device name</label><input id="name" name="name" maxlength="80" required placeholder="e.g. Cedar MacBook"><label for="peer_id">Sangama peer ID</label><input id="peer_id" name="peer_id" minlength="32" maxlength="128" required autocomplete="off"><small>Copy the peer ID from your local Sangama node. It is stored privately for now; registration does not verify ownership.</small><label for="platform">Operating system</label><select id="platform" name="platform"><option>macOS</option><option>Linux</option><option>Windows</option><option>Other</option></select><label for="memory_gib">Memory you have available (GiB)</label><input id="memory_gib" name="memory_gib" type="number" min="1" max="4096" required><label for="invitation">Invitation code</label><input id="invitation" name="invitation" type="password" maxlength="64" required autocomplete="off"><small>Valid for 24 hours and one registration. Never enter a model-worker token here.</small><p><label class="check"><input type="checkbox" name="consent" value="yes" required> I agree to publish my device name, platform, and memory size.</label></p><button type="submit">Register device</button></form><p>Need an invitation? Ask the person who invited you to Sangama.</p>"#,
    )
}
#[derive(Deserialize)]
struct Registration {
    name: String,
    peer_id: String,
    platform: String,
    memory_gib: i32,
    invitation: String,
    consent: Option<String>,
}
fn valid(form: &Registration) -> bool {
    !form.name.trim().is_empty()
        && form.name.chars().count() <= 80
        && !form.name.chars().any(char::is_control)
        && (32..=128).contains(&form.peer_id.len())
        && form.peer_id.bytes().all(|c| c.is_ascii_alphanumeric())
        && ["macOS", "Linux", "Windows", "Other"].contains(&form.platform.as_str())
        && (1..=4096).contains(&form.memory_gib)
        && form.invitation.len() == 64
        && form.invitation.bytes().all(|c| c.is_ascii_hexdigit())
        && form.consent.as_deref() == Some("yes")
}
async fn register(State(app): State<App>, Form(form): Form<Registration>) -> Response {
    if !valid(&form) {
        return failure(
            StatusCode::BAD_REQUEST,
            "Please check the form fields and publication consent.",
        );
    }
    let result=app.db.query("WITH claimed AS (UPDATE invitations SET used_at=now() WHERE token_hash=$1 AND used_at IS NULL AND expires_at>now() RETURNING id) INSERT INTO registrations (name,peer_id,platform,memory_gib,invitation_id) SELECT $2,$3,$4,$5,id FROM claimed RETURNING id",&[&hash(&form.invitation),&form.name.trim(),&form.peer_id,&form.platform,&form.memory_gib]).await;
    match result {
        Ok(rows) if !rows.is_empty()=>(StatusCode::CREATED,page("Device registered","<h1>Your device is registered.</h1><p>Nothing is running on your computer yet. Ask the operator for the private network invitation and worker setup instructions before connecting.</p><p><a href=\"/\">View the directory →</a></p>")).into_response(),
        Ok(_)=>failure(StatusCode::FORBIDDEN,"This invitation is invalid, expired, or already used."),
        Err(e) if e.code()==Some(&tokio_postgres::error::SqlState::UNIQUE_VIOLATION)=>failure(StatusCode::CONFLICT,"This device is already registered."),
        Err(_)=>failure(StatusCode::SERVICE_UNAVAILABLE,"Registration is temporarily unavailable. Please try again."),
    }
}
async fn about() -> Html<String> {
    page(
        "How it works",
        r#"<p class="eyebrow">HOW IT WORKS</p><h1>One model. Several computers.</h1><ol><li><strong>Register.</strong> Introduce your device using an invitation.</li><li><strong>Connect privately.</strong> The operator helps you join the approved network and configure your worker.</li><li><strong>Discover.</strong> A bootstrap node introduces peers. The distributed hash table helps them find signed model advertisements.</li><li><strong>Run a shard.</strong> Each configured worker loads its assigned part of the model. Requests pass through workers in layer order.</li></ol><h2>What this portal does</h2><p>It keeps an invitation-based device directory. It does not allocate model layers, verify device ownership, grant inference access, or show live availability.</p><h2>What stays on your computer</h2><p>Your local worker holds its model shard and runs inference. Only contribute with people you trust: encrypted connections do not make an untrusted inference peer safe for private prompts.</p><p><a href="/join">Register a device →</a></p>"#,
    )
}
async fn health(State(app): State<App>) -> Response {
    match app.db.simple_query("SELECT 1").await {
        Ok(_) => (StatusCode::OK, "ok\n").into_response(),
        Err(_) => (StatusCode::SERVICE_UNAVAILABLE, "unavailable\n").into_response(),
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    let password = std::fs::read_to_string(
        env::var("DATABASE_PASSWORD_FILE").context("DATABASE_PASSWORD_FILE is required")?,
    )?;
    let mut config = tokio_postgres::Config::new();
    config
        .host(env::var("DATABASE_HOST").unwrap_or_else(|_| "postgres".into()))
        .user("sangama")
        .dbname("sangama")
        .password(password.trim())
        .connect_timeout(std::time::Duration::from_secs(5));
    let (client, connection) = config
        .connect(NoTls)
        .await
        .context("connect to private PostgreSQL")?;
    tokio::spawn(async move {
        if connection.await.is_err() {
            eprintln!("Database connection lost; restarting portal");
            std::process::exit(1);
        }
    });
    client.batch_execute(include_str!("../schema.sql")).await?;
    if env::args().nth(1).as_deref() == Some("invite") {
        let token = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        client
            .execute(
                "INSERT INTO invitations (token_hash) VALUES ($1)",
                &[&hash(&token)],
            )
            .await?;
        println!("{token}");
        return Ok(());
    }
    let origin = env::var("PUBLIC_ORIGIN").context("PUBLIC_ORIGIN is required")?;
    let host = origin
        .strip_prefix("https://")
        .or_else(|| origin.strip_prefix("http://"))
        .context("origin must use HTTPS or localhost HTTP")?
        .to_owned();
    ensure!(
        !host.is_empty() && !host.contains('/') && !host.contains('@'),
        "invalid origin"
    );
    ensure!(
        origin.starts_with("https://") || host.starts_with("127.0.0.1:"),
        "HTTP is permitted only for localhost tests"
    );
    let app = App {
        db: Arc::new(client),
        origin,
        host,
        capacity: Arc::new(Semaphore::new(16)),
    };
    let router = Router::new()
        .route("/", get(index))
        .route("/join", get(join).post(register))
        .route("/about", get(about))
        .route("/healthz", get(health))
        .route(
            "/style.css",
            get(|| async {
                (
                    [("content-type", "text/css")],
                    include_str!("../static/style.css"),
                )
            }),
        )
        .layer(DefaultBodyLimit::max(8192))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .with_state(app);
    let listen = env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let listener = tokio::net::TcpListener::bind(listen).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escapes_user_content() {
        assert_eq!(escape("<script>&\"'"), "&lt;script&gt;&amp;&quot;&#39;");
    }
    #[test]
    fn requires_consent_and_bounded_values() {
        let mut f = Registration {
            name: "My Mac".into(),
            peer_id: "a".repeat(52),
            platform: "macOS".into(),
            memory_gib: 24,
            invitation: "a".repeat(64),
            consent: Some("yes".into()),
        };
        assert!(valid(&f));
        f.consent = None;
        assert!(!valid(&f));
        f.consent = Some("yes".into());
        f.memory_gib = -1;
        assert!(!valid(&f));
    }
}
