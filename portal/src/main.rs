mod accounts;
mod admin;
mod membership;
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
    authority: Option<Arc<sangama_network_auth::libp2p_identity::Keypair>>,
    network: String,
    admin_token: Option<String>,
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
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{} · Sangama</title><link rel="stylesheet" href="/style.css"></head><body><div class="sheet"><header><a class="wordmark" href="/">संगम <span>Sangama</span></a><p>Computers working together.</p></header><nav aria-label="Main"><a href="/">Network directory</a><a href="/join">Sign up</a><a href="/signin">Sign in</a><a href="/connect">Connect a worker</a><a href="/about">How it works</a><a href="/admin">Operator</a></nav><main>{}</main><footer>Sangama · Experimental peer-to-peer inference<br>Portal accounts and network membership are separate. Membership verifies a peer key; it does not prove that a worker is online.</footer></div></body></html>"#,
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
    let machine_api = request.uri().path().starts_with("/v1/membership/");
    if machine_api && request.headers().contains_key("origin") {
        return failure(
            StatusCode::FORBIDDEN,
            "Machine API does not accept browser requests.",
        );
    }
    if !machine_api
        && request.method() == axum::http::Method::POST
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
    let rows=match app.db.query("SELECT r.name, r.platform, r.memory_gib, coalesce(m.role,''), coalesce(NOT m.revoked AND m.expires_at>now(),false) FROM registrations r LEFT JOIN network_members m ON m.peer_id=r.peer_id ORDER BY r.created_at DESC LIMIT 100",&[]).await {Ok(r)=>r,Err(_)=>return failure(StatusCode::SERVICE_UNAVAILABLE,"The directory is temporarily unavailable.")};
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
        r#"<p class="eyebrow">THE SANGAMA NETWORK</p><h1>A place for your computer<br>to be part of something larger.</h1><p class="intro">Create an account, meet the network, and help explore running language models across computers.</p><p><a class="button" href="/join">Sign up</a> <a class="button" href="/signin">Sign in</a></p><section class="notice"><strong>Early access</strong><p>We are testing with invited peers. Registration does not connect your device or share its resources automatically.</p></section><h2>Device directory <span class="count">{count} registered</span></h2><p>Declared specifications. Listed devices may be offline. Showing the latest 100 registrations.</p><div class="table-wrap"><table><thead><tr><th>Device</th><th>Platform</th><th>Memory</th><th>Status</th></tr></thead><tbody>"#
    );
    if rows.is_empty() {
        body.push_str("<tr><td colspan=\"4\" class=\"empty\">No devices registered yet. Yours can be the first.</td></tr>");
    }
    for row in rows {
        let member: bool = row.get(4);
        let status = if member {
            format!("Admitted {} · availability unknown", row.get::<_, &str>(3))
        } else {
            "Registered · not admitted".into()
        };
        body.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{} GiB</td><td>{}</td></tr>",
            escape(row.get::<_, &str>(0)),
            escape(row.get::<_, &str>(1)),
            row.get::<_, i32>(2),
            escape(&status)
        ));
    }
    body.push_str("</tbody></table></div><h2>Start small. Measure honestly.</h2><p>Sangama currently splits one supported Qwen model between workers. Invited peers can use an encrypted relay, memory-aware shard loading and readiness checks. Manage allocation and inference from your local UI. This is an experimental trusted network; arbitrary models and anonymous public participation are not supported.</p>");
    page("Network directory", &body).into_response()
}
async fn connect() -> Html<String> {
    page(
        "Connect a worker",
        r#"<h1>Connect your worker</h1><p>Portal signup and cryptographic admission use different invitations. Request a network invitation with the worker role, the authority public key through a trusted channel, a relay address and your node configuration from the operator.</p><ol><li>Build Sangama and prepare the supported Qwen shard files locally.</li><li>Create a persistent identity. Share only the peer ID/public key with the operator.</li><li>Save your network invitation to a private file (0600), then redeem it locally. Your worker signs the challenge; never upload its private key here.</li><li>Start the managed node. It begins unloaded; the local inference UI can allocate prepared shards and check readiness.</li></ol><pre>target/release/sangama mesh-identity --state-dir .mesh/worker
 target/release/sangama mesh-join --config worker.json --invitation-file /path/to/private-invitation
 python3 scripts/mesh-node.py --config worker.json</pre><h2>Understand your status</h2><dl><dt>Registered</dt><dd>A directory entry exists; no network access is implied.</dd><dt>Admitted</dt><dd>A peer key has valid scoped membership. It may still be offline.</dd><dt>Reachable / loaded / ready</dt><dd>Checked by your local UI against configured bridges. A complete verified route is needed to generate.</dd><dt>Interrupted</dt><dd>Recover workers, allocate/check the route and start a new request. Previous KV state is not resumed.</dd></dl><p>The portal cannot measure worker memory or relay paths. Model weights and credentials stay on your computer. Participating workers can inspect their inference data.</p><p><a href="https://github.com/devdil/sangama/blob/main/docs/admitted-mesh.md">Full node configuration and setup guide</a></p>"#,
    )
}
async fn about() -> Html<String> {
    page(
        "How it works",
        r#"<p class="eyebrow">HOW IT WORKS</p><h1>One model. Several computers.</h1><ol><li><strong>Register.</strong> Create an account using an invitation.</li><li><strong>Connect privately.</strong> Use a scoped invitation and the pinned authority public key. Your worker signs a one-time challenge to join. Expired or revoked memberships lose access.</li><li><strong>Discover.</strong> A bootstrap node introduces peers. The distributed hash table helps them find signed model advertisements.</li><li><strong>Run a shard.</strong> Each configured worker loads its assigned part of the model. Requests pass through workers in layer order.</li></ol><h2>What this portal does</h2><p>It provides invite-only accounts and a device directory. The directory alone does not grant access. The membership service verifies peer-key ownership and authorizes a worker, client, or relay role for 24 hours. It does not prove physical device ownership or show live worker availability.</p><h2>What stays on your computer</h2><p>Your local worker holds its model shard and runs inference. Only contribute with people you trust: encrypted connections do not make an untrusted inference peer safe for private prompts.</p><p><a href="/join">Register a device →</a></p>"#,
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
    if env::args().nth(1).as_deref() == Some("authority-init") {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let path = std::path::PathBuf::from(
            env::var("MEMBERSHIP_KEY_FILE").context("MEMBERSHIP_KEY_FILE required")?,
        );
        if !path.exists() {
            let key = sangama_network_auth::libp2p_identity::Keypair::generate_ed25519();
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)?;
            file.write_all(&key.to_protobuf_encoding()?)?;
        }
        ensure!(
            !path.symlink_metadata()?.file_type().is_symlink(),
            "authority key must not be a symlink"
        );
        let key = sangama_network_auth::libp2p_identity::Keypair::from_protobuf_encoding(
            &std::fs::read(&path)?,
        )?;
        std::fs::write(path.with_extension("pub"), key.public().encode_protobuf())?;
        // A named read-only Docker secret; its host parent must remain private.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444))?;
        println!("Authority initialized; distribute only the .pub file.");
        return Ok(());
    }

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
    let action = env::args().nth(1).unwrap_or_default();
    if action == "network-invite" {
        let role = env::args().nth(2).unwrap_or_else(|| "worker".into());
        ensure!(
            ["worker", "client", "relay"].contains(&role.as_str()),
            "invalid role"
        );
        let token = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        client
            .execute(
                "INSERT INTO network_invitations (token_hash,role) VALUES ($1,$2)",
                &[&hash(&token), &role],
            )
            .await?;
        println!("{token}");
        return Ok(());
    }
    if action == "revoke" {
        let peer = env::args().nth(2).context("peer ID required")?;
        ensure!(
            client
                .execute(
                    "UPDATE network_members SET revoked=true WHERE peer_id=$1",
                    &[&peer]
                )
                .await?
                == 1,
            "member not found"
        );
        println!("Membership revoked");
        return Ok(());
    }
    let authority = env::var("MEMBERSHIP_KEY_FILE")
        .ok()
        .map(|p| -> Result<_> {
            let bytes = std::fs::read(p)?;
            Ok(Arc::new(
                sangama_network_auth::libp2p_identity::Keypair::from_protobuf_encoding(&bytes)?,
            ))
        })
        .transpose()?;
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
        origin.starts_with("https://")
            || host.starts_with("127.0.0.1:")
            || env::var("SIMULATION_HTTP").as_deref() == Ok("1"),
        "HTTP is permitted only for localhost tests"
    );
    let app = App {
        db: Arc::new(client),
        origin,
        host,
        capacity: Arc::new(Semaphore::new(16)),
        authority,
        admin_token: env::var("ADMIN_TOKEN_FILE")
            .ok()
            .map(|path| -> Result<String> {
                let token = std::fs::read_to_string(path)?.trim().to_owned();
                ensure!(
                    token.len() == 64 && token.bytes().all(|c| c.is_ascii_hexdigit()),
                    "admin credential must be 32 random bytes encoded as hex"
                );
                Ok(token)
            })
            .transpose()?,
        network: env::var("NETWORK_ID").unwrap_or_else(|_| "sangama-private-v1".into()),
    };
    let router = Router::new()
        .route("/", get(index))
        .route("/join", get(accounts::signup_form).post(accounts::signup))
        .route("/signin", get(accounts::signin_form).post(accounts::signin))
        .route("/account", get(accounts::account))
        .route("/signout", axum::routing::post(accounts::signout))
        .route("/about", get(about))
        .route("/connect", get(connect))
        .route("/admin", get(admin::form).post(admin::submit))
        .route("/healthz", get(health))
        .route(
            "/v1/membership/challenge",
            axum::routing::post(membership::challenge),
        )
        .route(
            "/v1/membership/redeem",
            axum::routing::post(membership::redeem),
        )
        .route("/v1/membership/snapshot", get(membership::snapshot))
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
}
