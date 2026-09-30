//! Authenticated libp2p transport for the existing loopback-only Qwen API.
//! No remote URL is accepted: each node can reach only its configured local worker.
use anyhow::{Context, Result, ensure};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::any,
};
use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, SwarmBuilder, Transport,
    allow_block_list::{self, AllowedPeers},
    autonat, dcutr, identify, noise, relay, request_response as rr,
    swarm::{NetworkBehaviour, SwarmEvent, behaviour::toggle::Toggle},
    yamux,
};
use sangama_network_auth::{
    Join, SignedSnapshot, credits::SignedStanding, libp2p_identity::PublicKey, now, proof,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Semaphore, mpsc, oneshot};
const LIMIT: usize = crate::qwen::wire::MAX_FRAME_BYTES;
// Sized for workers that serve many sessions at once (`qwen-worker --slots`): each session keeps
// a result request open on the last stage while it waits, and sends one frame per token through
// every hop, so these must exceed the sessions a worker serves (up to 256).
/// Requests a local bridge forwards at once.
const BRIDGE_CONCURRENCY: usize = 512;
/// Requests this node serves for peers at once.
const SERVE_CONCURRENCY: usize = 512;
/// Calls to peers this node has in flight at once.
const PENDING_CALLS: usize = 1024;
/// Streams at once over one connection, e.g. one relayed circuit between two stages.
const STREAMS_PER_CONNECTION: usize = 512;
/// Per-peer requests and reply bytes in each 60-second window.
const PEER_REQUESTS_PER_MINUTE: usize = 20_000;
const PEER_BYTES_PER_MINUTE: usize = 2 << 30;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bridge {
    pub listen: SocketAddr,
    pub peer: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub state_dir: PathBuf,
    pub authority_file: PathBuf,
    pub network: String,
    pub portal: String,
    #[serde(default)]
    pub test_http: bool,
    pub listen: Multiaddr,
    #[serde(default)]
    pub external: Vec<Multiaddr>,
    #[serde(default)]
    pub relay_server: bool,
    pub relay: Option<Multiaddr>,
    pub worker: Option<SocketAddr>,
    pub managed: Option<crate::managed_worker::Config>,
    pub token_file: PathBuf,
    #[serde(default)]
    pub bridges: Vec<Bridge>,
    #[serde(default)]
    pub force_relay: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    path: String,
    #[serde(with = "serde_bytes")]
    body: Vec<u8>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Reply {
    status: u16,
    #[serde(with = "serde_bytes")]
    body: Vec<u8>,
}
impl Reply {
    fn error(status: u16, message: &str) -> Self {
        Self {
            status,
            body: message.as_bytes().to_vec(),
        }
    }
}
#[derive(NetworkBehaviour)]
struct Behaviour {
    allow: allow_block_list::Behaviour<AllowedPeers>,
    limits: libp2p::connection_limits::Behaviour,
    relay_client: relay::client::Behaviour,
    relay_server: Toggle<relay::Behaviour>,
    identify: identify::Behaviour,
    dcutr: Toggle<dcutr::Behaviour>,
    autonat: Toggle<autonat::Behaviour>,
    rpc: rr::cbor::Behaviour<Request, Reply>,
    kad: libp2p::kad::Behaviour<crate::mesh_store::Store>,
}
enum Command {
    Call {
        peer: PeerId,
        request: Request,
        result: oneshot::Sender<Reply>,
    },
    Reply {
        peer: PeerId,
        channel: rr::ResponseChannel<Reply>,
        reply: Reply,
    },
    Membership(SignedSnapshot),
    Offer(crate::qwen::network::Info),
    Standing(Option<SignedStanding>),
    Discover(oneshot::Sender<Reply>),
    Status(PeerId, oneshot::Sender<Reply>),
}
#[derive(Clone)]
struct Proxy {
    tx: mpsc::Sender<Command>,
    peer: PeerId,
    token: String,
    slots: Arc<Semaphore>,
    meter: Arc<std::sync::Mutex<crate::credits::Meter>>,
    bridges: Arc<HashMap<SocketAddr, PeerId>>,
}
/// Every request opens a new substream. Yamux starts each at a 256 KiB window and grows it only
/// round trip by round trip, so a 4 MiB frame over a long relayed hop was window-bound. Grant
/// a whole frame's window as soon as a substream opens instead.
fn multiplexer() -> yamux::Config {
    let mut config = yamux::Config::default();
    #[allow(deprecated)]
    config
        .set_receive_window_size(2 * LIMIT as u32)
        .set_max_buffer_size(2 * LIMIT);
    config
}
fn valid_path(path: &str) -> bool {
    [
        "/v1/qwen/info",
        "/v1/qwen/forward",
        "/v1/qwen/result",
        "/v1/qwen/reset",
        "/v1/qwen/reserve",
        "/v1/node/capacity",
        "/v1/node/reserve",
        "/v1/node/load",
        "/v1/node/release",
    ]
    .contains(&path)
}
async fn proxy(
    State(s): State<Proxy>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    bytes: Bytes,
) -> Response {
    use subtle::ConstantTimeEq;
    let expected = format!("Bearer {}", s.token);
    let supplied = headers
        .get("authorization")
        .map(|x| x.as_bytes())
        .unwrap_or_default();
    if headers.contains_key("origin") || supplied.ct_eq(expected.as_bytes()).unwrap_u8() != 1 {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let path = uri.path();
    if ["/v1/mesh/offers", "/v1/mesh/status"].contains(&path)
        && method == Method::GET
        && uri.query().is_none()
    {
        let (tx, rx) = oneshot::channel();
        let command = if path.ends_with("status") {
            Command::Status(s.peer, tx)
        } else {
            Command::Discover(tx)
        };
        if s.tx.try_send(command).is_err() {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        return match tokio::time::timeout(Duration::from_secs(2), rx).await {
            Ok(Ok(r)) => (
                StatusCode::from_u16(r.status).unwrap_or(StatusCode::BAD_GATEWAY),
                [("content-type", "application/json")],
                r.body,
            )
                .into_response(),
            _ => StatusCode::GATEWAY_TIMEOUT.into_response(),
        };
    }
    if !valid_path(path)
        || uri.query().is_some()
        || ((path.ends_with("info") || path.ends_with("capacity")) && method != Method::GET)
        || (!(path.ends_with("info") || path.ends_with("capacity")) && method != Method::POST)
    {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let Ok(_permit) = s.slots.try_acquire() else {
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    };
    let (tx, rx) = oneshot::channel();
    if s.tx
        .try_send(Command::Call {
            peer: s.peer,
            request: Request {
                path: path.into(),
                body: bytes.to_vec(),
            },
            result: tx,
        })
        .is_err()
    {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    match tokio::time::timeout(Duration::from_secs(65), rx).await {
        Ok(Ok(reply)) => {
            if (200..300).contains(&reply.status) {
                meter_client(&s, path, &bytes, &reply.body);
            }
            (
                StatusCode::from_u16(reply.status).unwrap_or(StatusCode::BAD_GATEWAY),
                [(
                    "content-type",
                    if path.ends_with("forward") || path.ends_with("result") {
                        "application/octet-stream"
                    } else {
                        "application/json"
                    },
                )],
                reply.body,
            )
                .into_response()
        }
        _ => StatusCode::GATEWAY_TIMEOUT.into_response(),
    }
}
/// Counts this client's sessions: which peer took each stage, and the tokens the route returned.
fn meter_client(s: &Proxy, path: &str, request: &[u8], reply: &[u8]) {
    let now = Instant::now();
    if path == "/v1/qwen/info" {
        if let Ok(info) = serde_json::from_slice::<crate::qwen::network::Info>(reply) {
            s.meter.lock().unwrap().record_info(
                s.peer,
                &info.model_hash,
                (info.shard.start as u32, info.shard.end as u32),
            );
        }
    } else if path == "/v1/qwen/reserve" {
        #[derive(Deserialize)]
        struct Reserve {
            session: String,
            #[serde(default)]
            upstream: Option<SocketAddr>,
        }
        let Ok(r) = serde_json::from_slice::<Reserve>(request) else {
            return;
        };
        let upstream = match r.upstream {
            Some(address) => match s.bridges.get(&address) {
                Some(peer) => Some(*peer),
                None => return,
            },
            None => None,
        };
        s.meter
            .lock()
            .unwrap()
            .record_reserve(&r.session, s.peer, upstream, now);
    } else if path == "/v1/qwen/forward" {
        use crate::qwen::wire::Frame;
        let (Ok(sent), Ok(output)) = (Frame::header(request), Frame::header(reply)) else {
            return;
        };
        if output.session == sent.session && output.position == sent.position {
            s.meter.lock().unwrap().record_usage(
                &sent.session,
                &sent.model_hash,
                sent.seq_len as u64,
                now,
            );
        }
    }
}
fn client(portal: &str, test_http: bool) -> Result<reqwest::Client> {
    let url = reqwest::Url::parse(portal)?;
    ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "invalid portal URL"
    );
    ensure!(
        url.scheme() == "https" || (test_http && url.scheme() == "http"),
        "portal requires HTTPS (test-http is for isolated simulations only)"
    );
    Ok(reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(5))
        .build()?)
}
async fn fetch(
    http: &reqwest::Client,
    portal: &str,
    authority: &PublicKey,
    network: &str,
) -> Result<SignedSnapshot> {
    let mut r = http
        .get(format!(
            "{}/v1/membership/snapshot",
            portal.trim_end_matches('/')
        ))
        .send()
        .await?
        .error_for_status()?;
    let mut bytes = Vec::new();
    while let Some(c) = r.chunk().await? {
        ensure!(
            bytes.len() + c.len() <= 256 * 1024,
            "membership response too large"
        );
        bytes.extend_from_slice(&c);
    }
    let snapshot: SignedSnapshot = serde_json::from_slice(&bytes)?;
    snapshot.verify(authority, network, now())?;
    Ok(snapshot)
}
pub fn identity(dir: &Path) -> Result<()> {
    let (key, _lock) = crate::dht::identity(dir)?;
    std::fs::write(dir.join("public.key"), key.public().encode_protobuf())?;
    println!("{}", key.public().to_peer_id());
    Ok(())
}
pub async fn join(config: &Path, invitation: &Path) -> Result<()> {
    let c: Config = serde_json::from_slice(&std::fs::read(config)?)?;
    let (key, _lock) = crate::dht::identity(&c.state_dir)?;
    let authority = PublicKey::try_decode_protobuf(&std::fs::read(&c.authority_file)?)?;
    let http = client(&c.portal, c.test_http)?;
    // Authenticate the pinned authority before transmitting the invitation.
    fetch(&http, &c.portal, &authority, &c.network).await?;
    let invitation = crate::security::read_token(invitation)?;
    let response: serde_json::Value = http
        .post(format!(
            "{}/v1/membership/challenge",
            c.portal.trim_end_matches('/')
        ))
        .json(&serde_json::json!({"invitation":invitation}))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        response["network"].as_str() == Some(&c.network),
        "wrong challenge network"
    );
    let nonce = response["nonce"]
        .as_str()
        .context("missing nonce")?
        .to_string();
    let signature = key.sign(&proof(&c.network, &nonce, &invitation)?)?;
    http.post(format!(
        "{}/v1/membership/redeem",
        c.portal.trim_end_matches('/')
    ))
    .json(&Join {
        invitation,
        nonce,
        public_key: key.public().encode_protobuf(),
        signature,
    })
    .send()
    .await?
    .error_for_status()?;
    let s = fetch(&http, &c.portal, &authority, &c.network).await?;
    ensure!(
        s.snapshot
            .members
            .iter()
            .any(|m| m.peer == key.public().to_peer_id().to_string() && m.expires > now()),
        "membership not active"
    );
    println!("Joined {} as {}", c.network, key.public().to_peer_id());
    Ok(())
}
async fn worker(
    http: reqwest::Client,
    address: SocketAddr,
    token: String,
    request: Request,
) -> Reply {
    let result: Result<Reply> = async {
        ensure!(
            valid_path(&request.path) && request.body.len() <= LIMIT,
            "invalid request"
        );
        let url = format!("http://{address}{}", request.path);
        let r = if request.path.ends_with("info") {
            http.get(url)
        } else {
            http.post(url)
                .header(
                    "content-type",
                    if request.path.ends_with("forward") {
                        "application/octet-stream"
                    } else {
                        "application/json"
                    },
                )
                .body(request.body)
        };
        let mut r = r.bearer_auth(token).send().await?;
        let status = r.status().as_u16();
        let mut body = Vec::new();
        while let Some(c) = r.chunk().await? {
            ensure!(body.len() + c.len() <= LIMIT, "worker response too large");
            body.extend_from_slice(&c);
        }
        Ok(Reply { status, body })
    }
    .await;
    result.unwrap_or_else(|_| Reply::error(503, "Local worker unavailable"))
}

pub async fn run(path: &Path) -> Result<()> {
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        c.bridges.len() <= crate::qwen::MAX_SHARDS,
        "too many bridges"
    );
    if let Some(w) = c.worker {
        crate::security::loopback(w)?;
    }
    for b in &c.bridges {
        crate::security::loopback(b.listen)?;
        let _: PeerId = b.peer.parse()?;
    }
    let token = crate::security::read_token(&c.token_file)?;
    let (key, _lock) = crate::dht::identity(&c.state_dir)?;
    let me = key.public().to_peer_id();
    let authority = PublicKey::try_decode_protobuf(&std::fs::read(&c.authority_file)?)?;
    let http = client(&c.portal, c.test_http)?;
    let mut membership = fetch(&http, &c.portal, &authority, &c.network).await?;
    let own_role = if c.relay_server {
        "relay"
    } else if c.worker.is_some() {
        "worker"
    } else {
        "client"
    };
    ensure!(
        membership
            .snapshot
            .members
            .iter()
            .any(|m| m.peer == me.to_string() && m.role == own_role && m.expires > now()),
        "node lacks active membership with required role"
    );
    let managed = if let Some(config) = c.managed.clone() {
        Some(Arc::new(crate::managed_worker::Manager::new(
            config,
            me.to_string(),
            c.worker
                .context("managed node requires worker loopback address")?,
            token.clone(),
            c.token_file.clone(),
            c.bridges.iter().map(|b| b.listen).collect(),
        )?))
    } else {
        None
    };
    let signing_key = key.clone();
    let outbox = Arc::new(tokio::sync::Mutex::new(crate::credits::Outbox::default()));
    let meter = Arc::new(std::sync::Mutex::new(crate::credits::Meter::new(
        me,
        c.network.clone(),
        crate::qwen::LAYERS as u32,
    )));
    let store = crate::mesh_store::Store::open(
        &c.state_dir.join("mesh-discovery.sqlite"),
        me,
        c.network.clone(),
    )?;
    let mut kc = libp2p::kad::Config::new(StreamProtocol::new("/sangama/admitted-kad/1"));
    kc.set_query_timeout(Duration::from_secs(10))
        .set_record_filtering(libp2p::kad::StoreInserts::FilterBoth)
        .set_max_packet_size(32 * 1024)
        .set_record_ttl(Some(Duration::from_secs(60)));
    let mut kad = libp2p::kad::Behaviour::with_config(me, store, kc);
    // Nodes reached through a relay are DHT clients: they store and query offers on the relay.
    // As servers, every node would query and store on every other node through relayed
    // circuits every few seconds (measured: 20 workers opened ~170 circuits and the relay
    // path's round trip grew from 6 ms to ~350 ms).
    kad.set_mode(Some(if c.relay.is_some() {
        libp2p::kad::Mode::Client
    } else {
        libp2p::kad::Mode::Server
    }));
    let transport_noise = noise::Config::new(&key)?;
    let mut swarm = SwarmBuilder::with_existing_identity(key)
        .with_tokio()
        .with_other_transport(|_| {
            crate::mesh_transport::GuardedTcp::new(c.relay.as_ref(), c.force_relay)
                .upgrade(libp2p::core::upgrade::Version::V1Lazy)
                .authenticate(transport_noise)
                .multiplex(multiplexer())
                .boxed()
        })?
        .with_relay_client(noise::Config::new, multiplexer)?
        .with_behaviour(|key, relay_client| {
            let rc = relay::Config {
                max_reservations: 64,
                max_reservations_per_peer: 1,
                reservation_duration: Duration::from_secs(120),
                // A client holds one circuit per worker on its route, and routes over many
                // small devices can have up to MAX_SHARDS stages.
                max_circuits: 256,
                max_circuits_per_peer: 64,
                // A circuit carries whole inference sessions. Closing it mid-request fails
                // that request (there is no safe replay of a KV-cache step), so circuits must
                // outlive normal sessions; admitted-member and per-peer limits still apply.
                max_circuit_duration: Duration::from_secs(60 * 60),
                max_circuit_bytes: 1024 * 1024 * 1024,
                // libp2p's defaults refill one reservation token per minute per IP and one per
                // two minutes per peer. Members renew every ~90 s, so peers behind one NAT
                // (measured: 21 container peers seen as one address) exhausted the IP bucket in
                // three rounds, and any peer would exhaust its own bucket in about three hours.
                // Only admitted members reach the relay, so the per-IP limits are generous.
                reservation_rate_limiters: Vec::new(),
                circuit_src_rate_limiters: Vec::new(),
            }
            .reservation_rate_per_peer(NonZeroU32::new(30).unwrap(), Duration::from_secs(30))
            .reservation_rate_per_ip(NonZeroU32::new(256).unwrap(), Duration::from_secs(1))
            .circuit_src_per_peer(NonZeroU32::new(64).unwrap(), Duration::from_secs(2))
            .circuit_src_per_ip(NonZeroU32::new(256).unwrap(), Duration::from_secs(1));
            let codec = rr::cbor::codec::Codec::default()
                .set_request_size_maximum((LIMIT + 1024) as u64)
                .set_response_size_maximum((LIMIT + 1024) as u64);
            Behaviour {
                kad,
                allow: Default::default(),
                limits: libp2p::connection_limits::Behaviour::new(
                    libp2p::connection_limits::ConnectionLimits::default()
                        .with_max_pending_incoming(Some(16))
                        .with_max_pending_outgoing(Some(16))
                        .with_max_established(Some(64))
                        .with_max_established_per_peer(Some(4)),
                ),
                relay_client,
                relay_server: c.relay_server.then(|| relay::Behaviour::new(me, rc)).into(),
                // A relay-only node's listen addresses (e.g. 127.0.0.1:9000) are useless to others
                // and, cached by identify, replaced the circuit address a peer must be dialled at.
                identify: identify::Behaviour::new(
                    identify::Config::new("/sangama/mesh/1".into(), key.public())
                        .with_hide_listen_addrs(c.force_relay),
                ),
                dcutr: (!c.force_relay).then(|| dcutr::Behaviour::new(me)).into(),
                autonat: (!c.force_relay)
                    .then(|| autonat::Behaviour::new(me, Default::default()))
                    .into(),
                rpc: rr::Behaviour::with_codec(
                    codec,
                    [(
                        StreamProtocol::new("/sangama/qwen/1"),
                        rr::ProtocolSupport::Full,
                    )],
                    rr::Config::default()
                        .with_request_timeout(Duration::from_secs(60))
                        .with_max_concurrent_streams(STREAMS_PER_CONNECTION),
                ),
            }
        })?
        .with_swarm_config(|s| {
            // Send the protocol proposal with the first request data instead of waiting a round
            // trip for confirmation; every request opens a new substream.
            s.with_idle_connection_timeout(Duration::from_secs(180))
                .with_substream_upgrade_protocol_override(libp2p::core::upgrade::Version::V1Lazy)
        })
        .build();
    let mut allowed = HashSet::new();
    for m in &membership.snapshot.members {
        if m.expires > now() {
            let p: PeerId = m.peer.parse()?;
            swarm.behaviour_mut().allow.allow_peer(p);
            allowed.insert(p);
        }
    }
    swarm.listen_on(c.listen.clone())?;
    for a in &c.external {
        swarm.add_external_address(a.clone());
    }
    let mut relay_listener = None;
    let mut relay_retry = Instant::now();
    let relay_peer = if let Some(address) = &c.relay {
        let Some(libp2p::multiaddr::Protocol::P2p(peer)) = address.iter().last() else {
            anyhow::bail!("relay address must end with pinned /p2p/peer-id")
        };
        ensure!(
            membership
                .snapshot
                .members
                .iter()
                .any(|m| m.peer == peer.to_string() && m.role == "relay" && m.expires > now()),
            "relay lacks membership"
        );
        swarm
            .behaviour_mut()
            .kad
            .add_address(&peer, address.clone());
        relay_listener = Some(
            swarm.listen_on(
                address
                    .clone()
                    .with(libp2p::multiaddr::Protocol::P2pCircuit),
            )?,
        );
        for b in &c.bridges {
            let target: PeerId = b.peer.parse()?;
            swarm.add_peer_address(
                target,
                address
                    .clone()
                    .with(libp2p::multiaddr::Protocol::P2pCircuit)
                    .with(libp2p::multiaddr::Protocol::P2p(target)),
            );
        }
        Some(peer)
    } else {
        None
    };
    let (tx, mut rx) = mpsc::channel(PENDING_CALLS);
    let bridges: Arc<HashMap<SocketAddr, PeerId>> = Arc::new(
        c.bridges
            .iter()
            .map(|b| Ok((b.listen, b.peer.parse()?)))
            .collect::<Result<_>>()?,
    );
    for b in &c.bridges {
        let listener = tokio::net::TcpListener::bind(b.listen).await?;
        let proxy = Proxy {
            tx: tx.clone(),
            peer: b.peer.parse()?,
            token: token.clone(),
            slots: Arc::new(Semaphore::new(BRIDGE_CONCURRENCY)),
            meter: meter.clone(),
            bridges: bridges.clone(),
        };
        let app = Router::new()
            .fallback(any(proxy_handler))
            .layer(DefaultBodyLimit::max(LIMIT))
            .with_state(proxy);
        tokio::spawn(async move {
            if let Err(e) = axum::serve(crate::server::nodelay(listener), app).await {
                tracing::error!(%e,"bridge stopped");
            }
        });
    }
    {
        // Receipts are signed once a session has finished and sent in the background.
        let meter = meter.clone();
        let outbox = outbox.clone();
        let key = signing_key.clone();
        let receipt_http = http.clone();
        let portal = c.portal.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let receipts = meter.lock().unwrap().drain(Instant::now(), false);
                let mut outbox = outbox.lock().await;
                crate::credits::sign(receipts, &key, &mut outbox);
                outbox.flush(&receipt_http, &portal).await;
            }
        });
        let tx = tx.clone();
        let http = http.clone();
        let portal = c.portal.clone();
        let network = c.network.clone();
        let authority = authority.clone();
        tokio::spawn(async move {
            loop {
                // A failed fetch keeps the last standing until it expires.
                if let Ok(s) = crate::credits::standing(&http, &portal, &authority, &network).await
                    && tx.send(Command::Standing(s)).await.is_err()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        });
    }
    let refresh_tx = tx.clone();
    let refresh_http = http.clone();
    let portal = c.portal.clone();
    let network = c.network.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            if let Ok(s) = fetch(&refresh_http, &portal, &authority, &network).await
                && refresh_tx.send(Command::Membership(s)).await.is_err()
            {
                break;
            }
        }
    });
    if let Some(address) = c.worker {
        let tx = tx.clone();
        let token = token.clone();
        tokio::spawn(async move {
            let http = reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap();
            loop {
                if let Ok(r) = http
                    .get(format!("http://{address}/v1/qwen/info"))
                    .bearer_auth(&token)
                    .send()
                    .await
                    && let Ok(info) = r.json::<crate::qwen::network::Info>().await
                    && tx.send(Command::Offer(info)).await.is_err()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(20)).await;
            }
        });
    }
    let worker_http = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(55))
        .build()?;
    let slots = Arc::new(Semaphore::new(SERVE_CONCURRENCY));
    let owners = Arc::new(std::sync::Mutex::new(crate::mesh_owner::Owners::new(
        c.bridges
            .iter()
            .map(|b| Ok((b.listen, b.peer.parse()?)))
            .collect::<Result<_>>()?,
    )));
    let mut pending: HashMap<rr::OutboundRequestId, (PeerId, oneshot::Sender<Reply>)> =
        HashMap::new();
    let mut quota: HashMap<PeerId, (Instant, usize, usize)> = HashMap::new();
    let mut ad_quota: HashMap<PeerId, (Instant, usize)> = HashMap::new();
    let mut offers: HashMap<PeerId, crate::mesh_store::Offer> = HashMap::new();
    let mut standing: Option<SignedStanding> = None;
    let mut connections: HashMap<libp2p::swarm::ConnectionId, (PeerId, bool)> = HashMap::new();
    let mut search_tick = tokio::time::interval(Duration::from_secs(10));
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    let shutdown = shutdown();
    tokio::pin!(shutdown);
    tracing::info!(peer=%me,role=own_role,"mesh started");
    loop {
        tokio::select! {
            _=&mut shutdown=>break,
            _=search_tick.tick()=>{
                if membership.snapshot.expires>now() && allowed.contains(&me) {
                    for m in membership.snapshot.members.iter().filter(|m|m.role=="worker" && m.expires>now()).take(32) {
                        let p:PeerId=m.peer.parse()?;
                        if let Some(relay)=&c.relay
                            && p!=me {swarm.add_peer_address(p,relay.clone().with(libp2p::multiaddr::Protocol::P2pCircuit).with(libp2p::multiaddr::Protocol::P2p(p)));}
                        swarm.behaviour_mut().kad.get_record(libp2p::kad::RecordKey::new(&format!("/sangama/offer/{p}")));
                    }
                }
            },
            _=tick.tick()=>{
                let active=membership.snapshot.expires>now() && membership.snapshot.members.iter().any(|m|m.peer==me.to_string() && m.role==own_role && m.expires>now());
                let next:HashSet<PeerId>=if active {membership.snapshot.members.iter().filter(|m|m.expires>now()).filter_map(|m|m.peer.parse().ok()).collect()} else {HashSet::new()};
                for p in allowed.difference(&next) {swarm.behaviour_mut().allow.disallow_peer(*p);let _=swarm.disconnect_peer_id(*p);tracing::warn!(peer=%p,"membership expired or revoked; disconnected");}
                for p in next.difference(&allowed) {swarm.behaviour_mut().allow.allow_peer(*p);}
                allowed=next;
                if active && relay_listener.is_none() && relay_retry.elapsed()>Duration::from_secs(5)
                    && let Some(address)=&c.relay && relay_peer.is_some_and(|p|allowed.contains(&p)) {
                        relay_retry=Instant::now();
                        relay_listener=swarm.listen_on(address.clone().with(libp2p::multiaddr::Protocol::P2pCircuit)).ok();
                }
                pending.retain(|_,(p,result)| {if !allowed.contains(p) {false} else {!result.is_closed()}});
                quota.retain(|p,_|allowed.contains(p));
                ad_quota.retain(|p,_|allowed.contains(p));
                offers.retain(|p,o|allowed.contains(p) && o.expires>now());
            },
            Some(command)=rx.recv()=>match command {
                Command::Status(peer,result)=>{
                    let member=membership.snapshot.members.iter().find(|m|m.peer==peer.to_string());
                    let paths: Vec<&str> = connections.values().filter(|(p,_)|*p==peer).map(|(_,relay)|if *relay {"relay"} else {"direct"}).collect();
                    let body=serde_json::to_vec(&serde_json::json!({"peer":peer.to_string(),"admitted":allowed.contains(&me) && allowed.contains(&peer) && membership.snapshot.expires>now() && member.is_some_and(|m|m.expires>now()),"member_expires":member.map(|m|m.expires),"snapshot_expires":membership.snapshot.expires,"paths":paths}))?;
                    let _=result.send(Reply {status:200,body});
                },
                Command::Discover(result)=>{
                    let body=serde_json::to_vec(&offers.values().collect::<Vec<_>>())?;
                    let _=result.send(if membership.snapshot.expires>now() && allowed.contains(&me) {Reply {status:200,body}} else {Reply::error(403,"Membership expired")});
                },
                Command::Offer(info)=>{
                    if allowed.contains(&me) && membership.snapshot.expires>now()
                        && let Ok(o)=crate::mesh_store::Offer::new(c.network.clone(),info,&signing_key)
                            && let Ok(r)=o.record() {offers.insert(me,o);let _=swarm.behaviour_mut().kad.put_record(r,libp2p::kad::Quorum::One);}
                },
                Command::Membership(s)=> {if s.snapshot.issued>=membership.snapshot.issued {membership=s;}},
                Command::Standing(s)=>standing=s,
                Command::Call {peer,request,result}=>{
                    if membership.snapshot.expires<=now() || !allowed.contains(&peer) || !allowed.contains(&me) || pending.len()>=PENDING_CALLS {let _=result.send(Reply::error(403,"Membership unavailable or capacity exceeded"));continue;}
                    // Dial the relay circuit ourselves: cached addresses can go stale when a peer
                    // restarts, and a failed dial otherwise never tries the circuit.
                    if !swarm.is_connected(&peer) && let Some(relay)=&c.relay {
                        let circuit=relay.clone().with(libp2p::multiaddr::Protocol::P2pCircuit).with(libp2p::multiaddr::Protocol::P2p(peer));
                        let dial=libp2p::swarm::dial_opts::DialOpts::peer_id(peer).addresses(vec![circuit]).condition(libp2p::swarm::dial_opts::PeerCondition::DisconnectedAndNotDialing);
                        let _=swarm.dial(if c.force_relay {dial.build()} else {dial.extend_addresses_through_behaviour().build()});
                    }
                    let id=swarm.behaviour_mut().rpc.send_request(&peer,request);pending.insert(id,(peer,result));
                },
                Command::Reply {peer,channel,mut reply}=>{
                    if let Some(q)=quota.get_mut(&peer) {q.2=q.2.saturating_add(reply.body.len());if q.2>PEER_BYTES_PER_MINUTE {reply=Reply::error(429,"Peer byte quota exceeded");}}
                    let reply=if membership.snapshot.expires>now() && allowed.contains(&peer) && allowed.contains(&me) {reply} else {Reply::error(403,"Membership expired")};let _=swarm.behaviour_mut().rpc.send_response(channel,reply);},
            },
            event=swarm.select_next_some()=>match event {
                SwarmEvent::ListenerClosed {listener_id,..}=>{if relay_listener==Some(listener_id) {relay_listener=None;relay_retry=Instant::now();}},
                SwarmEvent::NewListenAddr {address,..}=>tracing::info!(%address,"mesh listening"),
                SwarmEvent::ConnectionEstablished {peer_id,connection_id,endpoint,..}=>{connections.insert(connection_id,(peer_id,endpoint.is_relayed()));tracing::info!(peer=%peer_id,relayed=endpoint.is_relayed(),"peer connected");},
                SwarmEvent::ConnectionClosed {connection_id,..}=>{connections.remove(&connection_id);},
                SwarmEvent::Behaviour(BehaviourEvent::Identify(identify::Event::Received {peer_id,info,..}))=>{
                    // Only admitted identities influence dial addresses. Forced-relay tests suppress direct upgrades.
                    if allowed.contains(&peer_id) && !c.force_relay {
                        for address in info.listen_addrs.into_iter().take(8) {if safe_direct(&address) {swarm.add_peer_address(peer_id,address);}}
                        if Some(peer_id)==relay_peer {swarm.add_external_address(info.observed_addr);}
                    }
                },
                SwarmEvent::Behaviour(BehaviourEvent::Kad(libp2p::kad::Event::OutboundQueryProgressed {result:libp2p::kad::QueryResult::GetRecord(Ok(libp2p::kad::GetRecordOk::FoundRecord(record))),..}))=>{
                    if membership.snapshot.expires>now()
                        && let Ok(o)=crate::mesh_store::Offer::decode(&record.record,&c.network)
                            && let Ok(p)=o.peer.parse::<PeerId>()
                                && allowed.contains(&p) && membership.snapshot.members.iter().any(|m|m.peer==o.peer && m.role=="worker") {offers.insert(p,o);}
                },
                SwarmEvent::Behaviour(BehaviourEvent::Kad(libp2p::kad::Event::InboundRequest {request:libp2p::kad::InboundRequest::PutRecord {source,record:Some(record),..}}))=>{
                    use libp2p::kad::store::RecordStore;
                    let q=ad_quota.entry(source).or_insert((Instant::now(),0));if q.0.elapsed()>Duration::from_secs(60) {*q=(Instant::now(),0);}q.1+=1;
                    if q.1<=120 && allowed.contains(&source) && membership.snapshot.expires>now()
                        && let Ok(o)=crate::mesh_store::Offer::decode(&record,&c.network)
                            && o.peer.parse::<PeerId>().is_ok_and(|p|allowed.contains(&p)) && membership.snapshot.members.iter().any(|m|m.peer==o.peer && m.role=="worker") {let _=swarm.behaviour_mut().kad.store_mut().put(record);}
                },
                SwarmEvent::Behaviour(BehaviourEvent::Rpc(rr::Event::Message {peer,message,..}))=>match message {
                    rr::Message::Request {request,channel,..}=>{
                        let role=membership.snapshot.members.iter().find(|m|m.peer==peer.to_string() && ["client","worker"].contains(&m.role.as_str()) && m.expires>now()).map(|m|m.role.clone());
                        let Some(role)=role else {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(403,"Request denied"));continue;};
                        if membership.snapshot.expires<=now() || !allowed.contains(&peer) || !allowed.contains(&me) || !valid_path(&request.path) || request.body.len()>LIMIT {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(403,"Request denied"));continue;}
                        let q=quota.entry(peer).or_insert((Instant::now(),0,0));if q.0.elapsed()>Duration::from_secs(60) {*q=(Instant::now(),0,0);}
                        q.1+=1;q.2+=request.body.len();
                        if q.1>PEER_REQUESTS_PER_MINUTE || q.2>PEER_BYTES_PER_MINUTE {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(429,"Peer quota exceeded"));continue;}
                        if request.path=="/v1/qwen/reserve" && crate::credits::over_allowance(standing.as_ref(),&peer) {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(402,"Credit allowance used up: contribute a worker or wait for credits"));continue;}
                        // The local worker trusts one shared token; bind reservations to the verified peer here.
                        let pending=match owners.lock().unwrap().admit(peer,&role,&request.path,&request.body,Instant::now()) {
                            Ok(pending)=>pending,
                            Err(reason)=>{let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(403,reason));continue;}
                        };
                        // Forwards are billed to the client that reserved the session, for this node's layers.
                        let billed=match (&pending,request.path.as_str()) {
                            (Some(p),"/v1/qwen/forward")=>crate::qwen::wire::Frame::header(&request.body).ok().map(|h|(p.owner(),h)),
                            _=>None,
                        };
                        let permit=slots.clone().try_acquire_owned();
                        if let (Some(address),Ok(permit))=(c.worker,permit) {
                            let tx=tx.clone();let http=worker_http.clone();let token=token.clone();let managed=managed.clone();let owners=owners.clone();let meter=meter.clone();
                            tokio::spawn(async move {
                                let _permit=permit;
                                let reply=if request.path.starts_with("/v1/node/") {
                                    if let Some(manager)=managed {
                                        match manager.handle(&request.path,&request.body).await {
                                            Ok(value)=>Reply {status:200,body:serde_json::to_vec(&value).unwrap_or_default()},
                                            Err(error)=>Reply::error(409,&error.to_string()),
                                        }
                                    } else {Reply::error(503,"Worker does not accept managed placement")}
                                } else if let Some(manager)=managed {
                                    if manager.ready().await {worker(http,address,token,request).await} else {Reply::error(503,"Worker is not loaded or has an active placement reservation")}
                                } else {worker(http,address,token,request).await};
                                if let Some(pending)=pending {owners.lock().unwrap().commit(pending,reply.status,Instant::now());}
                                if let Some((consumer,sent))=billed && (200..300).contains(&reply.status)
                                    && let Ok(output)=crate::qwen::wire::Frame::header(&reply.body)
                                    // The trace entry at this node's position was written by its own worker.
                                    && let Some(own)=output.trace.get(sent.trace.len()) {
                                    meter.lock().unwrap().record_work(&sent.session,consumer,&sent.model_hash,(own.start as u32,own.end as u32),sent.seq_len as u64,Instant::now());
                                }
                                let _=tx.send(Command::Reply {peer,channel,reply}).await;
                            });
                        } else {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(503,"No available worker"));}
                    },
                    rr::Message::Response {request_id,response}=>{if let Some((p,result))=pending.remove(&request_id) {let _=result.send(if allowed.contains(&p) && membership.snapshot.expires>now() {response} else {Reply::error(403,"Membership expired")});}},
                },
                SwarmEvent::Behaviour(BehaviourEvent::Rpc(rr::Event::OutboundFailure {request_id,error,..}))=>{if let Some((_,result))=pending.remove(&request_id) {let _=result.send(Reply::error(503,"Peer disconnected or unavailable"));}tracing::warn!(%error,"peer request failed");},
                SwarmEvent::Behaviour(BehaviourEvent::RelayServer(event))=>tracing::debug!(?event,"relay event"),
                SwarmEvent::Behaviour(BehaviourEvent::RelayClient(event))=>tracing::info!(?event,"relay client event"),
                SwarmEvent::Behaviour(BehaviourEvent::Dcutr(event))=>tracing::info!(?event,"direct upgrade"),
                _=>{},
            }
        }
    }
    // Report every metered session, finished or not, so a restart does not forfeit earned credits.
    let receipts = meter.lock().unwrap().drain(Instant::now(), true);
    let mut outbox = outbox.lock().await;
    crate::credits::sign(receipts, &signing_key, &mut outbox);
    if tokio::time::timeout(Duration::from_secs(4), outbox.flush(&http, &c.portal))
        .await
        .is_err()
    {
        tracing::warn!("portal unreachable; unsent credit receipts were dropped at shutdown");
    }
    Ok(())
}
/// Completes on Ctrl-C or, on Unix, SIGTERM (how the node launcher stops its children).
async fn shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = term.recv() => {},
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
// Reject loopback/link-local advertisements: otherwise an admitted peer could cause local dialing.
fn safe_direct(a: &Multiaddr) -> bool {
    crate::mesh_transport::public(a)
}
async fn proxy_handler(s: State<Proxy>, m: Method, u: Uri, h: HeaderMap, b: Bytes) -> Response {
    proxy(s, m, u, h, b).await
}
