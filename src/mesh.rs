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
use sangama_network_auth::{Join, SignedSnapshot, libp2p_identity::PublicKey, now, proof};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Semaphore, mpsc, oneshot};
const LIMIT: usize = crate::qwen::wire::MAX_FRAME_BYTES;

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
    Discover(oneshot::Sender<Reply>),
}
#[derive(Clone)]
struct Proxy {
    tx: mpsc::Sender<Command>,
    peer: PeerId,
    token: String,
    slots: Arc<Semaphore>,
}
fn valid_path(path: &str) -> bool {
    [
        "/v1/qwen/info",
        "/v1/qwen/forward",
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
    if path == "/v1/mesh/offers" && method == Method::GET && uri.query().is_none() {
        let (tx, rx) = oneshot::channel();
        if s.tx.try_send(Command::Discover(tx)).is_err() {
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
        Ok(Ok(reply)) => (
            StatusCode::from_u16(reply.status).unwrap_or(StatusCode::BAD_GATEWAY),
            [(
                "content-type",
                if path.ends_with("forward") {
                    "application/octet-stream"
                } else {
                    "application/json"
                },
            )],
            reply.body,
        )
            .into_response(),
        _ => StatusCode::GATEWAY_TIMEOUT.into_response(),
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
                    if request.path.ends_with("reset") || request.path.ends_with("reserve") {
                        "application/json"
                    } else {
                        "application/octet-stream"
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
    ensure!(c.bridges.len() <= 8, "too many bridges");
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
    kad.set_mode(Some(libp2p::kad::Mode::Server));
    let transport_noise = noise::Config::new(&key)?;
    let mut swarm = SwarmBuilder::with_existing_identity(key)
        .with_tokio()
        .with_other_transport(|_| {
            crate::mesh_transport::GuardedTcp::new(c.relay.as_ref(), c.force_relay)
                .upgrade(libp2p::core::upgrade::Version::V1Lazy)
                .authenticate(transport_noise)
                .multiplex(yamux::Config::default())
                .boxed()
        })?
        .with_relay_client(noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            let rc = relay::Config {
                max_reservations: 64,
                max_reservations_per_peer: 1,
                reservation_duration: Duration::from_secs(120),
                max_circuits: 32,
                max_circuits_per_peer: 4,
                max_circuit_duration: Duration::from_secs(120),
                max_circuit_bytes: 128 * 1024 * 1024,
                ..Default::default()
            };
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
                identify: identify::Behaviour::new(identify::Config::new(
                    "/sangama/mesh/1".into(),
                    key.public(),
                )),
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
                        .with_max_concurrent_streams(16),
                ),
            }
        })?
        .with_swarm_config(|s| s.with_idle_connection_timeout(Duration::from_secs(180)))
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
    let (tx, mut rx) = mpsc::channel(32);
    for b in &c.bridges {
        let listener = tokio::net::TcpListener::bind(b.listen).await?;
        let proxy = Proxy {
            tx: tx.clone(),
            peer: b.peer.parse()?,
            token: token.clone(),
            slots: Arc::new(Semaphore::new(4)),
        };
        let app = Router::new()
            .fallback(any(proxy_handler))
            .layer(DefaultBodyLimit::max(LIMIT))
            .with_state(proxy);
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!(%e,"bridge stopped");
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
    let slots = Arc::new(Semaphore::new(8));
    let mut pending: HashMap<rr::OutboundRequestId, (PeerId, oneshot::Sender<Reply>)> =
        HashMap::new();
    let mut quota: HashMap<PeerId, (Instant, usize, usize)> = HashMap::new();
    let mut ad_quota: HashMap<PeerId, (Instant, usize)> = HashMap::new();
    let mut offers: HashMap<PeerId, crate::mesh_store::Offer> = HashMap::new();
    let mut search_tick = tokio::time::interval(Duration::from_secs(10));
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tracing::info!(peer=%me,role=own_role,"mesh started");
    loop {
        tokio::select! {
            _=tokio::signal::ctrl_c()=>break,
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
                Command::Call {peer,request,result}=>{
                    if membership.snapshot.expires<=now() || !allowed.contains(&peer) || !allowed.contains(&me) || pending.len()>=16 {let _=result.send(Reply::error(403,"Membership unavailable or capacity exceeded"));continue;}
                    let id=swarm.behaviour_mut().rpc.send_request(&peer,request);pending.insert(id,(peer,result));
                },
                Command::Reply {peer,channel,mut reply}=>{
                    if let Some(q)=quota.get_mut(&peer) {q.2=q.2.saturating_add(reply.body.len());if q.2>128*1024*1024 {reply=Reply::error(429,"Peer byte quota exceeded");}}
                    let reply=if membership.snapshot.expires>now() && allowed.contains(&peer) && allowed.contains(&me) {reply} else {Reply::error(403,"Membership expired")};let _=swarm.behaviour_mut().rpc.send_response(channel,reply);},
            },
            event=swarm.select_next_some()=>match event {
                SwarmEvent::ListenerClosed {listener_id,..}=>{if relay_listener==Some(listener_id) {relay_listener=None;relay_retry=Instant::now();}},
                SwarmEvent::NewListenAddr {address,..}=>tracing::info!(%address,"mesh listening"),
                SwarmEvent::ConnectionEstablished {peer_id,endpoint,..}=>tracing::info!(peer=%peer_id,relayed=endpoint.is_relayed(),"peer connected"),
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
                        let role_ok=membership.snapshot.members.iter().any(|m|m.peer==peer.to_string() && ["client","worker"].contains(&m.role.as_str()) && m.expires>now());
                        if !role_ok || membership.snapshot.expires<=now() || !allowed.contains(&peer) || !allowed.contains(&me) || !valid_path(&request.path) || request.body.len()>LIMIT {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(403,"Request denied"));continue;}
                        let q=quota.entry(peer).or_insert((Instant::now(),0,0));if q.0.elapsed()>Duration::from_secs(60) {*q=(Instant::now(),0,0);}
                        q.1+=1;q.2+=request.body.len();
                        if q.1>1200 || q.2>128*1024*1024 {let _=swarm.behaviour_mut().rpc.send_response(channel,Reply::error(429,"Peer quota exceeded"));continue;}
                        let permit=slots.clone().try_acquire_owned();
                        if let (Some(address),Ok(permit))=(c.worker,permit) {
                            let tx=tx.clone();let http=worker_http.clone();let token=token.clone();let managed=managed.clone();
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
    Ok(())
}
// Reject loopback/link-local advertisements: otherwise an admitted peer could cause local dialing.
fn safe_direct(a: &Multiaddr) -> bool {
    crate::mesh_transport::public(a)
}
async fn proxy_handler(s: State<Proxy>, m: Method, u: Uri, h: HeaderMap, b: Bytes) -> Response {
    proxy(s, m, u, h, b).await
}
