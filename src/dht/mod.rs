pub mod record;
pub mod store;

use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, SwarmBuilder, identify,
    identity::Keypair,
    kad::{self, store::RecordStore},
    multiaddr::Protocol,
    noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
use record::{Advertisement, Signed};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use store::SqliteStore;
use tokio::sync::{RwLock, mpsc, oneshot};

#[derive(NetworkBehaviour)]
struct Behaviour {
    kad: kad::Behaviour<SqliteStore>,
    identify: identify::Behaviour,
    limits: libp2p::connection_limits::Behaviour,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Offer {
    pub model_hash: String,
    pub start: usize,
    pub end: usize,
}
#[derive(Clone, Debug, Serialize, Default)]
pub struct Snapshot {
    pub peer_id: String,
    pub address: String,
    pub connected_peers: usize,
    pub stored_records: usize,
    pub offer: Option<Offer>,
    pub published: bool,
    pub search: String,
    pub discoveries: Vec<Advertisement>,
    pub last_error: Option<String>,
}
#[derive(Clone)]
pub struct Handle {
    tx: mpsc::Sender<Command>,
    state: Arc<RwLock<Snapshot>>,
}
pub struct Node {
    pub handle: Handle,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Node {
    fn drop(&mut self) {
        self.task.abort();
    }
}
enum Command {
    Join(Multiaddr),
    Publish(Offer),
    Find(String),
}
impl Handle {
    pub async fn snapshot(&self) -> Snapshot {
        let mut snapshot = self.state.read().await.clone();
        snapshot.discoveries.retain(|p| p.expires > record::now());
        snapshot
    }
    pub async fn join(&self, value: &str) -> Result<()> {
        let addr = value.parse()?;
        bootstrap(&addr)?;
        self.tx
            .try_send(Command::Join(addr))
            .context("DHT command queue unavailable")?;
        Ok(())
    }
    pub async fn publish(&self, offer: Offer) -> Result<()> {
        record::model_key(&offer.model_hash)?;
        ensure!(
            offer.start < offer.end && offer.end <= 24,
            "invalid shard range"
        );
        self.tx
            .try_send(Command::Publish(offer))
            .context("DHT command queue unavailable")?;
        Ok(())
    }
    pub async fn find(&self, hash: String) -> Result<()> {
        record::model_key(&hash)?;
        self.tx
            .try_send(Command::Find(hash))
            .context("DHT command queue unavailable")?;
        Ok(())
    }
}
fn bootstrap(value: &Multiaddr) -> Result<(PeerId, Multiaddr)> {
    let mut address = value.clone();
    let Some(Protocol::P2p(peer)) = address.pop() else {
        anyhow::bail!("bootstrap must end with /p2p/PEER_ID");
    };
    record::address(&address, false)?;
    Ok((peer, address))
}
pub(crate) fn identity(dir: &Path) -> Result<(Keypair, File)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        if !dir.exists() {
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir)?;
        }
        ensure!(
            !dir.symlink_metadata()?.file_type().is_symlink()
                && dir.metadata()?.permissions().mode() & 0o077 == 0,
            "DHT state directory must be private (chmod 700) and not a symlink"
        );
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(dir)?;
    let mut lock_options = OpenOptions::new();
    lock_options
        .create(true)
        .truncate(false)
        .read(true)
        .write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        lock_options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let lock = lock_options.open(dir.join("node.lock"))?;
    lock.try_lock_exclusive()
        .context("DHT state directory is already in use")?;
    let path = dir.join("identity.key");
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let key = match options.open(&path) {
        Ok(file) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                ensure!(
                    file.metadata()?.permissions().mode() & 0o077 == 0,
                    "identity key must have mode 600"
                );
            }
            ensure!(
                file.metadata()?.is_file() && file.metadata()?.len() <= 4096,
                "invalid identity file"
            );
            let mut bytes = vec![];
            file.take(4097).read_to_end(&mut bytes)?;
            Keypair::from_protobuf_encoding(&bytes)?
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let key = Keypair::generate_ed25519();
            let mut options = OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&path)?;
            file.write_all(&key.to_protobuf_encoding()?)?;
            file.sync_all()?;
            key
        }
        Err(e) => return Err(e.into()),
    };
    for file in [
        "discovery.sqlite",
        "discovery.sqlite-wal",
        "discovery.sqlite-shm",
    ] {
        let p = dir.join(file);
        if let Ok(meta) = p.symlink_metadata() {
            ensure!(
                !meta.file_type().is_symlink(),
                "SQLite files must not be symlinks"
            );
        }
    }
    Ok((key, lock))
}

pub async fn start(dir: PathBuf, listen: Multiaddr, seeds: Vec<Multiaddr>) -> Result<Node> {
    record::address(&listen, true)?;
    ensure!(seeds.len() <= 16, "at most 16 bootstrap nodes");
    for seed in &seeds {
        bootstrap(seed)?;
    }
    let (key, lock) = identity(&dir)?;
    let peer = key.public().to_peer_id();
    let store = SqliteStore::open(&dir.join("discovery.sqlite"), peer)?;
    let mut config = kad::Config::new(StreamProtocol::new("/mesh/kad/1.0.0"));
    config
        .set_query_timeout(Duration::from_secs(12))
        .set_record_ttl(Some(Duration::from_secs(record::TTL)))
        .set_provider_record_ttl(Some(Duration::from_secs(record::TTL)))
        .set_publication_interval(None)
        .set_provider_publication_interval(Some(Duration::from_secs(120)))
        .set_max_packet_size(32 * 1024);
    let mut kad = kad::Behaviour::with_config(peer, store, config);
    kad.set_mode(Some(kad::Mode::Server));
    let behaviour = Behaviour {
        kad,
        identify: identify::Behaviour::new(identify::Config::new(
            "/mesh/identify/1".into(),
            key.public(),
        )),
        limits: libp2p::connection_limits::Behaviour::new(
            libp2p::connection_limits::ConnectionLimits::default()
                .with_max_pending_incoming(Some(16))
                .with_max_pending_outgoing(Some(16))
                .with_max_established(Some(64))
                .with_max_established_per_peer(Some(2)),
        ),
    };
    let mut swarm = SwarmBuilder::with_existing_identity(key.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|_| behaviour)?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(90)))
        .build();
    swarm.listen_on(listen)?;
    for seed in &seeds {
        let (p, a) = bootstrap(seed)?;
        swarm.behaviour_mut().kad.add_address(&p, a);
        swarm.dial(seed.clone())?;
    }
    let state = Arc::new(RwLock::new(Snapshot {
        peer_id: peer.to_string(),
        search: "idle".into(),
        ..Default::default()
    }));
    let (tx, mut rx) = mpsc::channel(16);
    let handle = Handle {
        tx,
        state: state.clone(),
    };
    let (ready_tx, ready_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let _lock = lock;
        let mut ready = Some(ready_tx);
        let mut address = None;
        let mut offer: Option<Offer> = None;
        let mut tick = tokio::time::interval(Duration::from_secs(30));
        let mut search_model = String::new();
        let mut queries: HashMap<kad::QueryId, PeerId> = HashMap::new();
        let mut providers_query = None;
        let mut seen = HashSet::new();
        loop {
            tokio::select! {
                _=tick.tick()=>{
                    if let (Some(offer),Some(address))=(&offer,&address)
                        && let Err(error)=publish(&mut swarm,&key,offer,address){state.write().await.last_error=Some(error.to_string());}
                    let mut s=state.write().await;s.discoveries.retain(|p|p.expires>record::now());
                }
                command=rx.recv()=>match command {
                    Some(Command::Join(seed))=>{if let Ok((p,a))=bootstrap(&seed){swarm.behaviour_mut().kad.add_address(&p,a);if let Err(error)=swarm.dial(seed){state.write().await.last_error=Some(error.to_string());}}}
                    Some(Command::Publish(value))=>{
                        if let Some(old)=&offer&& old.model_hash!=value.model_hash {swarm.behaviour_mut().kad.stop_providing(&kad::RecordKey::new(&record::model_key(&old.model_hash).unwrap()));}
                        offer=Some(value.clone());let mut s=state.write().await;s.offer=Some(value.clone());s.published=false;s.last_error=None;drop(s);
                        if let Some(address)=&address&& let Err(error)=publish(&mut swarm,&key,&value,address){state.write().await.last_error=Some(error.to_string());}
                    }
                    Some(Command::Find(hash))=>{
                        // Abort old searches so repeated UI requests cannot create unlimited queries.
                        if let Some(id)=providers_query.take()&& let Some(mut q)=swarm.behaviour_mut().kad.query_mut(&id){q.finish();}
                        for (id,_) in queries.drain(){if let Some(mut q)=swarm.behaviour_mut().kad.query_mut(&id){q.finish();}}
                        search_model=hash;seen.clear();
                        providers_query=Some(swarm.behaviour_mut().kad.get_providers(kad::RecordKey::new(&record::model_key(&search_model).unwrap())));
                        let mut s=state.write().await;s.search="searching".into();s.discoveries.clear();s.last_error=None;
                    }
                    None=>break,
                },
                event=swarm.select_next_some()=>{
                    match event {
                        SwarmEvent::NewListenAddr{address:a,..}=>{
                            swarm.add_external_address(a.clone());address=Some(a.clone());
                            state.write().await.address=format!("{a}/p2p/{peer}");
                            if let Some(ready)=ready.take(){let _=ready.send(());}
                        }
                        SwarmEvent::ConnectionEstablished{..}=>{
                            let _=swarm.behaviour_mut().kad.bootstrap();
                        }
                        SwarmEvent::Behaviour(BehaviourEvent::Identify(identify::Event::Received{peer_id,info,..}))=>{
                            if info.protocol_version=="/mesh/identify/1" {for a in info.listen_addrs.into_iter().take(4){if record::address(&a,false).is_ok(){swarm.behaviour_mut().kad.add_address(&peer_id,a);}}}
                            if let (Some(offer),Some(address))=(&offer,&address){let _=publish(&mut swarm,&key,offer,address);}
                        }
                        SwarmEvent::Behaviour(BehaviourEvent::Kad(kad::Event::OutboundQueryProgressed{id,result,step,..}))=>{
                            match result {
                                kad::QueryResult::PutRecord(result)=>{let mut s=state.write().await;s.published=result.is_ok();if let Err(error)=result{s.last_error=Some(error.to_string());}else{s.last_error=None;}}
                                kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders{providers,..})) if Some(id)==providers_query=>{
                                    for p in providers {if seen.len()<64 && seen.insert(p){let q=swarm.behaviour_mut().kad.get_record(kad::RecordKey::new(&format!("{}{p}",record::PREFIX)));queries.insert(q,p);}}
                                }
                                kad::QueryResult::GetProviders(Err(error)) if Some(id)==providers_query=>{state.write().await.last_error=Some(error.to_string());}
                                kad::QueryResult::GetRecord(Ok(kad::GetRecordOk::FoundRecord(found))) if queries.contains_key(&id)=>{
                                    if let Ok(signed)=Signed::decode(found.record.key.as_ref(),&found.record.value)
                                        && signed.payload.model_hash==search_model && Some(&signed.payload.peer_id)==queries.get(&id).map(|p|p.to_string()).as_ref(){
                                            let _=swarm.behaviour_mut().kad.store_mut().put(found.record);
                                            let mut s=state.write().await;s.discoveries.retain(|p|p.peer_id!=signed.payload.peer_id);s.discoveries.push(signed.payload);
                                        }
                                }
                                kad::QueryResult::GetRecord(Err(error)) if queries.contains_key(&id)=>{state.write().await.last_error=Some(error.to_string());}
                                _=>{}
                            }
                            if step.last {if Some(id)==providers_query{providers_query=None;}queries.remove(&id);}
                            if providers_query.is_none() && queries.is_empty() && !search_model.is_empty(){state.write().await.search="complete".into();}
                        }
                        SwarmEvent::OutgoingConnectionError{error,..}=>{state.write().await.last_error=Some(error.to_string());}
                        SwarmEvent::ListenerError{error,..}=>{state.write().await.last_error=Some(error.to_string());}
                        _=>{}
                    }
                    let mut s=state.write().await;s.connected_peers=swarm.connected_peers().count();s.stored_records=swarm.behaviour_mut().kad.store_mut().records().count();
                }
            }
        }
    });
    let node = Node { handle, task };
    tokio::time::timeout(Duration::from_secs(10), ready_rx)
        .await
        .context("DHT listener startup timed out")??;
    Ok(node)
}
fn publish(
    swarm: &mut libp2p::Swarm<Behaviour>,
    key: &Keypair,
    offer: &Offer,
    address: &Multiaddr,
) -> Result<()> {
    let signed = Signed::new(
        key,
        offer.model_hash.clone(),
        offer.start,
        offer.end,
        vec![address.to_string()],
    )?;
    let record = kad::Record {
        key: kad::RecordKey::new(&format!("{}{}", record::PREFIX, signed.payload.peer_id)),
        value: serde_json::to_vec(&signed)?,
        publisher: Some(key.public().to_peer_id()),
        expires: Some(std::time::Instant::now() + Duration::from_secs(record::TTL)),
    };
    swarm
        .behaviour_mut()
        .kad
        .put_record(record, kad::Quorum::One)?;
    swarm
        .behaviour_mut()
        .kad
        .start_providing(kad::RecordKey::new(&record::model_key(&offer.model_hash)?))?;
    Ok(())
}
