//! Binds worker reservations to the admitted peer that made them. The local worker sees only
//! a shared bearer token, so the mesh is the one place that knows who is asking.
//!
//! - Only `client` members reserve, reset or place; workers may only forward.
//! - A session or placement lease belongs to the peer that reserved it.
//! - A forward is accepted only from the session's declared upstream peer: the client for
//!   the first shard, otherwise the previous shard's worker, resolved through this node's
//!   own bridge map.
//! - A reservation held past its cap is dropped and its owner cools down before reserving
//!   here again, so one member cannot keep an idle worker locked.
use libp2p::PeerId;
use serde::Deserialize;
use std::{
    collections::HashMap,
    net::SocketAddr,
    time::{Duration, Instant},
};

const MAX_CLAIMS: usize = 64;
// Longer than the worker's 60 s session and 120 s lease expiry, so others can claim first.
const COOLDOWN: Duration = Duration::from_secs(180);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Session,
    Lease,
}
struct Claim {
    owner: PeerId,
    upstream: PeerId,
    first: Instant,
    used: Instant,
}
struct Table {
    claims: HashMap<String, Claim>,
    // Match the worker's own expiry: sessions 60 s, placement leases 120 s.
    idle: Duration,
    hold: Duration,
}
impl Table {
    fn new(idle: Duration, hold: Duration) -> Self {
        Self {
            claims: HashMap::new(),
            idle,
            hold,
        }
    }
}
#[derive(Deserialize)]
struct SessionRequest {
    session: String,
    #[serde(default)]
    upstream: Option<SocketAddr>,
}
#[derive(Deserialize)]
struct LeaseRequest {
    lease: String,
}

/// What to record once the local worker has accepted a request.
#[derive(Debug)]
pub struct Pending {
    kind: Kind,
    id: String,
    owner: PeerId,
    upstream: PeerId,
    create: bool,
    release: bool,
}

impl Pending {
    /// The client that reserved the session or lease.
    pub fn owner(&self) -> PeerId {
        self.owner
    }
}

pub struct Owners {
    sessions: Table,
    leases: Table,
    cooldown: HashMap<PeerId, Instant>,
    bridges: HashMap<SocketAddr, PeerId>,
}
impl Owners {
    pub fn new(bridges: HashMap<SocketAddr, PeerId>) -> Self {
        Self {
            sessions: Table::new(Duration::from_secs(60), Duration::from_secs(600)),
            leases: Table::new(Duration::from_secs(120), Duration::from_secs(300)),
            cooldown: HashMap::new(),
            bridges,
        }
    }
    fn table(&mut self, kind: Kind) -> &mut Table {
        match kind {
            Kind::Session => &mut self.sessions,
            Kind::Lease => &mut self.leases,
        }
    }
    fn prune(&mut self, now: Instant) {
        for table in [&mut self.sessions, &mut self.leases] {
            let idle = table.idle;
            table
                .claims
                .retain(|_, c| now.saturating_duration_since(c.used) <= idle);
        }
        self.cooldown.retain(|_, until| now < *until);
    }
    /// Decides whether `peer` may send this request. `Ok(None)` needs no bookkeeping.
    pub fn admit(
        &mut self,
        peer: PeerId,
        role: &str,
        path: &str,
        body: &[u8],
        now: Instant,
    ) -> Result<Option<Pending>, &'static str> {
        self.prune(now);
        match path {
            "/v1/qwen/info" | "/v1/node/capacity" => Ok(None),
            "/v1/qwen/reserve" | "/v1/qwen/reset" => {
                if role != "client" {
                    return Err("only clients reserve or reset workers");
                }
                let r: SessionRequest =
                    serde_json::from_slice(body).map_err(|_| "invalid session request")?;
                let upstream = match r.upstream {
                    Some(address) => *self
                        .bridges
                        .get(&address)
                        .ok_or("upstream is not a known bridge on this node")?,
                    None => peer,
                };
                let reserve = path.ends_with("reserve");
                self.claim(
                    Kind::Session,
                    r.session,
                    peer,
                    upstream,
                    reserve,
                    false,
                    now,
                )
            }
            "/v1/qwen/forward" => {
                let session = crate::qwen::wire::Frame::header(body)
                    .map_err(|_| "invalid frame")?
                    .session;
                let claim = self
                    .sessions
                    .claims
                    .get(&session)
                    .ok_or("session was not reserved through the mesh")?;
                if claim.upstream != peer {
                    return Err("only the upstream peer may forward this session");
                }
                let (owner, upstream) = (claim.owner, claim.upstream);
                self.over_hold(Kind::Session, &session, now)?;
                Ok(Some(Pending {
                    kind: Kind::Session,
                    id: session,
                    owner,
                    upstream,
                    create: false,
                    release: false,
                }))
            }
            "/v1/qwen/result" => {
                let r: SessionRequest =
                    serde_json::from_slice(body).map_err(|_| "invalid session request")?;
                match self.sessions.claims.get(&r.session) {
                    Some(c) if c.owner == peer => Ok(None),
                    _ => Err("only the session's client may collect its result"),
                }
            }
            "/v1/node/reserve" | "/v1/node/load" | "/v1/node/release" => {
                if role != "client" {
                    return Err("only clients place shards");
                }
                let r: LeaseRequest =
                    serde_json::from_slice(body).map_err(|_| "invalid assignment")?;
                let reserve = path.ends_with("reserve");
                let release = path.ends_with("release");
                if !reserve && !self.leases.claims.contains_key(&r.lease) {
                    return Err("placement lease was not reserved through the mesh");
                }
                self.claim(Kind::Lease, r.lease, peer, peer, reserve, release, now)
            }
            _ => Err("unsupported path"),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn claim(
        &mut self,
        kind: Kind,
        id: String,
        peer: PeerId,
        upstream: PeerId,
        create: bool,
        release: bool,
        now: Instant,
    ) -> Result<Option<Pending>, &'static str> {
        let closing = release || (kind == Kind::Session && !create);
        match self.table(kind).claims.get(&id) {
            Some(c) if c.owner != peer => return Err("reserved by another peer"),
            Some(_) if !closing => self.over_hold(kind, &id, now)?,
            Some(_) => {}
            None if kind == Kind::Session && !create => {
                return Err("session was not reserved by this peer");
            }
            None => {
                if self.cooldown.contains_key(&peer) {
                    return Err("peer held a reservation too long; try again later");
                }
                if self.table(kind).claims.len() >= MAX_CLAIMS {
                    return Err("too many reservations");
                }
            }
        }
        Ok(Some(Pending {
            kind,
            id,
            owner: peer,
            upstream,
            create,
            release,
        }))
    }
    /// Ends a reservation held past its cap and starts its owner's cooldown.
    fn over_hold(&mut self, kind: Kind, id: &str, now: Instant) -> Result<(), &'static str> {
        let table = self.table(kind);
        let hold = table.hold;
        if let Some(c) = table.claims.get(id)
            && now.saturating_duration_since(c.first) > hold
        {
            let owner = c.owner;
            table.claims.remove(id);
            self.cooldown.insert(owner, now + COOLDOWN);
            return Err("reservation held too long; try again later");
        }
        Ok(())
    }
    /// Records a request the local worker accepted. Rejected requests change nothing.
    pub fn commit(&mut self, pending: Pending, status: u16, now: Instant) {
        if !(200..300).contains(&status) {
            return;
        }
        let table = self.table(pending.kind);
        if pending.release {
            table.claims.remove(&pending.id);
            return;
        }
        match table.claims.get_mut(&pending.id) {
            Some(c) => {
                c.used = now;
                if pending.create {
                    c.upstream = pending.upstream;
                }
            }
            None if pending.create => {
                table.claims.insert(
                    pending.id,
                    Claim {
                        owner: pending.owner,
                        upstream: pending.upstream,
                        first: now,
                        used: now,
                    },
                );
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::wire::{Frame, Header, Kind as FrameKind};

    fn frame(session: &str) -> Vec<u8> {
        Frame {
            header: Header {
                protocol: 1,
                sample: true,
                detached: false,
                bf16: false,
                speculative: false,
                model_hash: "h".into(),
                session: session.into(),
                position: 0,
                seq_len: 1,
                kind: FrameKind::Hidden,
                tokens: vec![],
                route: vec![],
                trace: vec![],
                inputs: vec![],
                mtp_drafts: 0,
                drafts: vec![],
            },
            values: vec![0.0],
        }
        .encode()
        .unwrap()
    }
    fn session(id: &str, upstream: Option<&str>) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"session":id,"upstream":upstream})).unwrap()
    }
    fn lease(id: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"lease":id,"model_hash":"h","shard":0})).unwrap()
    }
    fn ok(o: &mut Owners, peer: PeerId, role: &str, path: &str, body: &[u8], now: Instant) {
        let pending = o.admit(peer, role, path, body, now).expect(path).unwrap();
        o.commit(pending, 200, now);
    }

    #[test]
    fn sessions_accept_only_owner_resets_and_upstream_forwards() {
        let (client, other, head, tail) = (
            PeerId::random(),
            PeerId::random(),
            PeerId::random(),
            PeerId::random(),
        );
        let bridge: SocketAddr = "127.0.0.1:7901".parse().unwrap();
        let mut o = Owners::new(HashMap::from([(bridge, head)]));
        let t = Instant::now();
        let s = "0b6a3c8e-5f7d-4c1a-9e0b-2d4f6a8c0e1f";
        ok(
            &mut o,
            client,
            "client",
            "/v1/qwen/reserve",
            &session(s, Some("127.0.0.1:7901")),
            t,
        );
        // The tail's session names the head as upstream: the client itself cannot inject.
        assert!(
            o.admit(client, "client", "/v1/qwen/forward", &frame(s), t)
                .is_err()
        );
        assert!(
            o.admit(tail, "worker", "/v1/qwen/forward", &frame(s), t)
                .is_err()
        );
        ok(&mut o, head, "worker", "/v1/qwen/forward", &frame(s), t);
        // Only the session's client collects a detached result.
        let collect = serde_json::to_vec(&serde_json::json!({"session":s,"position":0})).unwrap();
        assert!(
            o.admit(client, "client", "/v1/qwen/result", &collect, t)
                .is_ok()
        );
        for (peer, role) in [(other, "client"), (head, "worker")] {
            assert!(o.admit(peer, role, "/v1/qwen/result", &collect, t).is_err());
        }
        // Another client, or a worker in the route, cannot reset or re-reserve it.
        assert!(
            o.admit(other, "client", "/v1/qwen/reset", &session(s, None), t)
                .is_err()
        );
        assert!(
            o.admit(other, "client", "/v1/qwen/reserve", &session(s, None), t)
                .is_err()
        );
        assert!(
            o.admit(head, "worker", "/v1/qwen/reset", &session(s, None), t)
                .is_err()
        );
        // Reset keeps ownership so a verified run can restart at position zero.
        ok(
            &mut o,
            client,
            "client",
            "/v1/qwen/reset",
            &session(s, None),
            t,
        );
        ok(&mut o, head, "worker", "/v1/qwen/forward", &frame(s), t);
        // Unknown upstreams and unreserved sessions are refused.
        let u = "1c7b4d9f-6a8e-4d2b-8f1c-3e5a7b9d1f2a";
        assert!(
            o.admit(
                client,
                "client",
                "/v1/qwen/reserve",
                &session(u, Some("127.0.0.1:9")),
                t
            )
            .is_err()
        );
        assert!(
            o.admit(client, "client", "/v1/qwen/reset", &session(u, None), t)
                .is_err()
        );
        assert!(
            o.admit(client, "client", "/v1/qwen/forward", &frame(u), t)
                .is_err()
        );
        // Idle expiry frees the session for anyone.
        let later = t + Duration::from_secs(61);
        ok(
            &mut o,
            other,
            "client",
            "/v1/qwen/reserve",
            &session(s, None),
            later,
        );
    }

    #[test]
    fn rejected_worker_replies_record_nothing() {
        let client = PeerId::random();
        let mut o = Owners::new(HashMap::new());
        let t = Instant::now();
        let s = "2d8c5e0a-7b9f-4e3c-9a2d-4f6b8c0e2a3b";
        let pending = o
            .admit(client, "client", "/v1/qwen/reserve", &session(s, None), t)
            .unwrap()
            .unwrap();
        o.commit(pending, 409, t);
        assert!(
            o.admit(client, "client", "/v1/qwen/forward", &frame(s), t)
                .is_err()
        );
    }

    #[test]
    fn placement_leases_are_client_only_owned_and_capped() {
        let (client, other, worker) = (PeerId::random(), PeerId::random(), PeerId::random());
        let mut o = Owners::new(HashMap::new());
        let t = Instant::now();
        let l = "3e9d6f1b-8c0a-4f4d-8b3e-5a7c9d1f3b4c";
        assert!(
            o.admit(worker, "worker", "/v1/node/reserve", &lease(l), t)
                .is_err()
        );
        assert!(
            o.admit(client, "client", "/v1/node/load", &lease(l), t)
                .is_err()
        );
        ok(&mut o, client, "client", "/v1/node/reserve", &lease(l), t);
        assert!(
            o.admit(other, "client", "/v1/node/load", &lease(l), t)
                .is_err()
        );
        assert!(
            o.admit(other, "client", "/v1/node/release", &lease(l), t)
                .is_err()
        );
        ok(&mut o, client, "client", "/v1/node/load", &lease(l), t);
        ok(&mut o, client, "client", "/v1/node/release", &lease(l), t);
        assert!(
            o.admit(client, "client", "/v1/node/load", &lease(l), t)
                .is_err()
        );
        // Renewing past the hold cap ends the lease and blocks new reservations for a while.
        ok(&mut o, client, "client", "/v1/node/reserve", &lease(l), t);
        let mut now = t;
        for _ in 0..3 {
            now += Duration::from_secs(100);
            ok(&mut o, client, "client", "/v1/node/reserve", &lease(l), now);
        }
        now += Duration::from_secs(100);
        assert!(
            o.admit(client, "client", "/v1/node/reserve", &lease(l), now)
                .is_err()
        );
        let fresh = "4f0e7a2c-9d1b-4a5e-8c4f-6b8d0e2a4c5d";
        assert!(
            o.admit(client, "client", "/v1/node/reserve", &lease(fresh), now)
                .is_err()
        );
        ok(
            &mut o,
            other,
            "client",
            "/v1/node/reserve",
            &lease(fresh),
            now,
        );
        let after = now + COOLDOWN + Duration::from_secs(1);
        assert!(
            o.admit(client, "client", "/v1/node/reserve", &lease(l), after)
                .is_ok()
        );
    }

    #[test]
    fn info_and_capacity_need_no_reservation() {
        let mut o = Owners::new(HashMap::new());
        let t = Instant::now();
        for path in ["/v1/qwen/info", "/v1/node/capacity"] {
            assert!(
                o.admit(PeerId::random(), "worker", path, b"", t)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            o.admit(PeerId::random(), "client", "/v1/other", b"", t)
                .is_err()
        );
    }
}
