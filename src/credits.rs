//! Meters admitted sessions and turns finished ones into signed credit receipts.
//!
//! A worker node counts the tokens its own layers processed for each session and
//! the client that reserved it. A client node counts the tokens it sent and learns
//! the route from its reservations, taking each stage's layers from that peer's own
//! info reply, so no peer can describe another's work. Once a session has been
//! idle longer than any worker keeps it, both sides sign what they saw and send it
//! to the portal, which credits only what they agree on.
//!
//! Metering happens in the mesh layer because only it knows verified peer
//! identities. Nothing here runs per token or changes the inference path.
use anyhow::{Result, ensure};
use libp2p::PeerId;
use sangama_network_auth::{
    credits::{Receipt, ReceiptKind, SignedReceipt, SignedStanding, Stage},
    libp2p_identity::{Keypair, PublicKey},
    now,
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

/// Longer than the worker's 60 s session expiry, so a finished session cannot resume.
const IDLE: Duration = Duration::from_secs(90);
const MAX_SESSIONS: usize = 256;
/// Signed receipts waiting for the portal; the oldest are dropped beyond this.
const MAX_OUTBOX: usize = 1024;

struct Work {
    consumer: PeerId,
    model_hash: String,
    start: u32,
    end: u32,
    tokens: u64,
    last: Instant,
}

#[derive(Default)]
struct Usage {
    /// Each reserved stage and the peer it accepts forwards from; `None` is the head.
    upstream: HashMap<PeerId, Option<PeerId>>,
    model_hash: Option<String>,
    tokens: u64,
    last: Option<Instant>,
}

pub struct Meter {
    me: PeerId,
    network: String,
    layers: u32,
    work: HashMap<String, Work>,
    usage: HashMap<String, Usage>,
    /// Model and layers each peer last reported about itself.
    shards: HashMap<PeerId, (String, u32, u32)>,
}

impl Meter {
    pub fn new(me: PeerId, network: String, layers: u32) -> Self {
        Self {
            me,
            network,
            layers,
            work: HashMap::new(),
            usage: HashMap::new(),
            shards: HashMap::new(),
        }
    }

    /// Worker side: this node's layers processed `tokens` tokens for `consumer`.
    pub fn record_work(
        &mut self,
        session: &str,
        consumer: PeerId,
        model_hash: &str,
        (start, end): (u32, u32),
        tokens: u64,
        now: Instant,
    ) {
        if consumer == self.me || start >= end || end > self.layers {
            return;
        }
        if let Some(w) = self.work.get_mut(session) {
            // A session keeps one consumer, model and layer range; anything else is not billed.
            if w.consumer == consumer
                && w.model_hash == model_hash
                && (w.start, w.end) == (start, end)
            {
                w.tokens = w.tokens.saturating_add(tokens);
                w.last = now;
            }
        } else if self.work.len() < MAX_SESSIONS {
            self.work.insert(
                session.into(),
                Work {
                    consumer,
                    model_hash: model_hash.into(),
                    start,
                    end,
                    tokens,
                    last: now,
                },
            );
        }
    }

    /// Client side: `peer` accepted this session, taking forwards from `upstream`.
    pub fn record_reserve(
        &mut self,
        session: &str,
        peer: PeerId,
        upstream: Option<PeerId>,
        now: Instant,
    ) {
        if !self.usage.contains_key(session) && self.usage.len() >= MAX_SESSIONS {
            return;
        }
        let u = self.usage.entry(session.into()).or_default();
        u.upstream.insert(peer, upstream);
        u.last = Some(now);
    }

    /// Client side: `peer` reported serving these layers of this model.
    pub fn record_info(&mut self, peer: PeerId, model_hash: &str, (start, end): (u32, u32)) {
        if self.shards.contains_key(&peer) || self.shards.len() < MAX_SESSIONS {
            self.shards.insert(peer, (model_hash.into(), start, end));
        }
    }

    /// Client side: the route returned output for `tokens` tokens.
    pub fn record_usage(&mut self, session: &str, model_hash: &str, tokens: u64, now: Instant) {
        // Only sessions this node reserved are its own; a worker forwarding downstream is not a consumer.
        let Some(u) = self.usage.get_mut(session) else {
            return;
        };
        if u.model_hash.as_deref().is_some_and(|h| h != model_hash) {
            return;
        }
        u.model_hash = Some(model_hash.into());
        u.tokens = u.tokens.saturating_add(tokens);
        u.last = Some(now);
    }

    /// Removes sessions idle past `IDLE` (or all, when `all`) and returns their receipts.
    pub fn drain(&mut self, now: Instant, all: bool) -> Vec<Receipt> {
        let done = |last: Instant| all || now.saturating_duration_since(last) > IDLE;
        let issued = sangama_network_auth::now();
        let mut receipts = Vec::new();
        let finished: Vec<String> = self
            .work
            .iter()
            .filter(|(_, w)| done(w.last))
            .map(|(s, _)| s.clone())
            .collect();
        for session in finished {
            let w = self.work.remove(&session).unwrap();
            if w.tokens == 0 {
                continue;
            }
            receipts.push(Receipt {
                network: self.network.clone(),
                session,
                kind: ReceiptKind::Work,
                signer: self.me.to_string(),
                consumer: w.consumer.to_string(),
                model_hash: w.model_hash,
                layers: self.layers,
                tokens: w.tokens,
                stages: vec![Stage {
                    peer: self.me.to_string(),
                    start: w.start,
                    end: w.end,
                }],
                issued,
            });
        }
        let finished: Vec<String> = self
            .usage
            .iter()
            .filter(|(_, u)| u.last.is_none_or(done))
            .map(|(s, _)| s.clone())
            .collect();
        for session in finished {
            let u = self.usage.remove(&session).unwrap();
            let Some(model_hash) = u.model_hash.clone() else {
                continue;
            };
            let Some(stages) = u.stages(&model_hash, &self.shards) else {
                continue;
            };
            if u.tokens == 0 {
                continue;
            }
            receipts.push(Receipt {
                network: self.network.clone(),
                session,
                kind: ReceiptKind::Usage,
                signer: self.me.to_string(),
                consumer: self.me.to_string(),
                model_hash,
                layers: self.layers,
                tokens: u.tokens,
                stages,
                issued,
            });
        }
        receipts
    }
}

impl Usage {
    /// Orders reserved peers from the head along their upstream links, with the layers
    /// each reported for this model. An ambiguous route or unknown stage yields nothing;
    /// the portal rejects routes that do not cover the model.
    fn stages(
        &self,
        model_hash: &str,
        shards: &HashMap<PeerId, (String, u32, u32)>,
    ) -> Option<Vec<Stage>> {
        let mut stages = Vec::new();
        let mut previous = None;
        while stages.len() < self.upstream.len() {
            let mut next = self
                .upstream
                .iter()
                .filter(|(_, up)| **up == previous)
                .map(|(peer, _)| *peer);
            let peer = next.next()?;
            if next.next().is_some() {
                return None;
            }
            let (hash, start, end) = shards.get(&peer)?;
            if hash != model_hash {
                return None;
            }
            stages.push(Stage {
                peer: peer.to_string(),
                start: *start,
                end: *end,
            });
            previous = Some(peer);
        }
        Some(stages)
    }
}

/// Signed receipts waiting to reach the portal.
#[derive(Default)]
pub struct Outbox(std::collections::VecDeque<SignedReceipt>);

impl Outbox {
    pub fn push(&mut self, receipt: SignedReceipt) {
        if self.0.len() >= MAX_OUTBOX {
            self.0.pop_front();
        }
        self.0.push_back(receipt);
    }
    /// Sends queued receipts in order, stopping at the first that the portal could not
    /// take right now. Receipts it refuses outright are dropped.
    pub async fn flush(&mut self, http: &reqwest::Client, portal: &str) {
        let url = format!("{}/v1/credits/receipts", portal.trim_end_matches('/'));
        while let Some(receipt) = self.0.front() {
            match http.post(&url).json(receipt).send().await {
                Ok(r) if r.status().is_success() || r.status().is_client_error() => {
                    if !r.status().is_success() {
                        tracing::warn!(status = %r.status(), session = %receipt.receipt.session, "portal refused credit receipt");
                    }
                    self.0.pop_front();
                }
                _ => break,
            }
        }
    }
}

pub fn sign(receipts: Vec<Receipt>, key: &Keypair, outbox: &mut Outbox) {
    for r in receipts {
        match SignedReceipt::sign(r, key) {
            Ok(signed) => outbox.push(signed),
            Err(error) => tracing::warn!(%error, "could not sign credit receipt"),
        }
    }
}

/// Fetches the authority-signed credit standing. `Ok(None)` means the portal does not
/// enforce credits.
pub async fn standing(
    http: &reqwest::Client,
    portal: &str,
    authority: &PublicKey,
    network: &str,
) -> Result<Option<SignedStanding>> {
    let mut r = http
        .get(format!(
            "{}/v1/credits/standing",
            portal.trim_end_matches('/')
        ))
        .send()
        .await?;
    if r.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    ensure!(r.status().is_success(), "standing unavailable");
    let mut bytes = Vec::new();
    while let Some(c) = r.chunk().await? {
        ensure!(bytes.len() + c.len() <= 256 * 1024, "standing too large");
        bytes.extend_from_slice(&c);
    }
    let s: SignedStanding = serde_json::from_slice(&bytes)?;
    s.verify(authority, network, now())?;
    Ok(Some(s))
}

/// Whether a reservation from `peer` must be refused. Without a current standing the
/// ledger is unreachable or disabled, and reservations proceed.
pub fn over_allowance(standing: Option<&SignedStanding>, peer: &PeerId) -> bool {
    standing.is_some_and(|s| s.standing.expires > now() && !s.may_reserve(&peer.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: &str = "0b6a3c8e-5f7d-4c1a-9e0b-2d4f6a8c0e1f";
    fn hash() -> String {
        "a".repeat(64)
    }

    #[test]
    fn worker_bills_its_range_once_idle_and_ignores_changes() {
        let (me, client, other) = (PeerId::random(), PeerId::random(), PeerId::random());
        let mut m = Meter::new(me, "test".into(), 24);
        let t = Instant::now();
        m.record_work(S, client, &hash(), (12, 24), 30, t);
        m.record_work(S, client, &hash(), (12, 24), 1, t);
        // A different consumer or range for the same session is not added.
        m.record_work(S, other, &hash(), (12, 24), 100, t);
        m.record_work(S, client, &hash(), (0, 24), 100, t);
        // Nobody bills themselves.
        m.record_work("self", me, &hash(), (12, 24), 5, t);
        assert!(m.drain(t + Duration::from_secs(10), false).is_empty());
        let receipts = m.drain(t + IDLE + Duration::from_secs(1), false);
        assert_eq!(receipts.len(), 1);
        let r = &receipts[0];
        assert_eq!((r.kind.clone(), r.tokens), (ReceiptKind::Work, 31));
        assert_eq!(r.consumer, client.to_string());
        assert_eq!((r.stages[0].start, r.stages[0].end), (12, 24));
        let key = Keypair::generate_ed25519();
        let mut m = Meter::new(key.public().to_peer_id(), "test".into(), 24);
        m.record_work(S, client, &hash(), (12, 24), 7, t);
        let mut outbox = Outbox::default();
        sign(m.drain(t, true), &key, &mut outbox);
        assert!(outbox.0[0].verify("test", now()).is_ok());
    }

    #[test]
    fn client_orders_route_from_reservations_and_info() {
        let (me, head, tail) = (PeerId::random(), PeerId::random(), PeerId::random());
        let mut m = Meter::new(me, "test".into(), 24);
        let t = Instant::now();
        m.record_info(head, &hash(), (0, 12));
        m.record_info(tail, &hash(), (12, 24));
        // Reservations arrive in any order; upstream links define the route.
        m.record_reserve(S, tail, Some(head), t);
        m.record_reserve(S, head, None, t);
        m.record_usage(S, &hash(), 20, t);
        m.record_usage(S, &hash(), 1, t);
        // Output for a different model is not counted.
        m.record_usage(S, &"b".repeat(64), 50, t);
        // Forwards for sessions this node never reserved are someone else's.
        m.record_usage("unreserved", &hash(), 9, t);
        let receipts = m.drain(t, true);
        assert_eq!(receipts.len(), 1);
        let r = &receipts[0];
        assert_eq!((r.kind.clone(), r.tokens), (ReceiptKind::Usage, 21));
        assert_eq!(r.stages[0].peer, head.to_string());
        assert_eq!(r.stages[1].peer, tail.to_string());
        assert_eq!((r.stages[1].start, r.stages[1].end), (12, 24));
    }

    #[test]
    fn ambiguous_or_incomplete_routes_are_not_reported() {
        let (me, a, b) = (PeerId::random(), PeerId::random(), PeerId::random());
        let t = Instant::now();
        let meter = || {
            let mut m = Meter::new(me, "test".into(), 24);
            m.record_info(a, &hash(), (0, 12));
            m.record_info(b, &hash(), (12, 24));
            m
        };
        // Two heads.
        let mut m = meter();
        m.record_reserve(S, a, None, t);
        m.record_reserve(S, b, None, t);
        m.record_usage(S, &hash(), 5, t);
        assert!(m.drain(t, true).is_empty());
        // Reserved but never produced output.
        let mut m = meter();
        m.record_reserve(S, a, None, t);
        assert!(m.drain(t, true).is_empty());
        // A stage that never described itself, or described another model.
        let mut m = meter();
        let c = PeerId::random();
        m.record_reserve(S, a, None, t);
        m.record_reserve(S, c, Some(a), t);
        m.record_usage(S, &hash(), 5, t);
        assert!(m.drain(t, true).is_empty());
        let mut m = meter();
        m.record_info(b, &"b".repeat(64), (12, 24));
        m.record_reserve(S, a, None, t);
        m.record_reserve(S, b, Some(a), t);
        m.record_usage(S, &hash(), 5, t);
        assert!(m.drain(t, true).is_empty());
    }
}
