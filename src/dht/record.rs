use anyhow::{Result, ensure};
use libp2p::{
    Multiaddr, PeerId,
    identity::{Keypair, PublicKey},
    multiaddr::Protocol,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

pub const TTL: u64 = 300;
pub const PREFIX: &str = "/mesh/peer/";
pub const MODEL_PREFIX: &str = "/mesh/model/";
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
pub fn model_key(hash: &str) -> Result<String> {
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|x| x.is_ascii_digit() || (b'a'..=b'f').contains(&x)),
        "model must be a lowercase SHA256 manifest hash"
    );
    Ok(format!("{MODEL_PREFIX}{hash}"))
}

/// Limit this first discovery network to explicit IP/TCP private-overlay endpoints.
pub fn address(address: &Multiaddr, listen: bool) -> Result<()> {
    let mut parts = address.iter();
    let allowed = match parts.next() {
        Some(Protocol::Ip4(ip)) => {
            ip.is_loopback()
                || ip.is_private()
                || (ip.octets()[0] == 100 && (64..=127).contains(&ip.octets()[1]))
        }
        Some(Protocol::Ip6(ip)) => ip.is_loopback() || (ip.segments()[0] & 0xfe00 == 0xfc00),
        _ => false,
    };
    ensure!(
        allowed,
        "DHT addresses must be explicit loopback, private LAN, or Tailscale IPs"
    );
    ensure!(
        matches!(parts.next(), Some(Protocol::Tcp(port)) if listen || port != 0),
        "expected TCP port"
    );
    ensure!(
        parts.next().is_none(),
        "expected only /ip4|ip6/ADDRESS/tcp/PORT"
    );
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Advertisement {
    pub version: u8,
    pub peer_id: String,
    pub model_hash: String,
    pub start: usize,
    pub end: usize,
    pub addresses: Vec<String>,
    pub issued: u64,
    pub expires: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signed {
    pub payload: Advertisement,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}
impl Signed {
    pub fn new(
        key: &Keypair,
        hash: String,
        start: usize,
        end: usize,
        addresses: Vec<String>,
    ) -> Result<Self> {
        let payload = Advertisement {
            version: 1,
            peer_id: key.public().to_peer_id().to_string(),
            model_hash: hash,
            start,
            end,
            addresses,
            issued: now(),
            expires: now() + TTL,
        };
        let signature = key.sign(&serde_json::to_vec(&payload)?)?;
        let signed = Self {
            payload,
            public_key: key.public().encode_protobuf(),
            signature,
        };
        signed.validate(&format!("{PREFIX}{}", signed.payload.peer_id))?;
        Ok(signed)
    }
    pub fn validate(&self, key: &str) -> Result<PeerId> {
        let p = &self.payload;
        ensure!(
            p.version == 1 && p.start < p.end && p.end <= 24,
            "unsupported advertisement/range"
        );
        model_key(&p.model_hash)?;
        let time = now();
        ensure!(
            p.issued <= time + 30
                && p.expires > time
                && p.expires > p.issued
                && p.expires - p.issued <= TTL
                && p.expires <= time + TTL + 30,
            "expired or invalid advertisement lifetime"
        );
        ensure!(
            !p.addresses.is_empty() && p.addresses.len() <= 4,
            "expected 1..4 addresses"
        );
        for a in &p.addresses {
            address(&a.parse()?, false)?;
        }
        ensure!(
            self.public_key.len() <= 128 && self.signature.len() <= 128,
            "oversized identity/signature"
        );
        let public = PublicKey::try_decode_protobuf(&self.public_key)?;
        let peer = public.to_peer_id();
        ensure!(
            p.peer_id == peer.to_string() && key == format!("{PREFIX}{peer}"),
            "peer identity/key mismatch"
        );
        ensure!(
            public.verify(&serde_json::to_vec(p)?, &self.signature),
            "invalid advertisement signature"
        );
        Ok(peer)
    }
    pub fn decode(key: &[u8], bytes: &[u8]) -> Result<Self> {
        ensure!(key.len() <= 128 && bytes.len() <= 8192, "oversized record");
        let signed: Self = serde_json::from_slice(bytes)?;
        signed.validate(std::str::from_utf8(key)?)?;
        Ok(signed)
    }
}
