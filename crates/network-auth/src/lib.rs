//! Shared membership wire format. Authority keys are pinned out of band.
use anyhow::{Result, ensure};
pub use libp2p_identity;
use libp2p_identity::{Keypair, PublicKey};
use serde::{Deserialize, Serialize};

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Member {
    pub peer: String,
    pub role: String,
    pub expires: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub network: String,
    pub issued: u64,
    pub expires: u64,
    pub members: Vec<Member>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedSnapshot {
    pub snapshot: Snapshot,
    pub signature: Vec<u8>,
}
impl SignedSnapshot {
    pub fn sign(snapshot: Snapshot, key: &Keypair) -> Result<Self> {
        let signature = key.sign(&serde_json::to_vec(&snapshot)?)?;
        Ok(Self {
            snapshot,
            signature,
        })
    }
    pub fn verify(&self, authority: &PublicKey, network: &str, time: u64) -> Result<()> {
        let s = &self.snapshot;
        ensure!(s.network == network && !network.is_empty(), "wrong network");
        ensure!(
            s.issued <= time + 5 && s.expires > time && s.expires <= s.issued + 15,
            "stale membership snapshot"
        );
        ensure!(s.members.len() <= 1024, "too many members");
        ensure!(
            authority.verify(&serde_json::to_vec(s)?, &self.signature),
            "invalid authority signature"
        );
        let mut seen = std::collections::HashSet::new();
        for m in &s.members {
            let _: libp2p_identity::PeerId = m.peer.parse()?;
            ensure!(
                seen.insert(&m.peer) && ["worker", "client", "relay"].contains(&m.role.as_str()),
                "invalid member"
            );
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Join {
    pub invitation: String,
    pub nonce: String,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}
pub fn proof(network: &str, nonce: &str, invitation: &str) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(
        "sangama/join/1",
        network,
        nonce,
        invitation,
    ))?)
}
impl Join {
    pub fn peer(&self, network: &str) -> Result<String> {
        ensure!(
            self.invitation.len() == 64 && self.invitation.bytes().all(|x| x.is_ascii_hexdigit()),
            "invalid invitation"
        );
        ensure!(
            self.nonce.len() == 64 && self.nonce.bytes().all(|x| x.is_ascii_hexdigit()),
            "invalid nonce"
        );
        ensure!(
            self.public_key.len() <= 128 && self.signature.len() == 64,
            "invalid key/signature size"
        );
        let key = PublicKey::try_decode_protobuf(&self.public_key)?;
        ensure!(key.clone().try_into_ed25519().is_ok(), "Ed25519 required");
        ensure!(
            key.verify(
                &proof(network, &self.nonce, &self.invitation)?,
                &self.signature
            ),
            "invalid ownership proof"
        );
        Ok(key.to_peer_id().to_string())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn authority_scope_expiry_tampering() {
        let key = Keypair::generate_ed25519();
        let mut signed = SignedSnapshot::sign(
            Snapshot {
                network: "test".into(),
                issued: 100,
                expires: 115,
                members: vec![],
            },
            &key,
        )
        .unwrap();
        assert!(signed.verify(&key.public(), "test", 101).is_ok());
        assert!(signed.verify(&key.public(), "other", 101).is_err());
        assert!(signed.verify(&key.public(), "test", 115).is_err());
        assert!(
            signed
                .verify(&Keypair::generate_ed25519().public(), "test", 101)
                .is_err()
        );
        signed.snapshot.expires += 1;
        assert!(signed.verify(&key.public(), "test", 101).is_err());
    }
    #[test]
    fn proof_binds_invitation_network_and_key() {
        let key = Keypair::generate_ed25519();
        let mut j = Join {
            invitation: "a".repeat(64),
            nonce: "b".repeat(64),
            public_key: key.public().encode_protobuf(),
            signature: vec![],
        };
        j.signature = key
            .sign(&proof("test", &j.nonce, &j.invitation).unwrap())
            .unwrap();
        assert_eq!(
            j.peer("test").unwrap(),
            key.public().to_peer_id().to_string()
        );
        assert!(j.peer("other").is_err());
        j.invitation = "c".repeat(64);
        assert!(j.peer("test").is_err());
    }
}
