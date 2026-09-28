//! Contribution credits: signed receipts from both sides of a session, and the
//! authority-signed standing that nodes use to admit new reservations.
//!
//! Work is counted in layer-tokens: one token passed through one decoder layer.
//! A worker claims `tokens × its layers`; the consumer reports the whole route.
//! The portal credits only what both sides agree on, so credits are conserved:
//! every credit a worker earns is paid by a consumer.
use anyhow::{Result, ensure};
use libp2p_identity::{Keypair, PeerId, PublicKey};
use serde::{Deserialize, Serialize};

const RECEIPT_TAG: &str = "sangama/receipt/1";
const STANDING_TAG: &str = "sangama/standing/1";
/// Largest route and model the receipt format accepts.
const MAX_STAGES: usize = 8;
const MAX_LAYERS: u32 = 1024;
/// One session cannot plausibly move more tokens than this.
const MAX_TOKENS: u64 = 10_000_000;
/// Receipts are accepted up to a week after issue, and a minute ahead of the portal clock.
const RECEIPT_AGE: u64 = 7 * 24 * 3600;
const MAX_BALANCES: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptKind {
    /// A worker's claim for its own layer range.
    Work,
    /// A consumer's report of the whole route it used.
    Usage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub peer: String,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub network: String,
    pub session: String,
    pub kind: ReceiptKind,
    pub signer: String,
    pub consumer: String,
    pub model_hash: String,
    /// Decoder layers in the whole model.
    pub layers: u32,
    pub tokens: u64,
    /// Work: exactly the signer's own stage. Usage: every stage in layer order.
    pub stages: Vec<Stage>,
    pub issued: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedReceipt {
    pub receipt: Receipt,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

fn receipt_message(r: &Receipt) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(RECEIPT_TAG, r))?)
}

impl SignedReceipt {
    pub fn sign(receipt: Receipt, key: &Keypair) -> Result<Self> {
        ensure!(
            receipt.signer == key.public().to_peer_id().to_string(),
            "receipt signer must be the signing key"
        );
        Ok(Self {
            signature: key.sign(&receipt_message(&receipt)?)?,
            public_key: key.public().encode_protobuf(),
            receipt,
        })
    }

    /// Checks the signature and the receipt's internal consistency. Membership and
    /// roles are the portal's to check.
    pub fn verify(&self, network: &str, time: u64) -> Result<()> {
        let r = &self.receipt;
        ensure!(r.network == network && !network.is_empty(), "wrong network");
        ensure!(
            r.issued <= time + 60 && r.issued + RECEIPT_AGE >= time,
            "receipt issue time out of range"
        );
        ensure!(uuid_like(&r.session), "invalid session id");
        ensure!(
            r.model_hash.len() == 64 && r.model_hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid model hash"
        );
        ensure!(
            (1..=MAX_LAYERS).contains(&r.layers) && (1..=MAX_TOKENS).contains(&r.tokens),
            "layers or tokens out of range"
        );
        let signer: PeerId = r.signer.parse()?;
        let consumer: PeerId = r.consumer.parse()?;
        ensure!(
            self.public_key.len() <= 128 && self.signature.len() == 64,
            "invalid key/signature size"
        );
        let key = PublicKey::try_decode_protobuf(&self.public_key)?;
        ensure!(key.to_peer_id() == signer, "key does not match signer");
        ensure!(
            key.verify(&receipt_message(r)?, &self.signature),
            "invalid receipt signature"
        );
        ensure!(
            !r.stages.is_empty() && r.stages.len() <= MAX_STAGES,
            "invalid stage count"
        );
        for s in &r.stages {
            let _: PeerId = s.peer.parse()?;
            ensure!(s.start < s.end && s.end <= r.layers, "invalid layer range");
        }
        match r.kind {
            ReceiptKind::Work => {
                ensure!(
                    r.stages.len() == 1 && r.stages[0].peer == r.signer,
                    "work receipt must name only the signer's stage"
                );
                ensure!(signer != consumer, "a worker cannot bill itself");
            }
            ReceiptKind::Usage => {
                ensure!(signer == consumer, "usage must be signed by its consumer");
                let mut next = 0;
                let mut peers = std::collections::HashSet::new();
                for s in &r.stages {
                    ensure!(s.start == next, "usage stages must be contiguous");
                    ensure!(peers.insert(&s.peer), "a peer appears twice in the route");
                    next = s.end;
                }
                ensure!(next == r.layers, "usage must cover every layer");
            }
        }
        Ok(())
    }
}

fn uuid_like(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}

/// Layer-tokens earned by one stage for `tokens` tokens.
pub fn units(stage: &Stage, tokens: u64) -> u64 {
    tokens.saturating_mul(u64::from(stage.end - stage.start))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Balance {
    pub peer: String,
    /// Layer-tokens: positive for net contributors, negative for net consumers.
    pub balance: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Standing {
    pub network: String,
    pub issued: u64,
    pub expires: u64,
    /// How far below zero a peer may go before new reservations are refused.
    pub allowance: i64,
    /// Peers that are not listed have a zero balance.
    pub balances: Vec<Balance>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedStanding {
    pub standing: Standing,
    pub signature: Vec<u8>,
}

fn standing_message(s: &Standing) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(STANDING_TAG, s))?)
}

impl SignedStanding {
    pub fn sign(standing: Standing, key: &Keypair) -> Result<Self> {
        Ok(Self {
            signature: key.sign(&standing_message(&standing)?)?,
            standing,
        })
    }
    pub fn verify(&self, authority: &PublicKey, network: &str, time: u64) -> Result<()> {
        let s = &self.standing;
        ensure!(s.network == network && !network.is_empty(), "wrong network");
        ensure!(
            s.issued <= time + 5 && s.expires > time && s.expires <= s.issued + 60,
            "stale credit standing"
        );
        ensure!(
            s.allowance >= 0 && s.balances.len() <= MAX_BALANCES,
            "invalid standing"
        );
        ensure!(
            authority.verify(&standing_message(s)?, &self.signature),
            "invalid authority signature"
        );
        Ok(())
    }
    pub fn balance(&self, peer: &str) -> i64 {
        self.standing
            .balances
            .iter()
            .find(|b| b.peer == peer)
            .map_or(0, |b| b.balance)
    }
    /// Whether `peer` may start another session under this standing.
    pub fn may_reserve(&self, peer: &str) -> bool {
        self.balance(peer) > -self.standing.allowance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = "0b6a3c8e-5f7d-4c1a-9e0b-2d4f6a8c0e1f";

    fn stage(key: &Keypair, start: u32, end: u32) -> Stage {
        Stage {
            peer: key.public().to_peer_id().to_string(),
            start,
            end,
        }
    }
    fn receipt(
        kind: ReceiptKind,
        signer: &Keypair,
        consumer: &Keypair,
        stages: Vec<Stage>,
    ) -> Receipt {
        Receipt {
            network: "test".into(),
            session: SESSION.into(),
            kind,
            signer: signer.public().to_peer_id().to_string(),
            consumer: consumer.public().to_peer_id().to_string(),
            model_hash: "a".repeat(64),
            layers: 24,
            tokens: 100,
            stages,
            issued: 1_000,
        }
    }

    #[test]
    fn work_and_usage_receipts_verify_and_resist_tampering() {
        let (client, head, tail) = (
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
        );
        let work = SignedReceipt::sign(
            receipt(ReceiptKind::Work, &head, &client, vec![stage(&head, 0, 12)]),
            &head,
        )
        .unwrap();
        assert!(work.verify("test", 1_000).is_ok());
        assert!(work.verify("other", 1_000).is_err());
        assert!(work.verify("test", 1_000 + RECEIPT_AGE + 1).is_err());
        let mut inflated = work.clone();
        inflated.receipt.tokens = 1_000;
        assert!(inflated.verify("test", 1_000).is_err());

        let usage = SignedReceipt::sign(
            receipt(
                ReceiptKind::Usage,
                &client,
                &client,
                vec![stage(&head, 0, 12), stage(&tail, 12, 24)],
            ),
            &client,
        )
        .unwrap();
        assert!(usage.verify("test", 1_000).is_ok());
        assert_eq!(units(&usage.receipt.stages[1], 100), 1_200);

        // Another key cannot sign in the signer's name.
        assert!(
            SignedReceipt::sign(
                receipt(ReceiptKind::Work, &head, &client, vec![stage(&head, 0, 12)]),
                &tail
            )
            .is_err()
        );
        let mut forged = work.clone();
        forged.public_key = tail.public().encode_protobuf();
        assert!(forged.verify("test", 1_000).is_err());
    }

    #[test]
    fn receipts_must_describe_a_coherent_route() {
        let (client, head, tail) = (
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
            Keypair::generate_ed25519(),
        );
        let signed = |r: Receipt, k: &Keypair| SignedReceipt::sign(r, k).unwrap();
        // A worker may claim only its own stage, and never bill itself.
        let other_stage = receipt(ReceiptKind::Work, &head, &client, vec![stage(&tail, 0, 12)]);
        assert!(signed(other_stage, &head).verify("test", 1_000).is_err());
        let own = receipt(ReceiptKind::Work, &head, &head, vec![stage(&head, 0, 12)]);
        assert!(signed(own, &head).verify("test", 1_000).is_err());
        // Usage must cover every layer once, contiguously, with distinct peers.
        for stages in [
            vec![stage(&head, 0, 12)],
            vec![stage(&head, 0, 12), stage(&tail, 13, 24)],
            vec![stage(&head, 0, 12), stage(&head, 12, 24)],
            vec![stage(&head, 0, 12), stage(&tail, 12, 25)],
        ] {
            let usage = receipt(ReceiptKind::Usage, &client, &client, stages);
            assert!(signed(usage, &client).verify("test", 1_000).is_err());
        }
        // Usage is signed by its consumer.
        let usage = receipt(
            ReceiptKind::Usage,
            &head,
            &client,
            vec![stage(&head, 0, 12), stage(&tail, 12, 24)],
        );
        assert!(signed(usage, &head).verify("test", 1_000).is_err());
    }

    #[test]
    fn standing_is_authority_signed_fresh_and_enforces_allowance() {
        let authority = Keypair::generate_ed25519();
        let signed = SignedStanding::sign(
            Standing {
                network: "test".into(),
                issued: 100,
                expires: 130,
                allowance: 500,
                balances: vec![
                    Balance {
                        peer: "debtor".into(),
                        balance: -500,
                    },
                    Balance {
                        peer: "earner".into(),
                        balance: 2_000,
                    },
                ],
            },
            &authority,
        )
        .unwrap();
        assert!(signed.verify(&authority.public(), "test", 101).is_ok());
        assert!(signed.verify(&authority.public(), "test", 130).is_err());
        assert!(signed.verify(&authority.public(), "other", 101).is_err());
        assert!(
            signed
                .verify(&Keypair::generate_ed25519().public(), "test", 101)
                .is_err()
        );
        assert!(!signed.may_reserve("debtor"));
        assert!(signed.may_reserve("earner"));
        assert!(signed.may_reserve("newcomer"));
        let mut tampered = signed;
        tampered.standing.balances[0].balance = 0;
        assert!(tampered.verify(&authority.public(), "test", 101).is_err());
    }
}
