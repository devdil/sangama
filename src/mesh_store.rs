//! Bounded, persistent records for the admitted mesh's separate Kademlia namespace.
use crate::qwen::network::Info;
use anyhow::{Result, ensure};
use libp2p::{
    PeerId,
    identity::{Keypair, PublicKey},
    kad::{
        ProviderRecord, Record, RecordKey,
        store::{Error, MemoryStore, MemoryStoreConfig, RecordStore},
    },
};
use sangama_network_auth::now;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    path::Path,
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Offer {
    pub network: String,
    pub peer: String,
    pub info: Info,
    pub issued: u64,
    pub expires: u64,
    pub public_key: Vec<u8>,
    pub signature: Vec<u8>,
}
impl Offer {
    fn message(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&(
            &self.network,
            &self.peer,
            &self.info,
            self.issued,
            self.expires,
        ))?)
    }
    pub fn new(network: String, info: Info, key: &Keypair) -> Result<Self> {
        let mut s = Self {
            network,
            peer: key.public().to_peer_id().to_string(),
            info,
            issued: now(),
            expires: now() + 60,
            public_key: key.public().encode_protobuf(),
            signature: vec![],
        };
        s.signature = key.sign(&s.message()?)?;
        Ok(s)
    }
    pub fn decode(record: &Record, network: &str) -> Result<Self> {
        ensure!(record.value.len() <= 8192, "oversized advertisement");
        let s: Self = serde_json::from_slice(&record.value)?;
        ensure!(
            s.network == network
                && s.issued <= now() + 5
                && s.expires > now()
                && s.expires > s.issued
                && s.expires <= s.issued + 60,
            "invalid advertisement lifetime/scope"
        );
        ensure!(
            record.key == RecordKey::new(&format!("/sangama/offer/{}", s.peer)),
            "advertisement key mismatch"
        );
        ensure!(
            s.public_key.len() <= 128 && s.signature.len() == 64,
            "invalid signing key"
        );
        let key = PublicKey::try_decode_protobuf(&s.public_key)?;
        ensure!(
            key.to_peer_id().to_string() == s.peer && key.verify(&s.message()?, &s.signature),
            "invalid advertisement signature"
        );
        let h = &s.info.model_hash;
        ensure!(
            h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid manifest hash"
        );
        ensure!(
            s.info.model_id == crate::qwen::MODEL_ID
                && s.info.precision == "f32"
                && s.info.shard.start < s.info.shard.end
                && s.info.shard.end <= 24
                && s.info.shard.index < 8,
            "invalid shard advertisement"
        );
        Ok(s)
    }
    pub fn record(&self) -> Result<Record> {
        Ok(Record {
            key: RecordKey::new(&format!("/sangama/offer/{}", self.peer)),
            value: serde_json::to_vec(self)?,
            publisher: Some(self.peer.parse()?),
            expires: Some(Instant::now() + Duration::from_secs(self.expires.saturating_sub(now()))),
        })
    }
}
pub struct Store {
    memory: MemoryStore,
    db: rusqlite::Connection,
    network: String,
}
impl Store {
    pub fn open(path: &Path, peer: PeerId, network: String) -> Result<Self> {
        let db = rusqlite::Connection::open(path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS offers (key BLOB PRIMARY KEY,value BLOB NOT NULL);",
        )?;
        let memory = MemoryStore::with_config(
            peer,
            MemoryStoreConfig {
                max_records: 1024,
                max_value_bytes: 8192,
                max_provided_keys: 32,
                max_providers_per_key: 64,
            },
        );
        let mut s = Self {
            memory,
            db,
            network,
        };
        let rows: Vec<(Vec<u8>, Vec<u8>)> = {
            let mut stmt = s.db.prepare("SELECT key,value FROM offers LIMIT 1024")?;
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<std::result::Result<_, _>>()?
        };
        for (key, value) in rows {
            let r = Record::new(key, value);
            if let Ok(o) = Offer::decode(&r, &s.network) {
                s.memory.put(o.record()?)?;
            } else {
                s.db.execute("DELETE FROM offers WHERE key=?1", [r.key.to_vec()])?;
            }
        }
        Ok(s)
    }
}
impl RecordStore for Store {
    type RecordsIter<'a> = <MemoryStore as RecordStore>::RecordsIter<'a>;
    type ProvidedIter<'a> = <MemoryStore as RecordStore>::ProvidedIter<'a>;
    fn get(&self, k: &RecordKey) -> Option<Cow<'_, Record>> {
        self.memory.get(k)
    }
    fn put(&mut self, r: Record) -> std::result::Result<(), Error> {
        let o = Offer::decode(&r, &self.network).map_err(|_| Error::ValueTooLarge)?;
        if let Some(old) = self.memory.get(&r.key)
            && let Ok(previous) = Offer::decode(&old, &self.network)
            && previous.issued > o.issued
        {
            return Ok(());
        }
        let expired: Vec<_> = self
            .memory
            .records()
            .filter(|r| r.is_expired(Instant::now()))
            .map(|r| r.key.clone())
            .collect();
        for k in expired {
            self.remove(&k);
        }
        let normalized = o.record().map_err(|_| Error::ValueTooLarge)?;
        self.memory.put(normalized)?;
        self.db
            .execute(
                "INSERT OR REPLACE INTO offers (key,value) VALUES (?1,?2)",
                rusqlite::params![r.key.to_vec(), r.value],
            )
            .map_err(|_| Error::MaxRecords)?;
        Ok(())
    }
    fn remove(&mut self, k: &RecordKey) {
        self.memory.remove(k);
        let _ = self
            .db
            .execute("DELETE FROM offers WHERE key=?1", [k.to_vec()]);
    }
    fn records(&self) -> Self::RecordsIter<'_> {
        self.memory.records()
    }
    fn add_provider(&mut self, r: ProviderRecord) -> std::result::Result<(), Error> {
        self.memory.add_provider(r)
    }
    fn providers(&self, k: &RecordKey) -> Vec<ProviderRecord> {
        self.memory.providers(k)
    }
    fn provided(&self) -> Self::ProvidedIter<'_> {
        self.memory.provided()
    }
    fn remove_provider(&mut self, k: &RecordKey, p: &PeerId) {
        self.memory.remove_provider(k, p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn offer() -> Offer {
        Offer::new(
            "test".into(),
            Info {
                model_id: crate::qwen::MODEL_ID.into(),
                model_hash: "a".repeat(64),
                shard: crate::qwen::ShardSpec {
                    index: 0,
                    start: 0,
                    end: 12,
                    file: "shard.safetensors".into(),
                    sha256: "b".repeat(64),
                    file_bytes: 100,
                    tensor_count: 1,
                },
                device: "cpu".into(),
                engine: "candle".into(),
                weights_sha256: None,
                precision: "f32".into(),
                pid: 1,
                busy: false,
                memory: None,
            },
            &Keypair::generate_ed25519(),
        )
        .unwrap()
    }
    #[test]
    fn rejects_forged_expired_cross_network_and_mismatched_ads() {
        let good = offer();
        let record = good.record().unwrap();
        assert!(Offer::decode(&record, "test").is_ok());
        assert!(Offer::decode(&record, "other").is_err());
        let mut wrong = record.clone();
        wrong.key = RecordKey::new(&"wrong");
        assert!(Offer::decode(&wrong, "test").is_err());
        let mut modified = good.clone();
        modified.info.shard.end = 24;
        assert!(Offer::decode(&modified.record().unwrap(), "test").is_err());
        let mut expired = good.clone();
        expired.expires = now() - 1;
        assert!(Offer::decode(&expired.record().unwrap(), "test").is_err());
        let mut forged = good;
        forged.public_key = Keypair::generate_ed25519().public().encode_protobuf();
        assert!(Offer::decode(&forged.record().unwrap(), "test").is_err());
    }
    #[test]
    fn sqlite_restores_only_valid_records() {
        let path = std::env::temp_dir().join(format!(
            "sangama-mesh-store-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let peer = Keypair::generate_ed25519().public().to_peer_id();
        let record = offer().record().unwrap();
        {
            let mut store = Store::open(&path, peer, "test".into()).unwrap();
            store.put(record.clone()).unwrap();
        }
        {
            let store = Store::open(&path, peer, "test".into()).unwrap();
            assert!(store.get(&record.key).is_some());
        }
        {
            let store = Store::open(&path, peer, "other".into()).unwrap();
            assert!(store.get(&record.key).is_none());
        }
        std::fs::remove_file(path).unwrap();
    }
}
