//! Bounded Kademlia store: SQLite persists signed records and provider pointers.
use super::record::{self, Signed};
use anyhow::Result;
use libp2p::{
    PeerId,
    kad::{
        ProviderRecord, Record, RecordKey,
        store::{Error, MemoryStore, MemoryStoreConfig, RecordStore},
    },
};
use rusqlite::{Connection, params};
use std::{
    borrow::Cow,
    path::Path,
    time::{Duration, Instant},
};

pub struct SqliteStore {
    db: Connection,
    memory: MemoryStore,
}
impl SqliteStore {
    pub fn open(path: &Path, peer: PeerId) -> Result<Self> {
        let db = Connection::open(path)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS records (key BLOB PRIMARY KEY, value BLOB NOT NULL, expires INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS providers (key BLOB NOT NULL, peer TEXT NOT NULL, addresses TEXT NOT NULL, expires INTEGER NOT NULL, PRIMARY KEY(key,peer));")?;
        let config = MemoryStoreConfig {
            max_records: 512,
            max_value_bytes: 8192,
            max_provided_keys: 256,
            max_providers_per_key: 20,
        };
        let mut memory = MemoryStore::with_config(peer, config);
        db.execute("DELETE FROM records WHERE expires <= ?", [record::now()])?;
        db.execute("DELETE FROM providers WHERE expires <= ?", [record::now()])?;
        {
            let mut stmt = db.prepare("SELECT key,value FROM records LIMIT 512")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?;
            for row in rows {
                let (key, value) = row?;
                if let Ok(signed) = Signed::decode(&key, &value) {
                    let record = Record {
                        key: RecordKey::new(&key),
                        value,
                        publisher: Some(signed.payload.peer_id.parse()?),
                        expires: Some(
                            Instant::now()
                                + Duration::from_secs(
                                    signed.payload.expires.saturating_sub(record::now()),
                                ),
                        ),
                    };
                    memory.put(record)?;
                }
            }
            let mut stmt =
                db.prepare("SELECT key,peer,addresses,expires FROM providers LIMIT 5120")?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u64>(3)?,
                ))
            })?;
            for row in rows {
                let (key, peer, addresses, expires) = row?;
                if expires <= record::now() {
                    continue;
                }
                let provider = ProviderRecord {
                    key: RecordKey::new(&key),
                    provider: peer.parse()?,
                    addresses: serde_json::from_str(&addresses)?,
                    expires: Some(
                        Instant::now() + Duration::from_secs(expires.saturating_sub(record::now())),
                    ),
                };
                if valid_provider(&provider) {
                    let _ = memory.add_provider(provider);
                }
            }
        }
        Ok(Self { db, memory })
    }
    fn failure(error: impl std::fmt::Display) -> Error {
        tracing::warn!(%error,"DHT record write rejected");
        Error::MaxRecords
    }
    fn prune(&mut self) {
        let expired: Vec<_> = self
            .memory
            .records()
            .filter(|r| r.is_expired(Instant::now()))
            .map(|r| r.key.clone())
            .collect();
        for key in expired {
            self.remove(&key);
        }
        // Remove expired pointers from both memory and SQLite.
        let expired: Vec<(Vec<u8>, String)> = (|| -> rusqlite::Result<_> {
            let mut stmt = self
                .db
                .prepare("SELECT key,peer FROM providers WHERE expires <= ?")?;
            stmt.query_map([record::now()], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect()
        })()
        .unwrap_or_default();
        for (key, peer) in expired {
            if let Ok(peer) = peer.parse() {
                self.remove_provider(&RecordKey::new(&key), &peer);
            }
        }
    }
}
fn valid_provider(p: &ProviderRecord) -> bool {
    let key = std::str::from_utf8(p.key.as_ref()).unwrap_or_default();
    key.strip_prefix(record::MODEL_PREFIX)
        .is_some_and(|hash| record::model_key(hash).is_ok())
        && p.addresses.len() <= 4
        && p.addresses
            .iter()
            .all(|a| record::address(a, false).is_ok())
}
impl RecordStore for SqliteStore {
    type RecordsIter<'a> = std::vec::IntoIter<Cow<'a, Record>>;
    type ProvidedIter<'a> = std::vec::IntoIter<Cow<'a, ProviderRecord>>;
    fn get(&self, key: &RecordKey) -> Option<Cow<'_, Record>> {
        self.memory
            .get(key)
            .filter(|r| !r.is_expired(Instant::now()))
    }
    fn put(&mut self, mut r: Record) -> std::result::Result<(), Error> {
        let signed = Signed::decode(r.key.as_ref(), &r.value).map_err(Self::failure)?;
        r.publisher = Some(signed.payload.peer_id.parse().map_err(Self::failure)?);
        r.expires = Some(
            Instant::now()
                + Duration::from_secs(signed.payload.expires.saturating_sub(record::now())),
        );
        if let Some(old) = self.get(&r.key) {
            let old = Signed::decode(old.key.as_ref(), &old.value).map_err(Self::failure)?;
            if old.payload.issued > signed.payload.issued {
                return Err(Error::MaxRecords);
            }
        }
        self.prune();
        let old = self.memory.get(&r.key).map(Cow::into_owned);
        self.memory.put(r.clone())?;
        if let Err(error) = self.db.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?)",
            params![r.key.as_ref(), r.value, signed.payload.expires],
        ) {
            self.memory.remove(&r.key);
            if let Some(old) = old {
                let _ = self.memory.put(old);
            }
            return Err(Self::failure(error));
        }
        Ok(())
    }
    fn remove(&mut self, key: &RecordKey) {
        if let Err(error) = self
            .db
            .execute("DELETE FROM records WHERE key=?", [key.as_ref()])
        {
            tracing::warn!(%error,"DHT delete failed");
        }
        self.memory.remove(key);
    }
    fn records(&self) -> Self::RecordsIter<'_> {
        self.memory
            .records()
            .filter(|r| !r.is_expired(Instant::now()))
            .map(|r| Cow::Owned(r.into_owned()))
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn add_provider(&mut self, mut p: ProviderRecord) -> std::result::Result<(), Error> {
        if !valid_provider(&p) {
            return Err(Error::ValueTooLarge);
        }
        self.prune();
        let ttl = p
            .expires
            .map(|e| e.saturating_duration_since(Instant::now()).as_secs())
            .unwrap_or(record::TTL)
            .min(record::TTL);
        p.expires = Some(Instant::now() + Duration::from_secs(ttl));
        let count: u64 = self
            .db
            .query_row("SELECT COUNT(DISTINCT key) FROM providers", [], |r| {
                r.get(0)
            })
            .map_err(Self::failure)?;
        if count >= 256 && self.memory.providers(&p.key).is_empty() {
            return Err(Error::MaxProvidedKeys);
        }
        self.memory.add_provider(p.clone())?;
        if let Err(error) = self.db.execute(
            "INSERT OR REPLACE INTO providers VALUES (?,?,?,?)",
            params![
                p.key.as_ref(),
                p.provider.to_string(),
                serde_json::to_string(&p.addresses).map_err(Self::failure)?,
                record::now() + ttl
            ],
        ) {
            self.memory.remove_provider(&p.key, &p.provider);
            return Err(Self::failure(error));
        }
        // Trim persisted provider candidates to the same closest-peer set retained by MemoryStore.
        let retained = self.memory.providers(&p.key);
        let peers: Vec<String> = retained.iter().map(|r| r.provider.to_string()).collect();
        let mut stmt = self
            .db
            .prepare("SELECT peer FROM providers WHERE key=?")
            .map_err(Self::failure)?;
        let old: Vec<String> = stmt
            .query_map([p.key.as_ref()], |r| r.get(0))
            .map_err(Self::failure)?
            .collect::<std::result::Result<_, _>>()
            .map_err(Self::failure)?;
        for old in old {
            if !peers.contains(&old) {
                self.db
                    .execute(
                        "DELETE FROM providers WHERE key=? AND peer=?",
                        params![p.key.as_ref(), old],
                    )
                    .map_err(Self::failure)?;
            }
        }
        Ok(())
    }
    fn providers(&self, key: &RecordKey) -> Vec<ProviderRecord> {
        self.memory
            .providers(key)
            .into_iter()
            .filter(|p| !p.is_expired(Instant::now()))
            .collect()
    }
    fn provided(&self) -> Self::ProvidedIter<'_> {
        self.memory
            .provided()
            .filter(|p| !p.is_expired(Instant::now()))
            .map(|p| Cow::Owned(p.into_owned()))
            .collect::<Vec<_>>()
            .into_iter()
    }
    fn remove_provider(&mut self, key: &RecordKey, peer: &PeerId) {
        if let Err(error) = self.db.execute(
            "DELETE FROM providers WHERE key=? AND peer=?",
            params![key.as_ref(), peer.to_string()],
        ) {
            tracing::warn!(%error,"DHT provider delete failed");
        }
        self.memory.remove_provider(key, peer);
    }
}
