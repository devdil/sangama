use libp2p::{
    identity::Keypair,
    kad::{ProviderRecord, Record, RecordKey, store::RecordStore},
};
use sangama::dht::{
    self, Offer,
    record::{PREFIX, Signed},
    store::SqliteStore,
};
use std::{path::PathBuf, time::Duration};
fn temp() -> PathBuf {
    let path = std::env::temp_dir().join(format!("mesh-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&path).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    path
}
fn signed(key: &Keypair) -> Signed {
    Signed::new(
        key,
        "a".repeat(64),
        0,
        12,
        vec!["/ip4/127.0.0.1/tcp/9000".into()],
    )
    .unwrap()
}
fn record(value: &Signed) -> Record {
    Record::new(
        RecordKey::new(&format!("{PREFIX}{}", value.payload.peer_id)),
        serde_json::to_vec(value).unwrap(),
    )
}
#[test]
fn sqlite_persists_and_rejects_tampering_expiry_and_identity_substitution() {
    let dir = temp();
    let key = Keypair::generate_ed25519();
    let peer = key.public().to_peer_id();
    let value = signed(&key);
    let r = record(&value);
    let model = RecordKey::new(&dht::record::model_key(&value.payload.model_hash).unwrap());
    {
        let mut store = SqliteStore::open(&dir.join("records.sqlite"), peer).unwrap();
        store.put(r.clone()).unwrap();
        store
            .add_provider(ProviderRecord::new(
                model.clone(),
                peer,
                vec!["/ip4/127.0.0.1/tcp/9000".parse().unwrap()],
            ))
            .unwrap();
        let mut tampered = value.clone();
        tampered.payload.end = 24;
        assert!(store.put(record(&tampered)).is_err());
        let mut expired = value.clone();
        expired.payload.expires = dht::record::now() - 1;
        expired.signature = key
            .sign(&serde_json::to_vec(&expired.payload).unwrap())
            .unwrap();
        assert!(store.put(record(&expired)).is_err());
        let mut wrong = value.clone();
        wrong.payload.peer_id = Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_string();
        wrong.signature = key
            .sign(&serde_json::to_vec(&wrong.payload).unwrap())
            .unwrap();
        assert!(store.put(record(&wrong)).is_err());
        assert!(
            store
                .put(Record::new(RecordKey::new(&"arbitrary-key"), vec![0; 9000]))
                .is_err()
        );
    }
    let store = SqliteStore::open(&dir.join("records.sqlite"), peer).unwrap();
    assert_eq!(store.get(&r.key).unwrap().value, r.value);
    assert_eq!(store.providers(&model).len(), 1);
    drop(store);
    std::fs::remove_dir_all(dir).unwrap();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn three_nodes_discover_signed_provider_through_bootstrap_and_keep_identity() {
    let root = temp();
    let listen = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
    let seed = dht::start(root.join("seed"), listen, vec![]).await.unwrap();
    let seed_address: libp2p::Multiaddr = seed.handle.snapshot().await.address.parse().unwrap();
    let provider = dht::start(
        root.join("provider"),
        "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
        vec![seed_address.clone()],
    )
    .await
    .unwrap();
    let before = provider.handle.snapshot().await.peer_id;
    assert!(
        dht::start(
            root.join("provider"),
            "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
            vec![]
        )
        .await
        .is_err()
    );
    provider
        .handle
        .publish(Offer {
            model_hash: "a".repeat(64),
            start: 12,
            end: 24,
        })
        .await
        .unwrap();
    let seeker = dht::start(
        root.join("seeker"),
        "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
        vec![seed_address],
    )
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            seeker.handle.find("a".repeat(64)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(400)).await;
            let found = seeker.handle.snapshot().await;
            if found
                .discoveries
                .iter()
                .any(|p| p.peer_id == before && p.start == 12 && p.end == 24)
            {
                break;
            }
        }
    })
    .await
    .expect("provider was not discovered through the bootstrap node");
    assert!(seed.handle.snapshot().await.stored_records > 0);
    drop(provider);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let restarted = dht::start(
        root.join("provider"),
        "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
        vec![],
    )
    .await
    .unwrap();
    assert_eq!(restarted.handle.snapshot().await.peer_id, before);
    drop((seed, seeker, restarted));
    tokio::time::sleep(Duration::from_millis(100)).await;
    std::fs::remove_dir_all(root).unwrap();
}
