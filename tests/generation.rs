//! Client protocol test. Fake logits isolate stopping/session behavior; real weights are tested separately.
use axum::{
    Json, Router,
    body::Bytes,
    routing::{get, post},
};
use sangama::qwen::{
    self, Manifest, ShardSpec,
    network::Info,
    runner::{self, Options},
    wire::{Frame, Kind, Trace},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[tokio::test]
async fn metadata_only_client_generates_stops_and_resets_without_a_baseline() {
    let dir =
        Scratch(std::env::temp_dir().join(format!("sangama-generation-{}", uuid::Uuid::new_v4())));
    std::fs::create_dir(&dir.0).unwrap();
    std::fs::write(dir.0.join("config.json"), "{}").unwrap();
    let tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [("[UNK]".into(), 0), ("hello".into(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.save(dir.0.join("tokenizer.json"), false).unwrap();
    let shard = ShardSpec {
        index: 0,
        start: 0,
        end: 24,
        file: "absent-shard.safetensors".into(),
        sha256: "a".repeat(64),
        file_bytes: 1,
        tensor_count: 1,
    };
    let manifest = Manifest {
        model_id: qwen::MODEL_ID.into(),
        revision: qwen::REVISION.into(),
        weights_sha256: qwen::WEIGHTS_SHA256.into(),
        config_sha256: qwen::sha256(&dir.0.join("config.json")).unwrap(),
        tokenizer_sha256: qwen::sha256(&dir.0.join("tokenizer.json")).unwrap(),
        shards: vec![shard.clone()],
    };
    std::fs::write(
        dir.0.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let (_, hash) = qwen::load_manifest(&dir.0).unwrap();
    let info = Info {
        busy: false,
        memory: None,
        model_id: qwen::MODEL_ID.into(),
        model_hash: hash,
        shard,
        device: "metal".into(),
        engine: "candle".into(),
        weights_sha256: None,
        precision: "f32".into(),
        pid: 1,
    };
    let seen = Arc::new(Mutex::new(vec![]));
    let steps = seen.clone();
    let resets = Arc::new(AtomicUsize::new(0));
    let cleared = resets.clone();
    let app = Router::new()
        .route(
            "/v1/qwen/reserve",
            post(|| async { Json(serde_json::json!({"reserved":true})) }),
        )
        .route(
            "/v1/qwen/info",
            get(move || {
                let info = info.clone();
                async move { Json(info) }
            }),
        )
        .route(
            "/v1/qwen/reset",
            post(move || {
                let cleared = cleared.clone();
                async move {
                    cleared.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({"reset":true}))
                }
            }),
        )
        .route(
            "/v1/qwen/forward",
            post(move |bytes: Bytes| {
                let steps = steps.clone();
                async move {
                    let mut frame = Frame::decode(&bytes).unwrap();
                    steps
                        .lock()
                        .unwrap()
                        .push((frame.header.position, frame.header.tokens.clone()));
                    let id = if frame.header.position == 0 {
                        1
                    } else {
                        151645
                    };
                    frame.header.route.clear();
                    frame.header.tokens.clear();
                    frame.header.kind = Kind::Logits;
                    frame.header.trace = vec![Trace {
                        shard: 0,
                        start: 0,
                        end: 24,
                        forward_ms: 0.1,
                    }];
                    frame.values = vec![0.; 151936];
                    frame.values[id] = 1.;
                    if frame.header.sample {
                        frame.header.kind = Kind::Sampled;
                        frame.header.tokens = vec![id as u32];
                        frame.values.clear();
                    }
                    frame.encode().unwrap()
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let peer = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let options = |max_tokens| Options {
        model_dir: dir.0.clone(),
        device: "metal".into(),
        prompt: "test".into(),
        max_tokens,
        peers: vec![peer],
        token: "test-only-generation-token".into(),
    };
    let result = runner::generate(options(8)).await.unwrap();
    assert_eq!(result.distributed_text, "hello");
    assert_eq!(result.distributed_token_ids, vec![1, 151645]);
    assert_eq!(result.operation, "generate");
    assert_eq!(result.finish_reason, "eos");
    assert!(
        result.local.is_none()
            && result.passed.is_none()
            && result.maximum_logit_absolute_error.is_none()
    );
    assert_eq!(seen.lock().unwrap().len(), 2, "no duplicate warmup prefill");
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert!(
        !dir.0.join("model.safetensors").exists()
            && !dir.0.join("absent-shard.safetensors").exists()
    );
    // A second session succeeds after cleanup and obeys its token cap.
    let limited = runner::generate(options(1)).await.unwrap();
    assert_eq!(limited.generated_tokens, 1);
    assert_eq!(limited.finish_reason, "max_tokens");
    assert!(limited.distributed.decode_tokens_per_second.is_none());
    assert_eq!(resets.load(Ordering::SeqCst), 2);
    let error = runner::run(options(1)).await.err().unwrap();
    assert!(error.to_string().contains("model.safetensors"));
    server.abort();
}
