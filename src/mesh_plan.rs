//! Select a ready route for the pinned Qwen manifest. Actual allocations remain
//! on the worker; this planner never assumes that advertised memory is available.
use crate::qwen::{load_manifest, network::Info};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::Path, time::Instant};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub address: SocketAddr,
    pub info: Info,
    pub probe_ms: f64,
}
#[derive(Debug, Serialize)]
pub struct Plan {
    pub model_hash: String,
    pub peers: Vec<SocketAddr>,
    pub candidates: Vec<Candidate>,
    pub estimated_route_probe_ms: f64,
    pub reservation_required: bool,
}
pub fn select(
    hash: &str,
    manifest: &crate::qwen::Manifest,
    candidates: &[Candidate],
) -> Result<Plan> {
    let mut peers = Vec::new();
    let mut cost = 0.0;
    for shard in &manifest.shards {
        let selected = candidates
            .iter()
            .filter(|c| {
                c.info.model_hash == hash
                    && c.info.model_id == manifest.model_id
                    && !c.info.busy
                    && c.info.precision == "f32"
                    && c.info.shard.index == shard.index
                    && c.info.shard.start == shard.start
                    && c.info.shard.end == shard.end
                    && c.info.shard.sha256 == shard.sha256
                    && c.probe_ms.is_finite()
                    && c.probe_ms >= 0.0
            })
            .min_by(|a, b| a.probe_ms.total_cmp(&b.probe_ms))
            .ok_or_else(|| {
                anyhow::anyhow!("no ready verified candidate for shard {}", shard.index)
            })?;
        ensure!(
            !peers.contains(&selected.address),
            "one worker cannot cover two shards"
        );
        peers.push(selected.address);
        cost += selected.probe_ms;
    }
    Ok(Plan {
        model_hash: hash.into(),
        peers,
        candidates: candidates.to_vec(),
        estimated_route_probe_ms: cost,
        reservation_required: true,
    })
}
pub async fn probe(dir: &Path, addresses: &[SocketAddr], token: &str) -> Result<Plan> {
    ensure!(
        !addresses.is_empty() && addresses.len() <= 32,
        "expected 1..32 candidate bridges"
    );
    let (manifest, hash) = load_manifest(dir)?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let mut candidates = Vec::new();
    for address in addresses {
        crate::security::loopback(*address)?;
        let start = Instant::now();
        if let Ok(response) = http
            .get(format!("http://{address}/v1/qwen/info"))
            .bearer_auth(token)
            .send()
            .await
            && response.status().is_success()
        {
            let body = response.bytes().await?;
            ensure!(body.len() <= 8192, "oversized worker info");
            if let Ok(info) = serde_json::from_slice(&body) {
                candidates.push(Candidate {
                    address: *address,
                    info,
                    probe_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
            }
        }
    }
    select(&hash, &manifest, &candidates)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_fast_ready_exact_coverage_and_rejects_gaps() {
        let spec = crate::qwen::ShardSpec {
            index: 0,
            start: 0,
            end: 24,
            file: "shard".into(),
            sha256: "abc".into(),
            file_bytes: 1,
            tensor_count: 1,
        };
        let manifest = crate::qwen::Manifest {
            model_id: "model".into(),
            revision: "r".into(),
            weights_sha256: "w".into(),
            config_sha256: "c".into(),
            tokenizer_sha256: "t".into(),
            shards: vec![spec.clone()],
            sliced: None,
        };
        let info = Info {
            model_id: "model".into(),
            model_hash: "hash".into(),
            shard: spec,
            device: "cpu".into(),
            engine: "candle".into(),
            weights_sha256: None,
            precision: "f32".into(),
            pid: 1,
            busy: false,
            memory: None,
        };
        let slow = Candidate {
            address: "127.0.0.1:1".parse().unwrap(),
            info: info.clone(),
            probe_ms: 50.0,
        };
        let fast = Candidate {
            address: "127.0.0.1:2".parse().unwrap(),
            info,
            probe_ms: 5.0,
        };
        assert_eq!(
            select("hash", &manifest, &[slow.clone(), fast.clone()])
                .unwrap()
                .peers,
            vec![fast.address]
        );
        let mut busy = fast.clone();
        busy.info.busy = true;
        assert_eq!(
            select("hash", &manifest, &[slow.clone(), busy])
                .unwrap()
                .peers,
            vec![slow.address]
        );
        let mut wrong = fast;
        wrong.info.shard.sha256 = "wrong".into();
        assert!(select("hash", &manifest, &[wrong]).is_err());
        assert!(select("hash", &manifest, &[]).is_err());
    }
}
