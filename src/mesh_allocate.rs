//! Two-phase placement for prepared Qwen shards: measure, reserve all, load, release.
use crate::managed_worker::{Assignment, Capacity};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    path::Path,
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub address: SocketAddr,
    pub capacity: Capacity,
    pub probe_ms: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub shard: usize,
    pub candidate: Candidate,
}
pub fn select(hash: &str, count: usize, candidates: &[Candidate]) -> Result<Vec<Placement>> {
    ensure!(
        (1..=8).contains(&count) && candidates.len() <= 32,
        "placement size limit"
    );
    let mut best: Vec<Option<(f64, Vec<Placement>)>> = vec![None; 1 << count];
    best[0] = Some((0.0, vec![]));
    let mut seen = std::collections::HashSet::new();
    for c in candidates {
        if c.capacity.model_hash != hash
            || c.capacity.busy
            || !c.probe_ms.is_finite()
            || c.probe_ms < 0.0
            || !seen.insert(&c.capacity.peer)
        {
            continue;
        }
        // Read the previous iteration so one physical peer cannot receive two shards.
        let previous = best.clone();
        for (mask, entry) in previous.into_iter().enumerate() {
            let Some((cost, route)) = entry else {
                continue;
            };
            for s in &c.capacity.shards {
                if s.index >= count
                    || mask & (1 << s.index) != 0
                    || (c.capacity.loaded_shard != Some(s.index)
                        && s.required_bytes > c.capacity.budget_bytes)
                {
                    continue;
                }
                let next = mask | (1 << s.index);
                let score = cost + c.probe_ms;
                if best[next].as_ref().is_none_or(|(old, _)| score < *old) {
                    let mut route = route.clone();
                    route.push(Placement {
                        shard: s.index,
                        candidate: c.clone(),
                    });
                    best[next] = Some((score, route));
                }
            }
        }
    }
    let mut route = best
        .last_mut()
        .unwrap()
        .take()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no complete placement fits available prepared shards and memory budgets"
            )
        })?
        .1;
    route.sort_by_key(|p| p.shard);
    Ok(route)
}
pub async fn allocate(dir: &Path, addresses: &[SocketAddr], token: &str) -> Result<Vec<Placement>> {
    ensure!(
        !addresses.is_empty() && addresses.len() <= 32,
        "expected 1..32 candidates"
    );
    let (manifest, hash) = crate::qwen::load_manifest(dir)?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(55))
        .build()?;
    let mut candidates = Vec::new();
    for address in addresses {
        crate::security::loopback(*address)?;
        let start = Instant::now();
        if let Ok(r) = http
            .get(format!("http://{address}/v1/node/capacity"))
            .bearer_auth(token)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            && r.status().is_success()
        {
            let capacity: Capacity = crate::server::decode(r).await?;
            candidates.push(Candidate {
                address: *address,
                capacity,
                probe_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        }
    }
    let route = select(&hash, manifest.shards.len(), &candidates)?;
    let lease = uuid::Uuid::new_v4().to_string();
    let mut acquired = Vec::new();
    let result: Result<()> = async {
        for p in &route {
            let a = Assignment {
                lease: lease.clone(),
                model_hash: hash.clone(),
                shard: p.shard,
            };
            http.post(format!("http://{}/v1/node/reserve", p.candidate.address))
                .bearer_auth(token)
                .json(&a)
                .send()
                .await?
                .error_for_status()?;
            acquired.push((p.candidate.address, a));
        }
        for (address, a) in &acquired {
            let r = http
                .post(format!("http://{address}/v1/node/load"))
                .bearer_auth(token)
                .json(a)
                .send()
                .await?
                .error_for_status()?;
            let value: serde_json::Value = crate::server::decode(r).await?;
            let info: crate::qwen::network::Info = serde_json::from_value(value["info"].clone())?;
            ensure!(
                info.model_hash == hash
                    && info.shard.index == a.shard
                    && info.shard.sha256 == manifest.shards[a.shard].sha256
                    && info.shard.start == manifest.shards[a.shard].start
                    && info.shard.end == manifest.shards[a.shard].end,
                "worker loaded an unexpected shard"
            );
        }
        Ok(())
    }
    .await;
    // Best effort release; unreachable reservations expire independently on each worker.
    let mut release_failed = false;
    for (address, a) in &acquired {
        let released = http
            .post(format!("http://{address}/v1/node/release"))
            .bearer_auth(token)
            .json(a)
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        release_failed |= !released.is_ok_and(|r| r.status().is_success());
    }
    result?;
    ensure!(
        !release_failed,
        "placement completed but lease release failed; retry after leases expire"
    );
    Ok(route)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn candidate(peer: &str, indexes: &[usize], budget: u64, ms: f64) -> Candidate {
        Candidate {
            address: "127.0.0.1:1".parse().unwrap(),
            probe_ms: ms,
            capacity: Capacity {
                peer: peer.into(),
                model_hash: "hash".into(),
                device: "cpu".into(),
                available_bytes: budget,
                budget_bytes: budget,
                shards: indexes
                    .iter()
                    .map(|i| crate::managed_worker::ShardCapacity {
                        index: *i,
                        required_bytes: 100,
                    })
                    .collect(),
                loaded_shard: None,
                busy: false,
            },
        }
    }
    #[test]
    fn fits_memory_and_preserves_scarce_shard_coverage() {
        let peers = vec![
            candidate("both", &[0, 1], 100, 1.0),
            candidate("first", &[0], 100, 2.0),
            candidate("small", &[1], 99, 0.1),
        ];
        let p = select("hash", 2, &peers).unwrap();
        assert_eq!(p[0].candidate.capacity.peer, "first");
        assert_eq!(p[1].candidate.capacity.peer, "both");
        assert!(select("hash", 2, &peers[..1]).is_err());
        assert!(select("hash", 1, &[candidate("small", &[0], 99, 1.0)]).is_err());
        let mut busy = peers[0].clone();
        busy.capacity.busy = true;
        assert!(select("hash", 2, &[busy, peers[1].clone()]).is_err());
    }
}
