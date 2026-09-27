use crate::protocol::{MAX_HOPS, ModelSpec, Peer, WorkerInfo};
use anyhow::{Result, bail};

/// Exact contiguous coverage. Coordinator-to-worker probes are an initial heuristic,
/// not a claim to have measured pairwise link costs.
pub fn plan(model: &ModelSpec, peers: &[Peer]) -> Result<Vec<WorkerInfo>> {
    model.validate()?;
    type Candidate = Option<(f64, Vec<WorkerInfo>)>;
    let mut best: Vec<Vec<Candidate>> = vec![vec![None; MAX_HOPS + 1]; model.layers + 1];
    best[0][0] = Some((0.0, vec![]));
    for position in 0..model.layers {
        for count in 0..MAX_HOPS {
            let Some((base, route)) = best[position][count].clone() else {
                continue;
            };
            for peer in peers {
                let w = &peer.worker;
                if w.model != *model || w.start != position || w.validate().is_err() {
                    continue;
                }
                if !peer.probe_ms.is_finite() || peer.probe_ms < 0.0 {
                    continue;
                }
                let cost = base
                    + w.estimated_compute_ms
                    + w.simulated_delay_ms as f64
                    + peer.probe_ms
                    + 0.05;
                if best[w.end][count + 1]
                    .as_ref()
                    .is_none_or(|(old, _)| cost < *old)
                {
                    let mut next = route.clone();
                    next.push(w.clone());
                    best[w.end][count + 1] = Some((cost, next));
                }
            }
        }
    }
    match best[model.layers]
        .iter_mut()
        .filter_map(Option::take)
        .min_by(|a, b| a.0.total_cmp(&b.0))
    {
        Some((_, route)) => Ok(route),
        None => bail!(
            "no live route covers every layer of {}; start workers covering [0, {}) without gaps",
            model.id(),
            model.layers
        ),
    }
}
