use crate::{
    kernel::{Shard, fixture_input},
    protocol::*,
    server::{self, WorkerConfig},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{net::SocketAddr, time::Instant};
use tokio::net::TcpListener;

#[derive(Debug, Serialize)]
pub struct Report {
    pub workload: &'static str,
    pub model: ModelSpec,
    pub samples: usize,
    pub execution_note: &'static str,
    pub local_p50_ms: f64,
    pub distributed_p50_ms: f64,
    pub distributed_p95_ms: f64,
    pub sequential_passes_per_second: f64,
    pub mean_worker_compute_ms: f64,
    pub mean_injected_delay_ms: f64,
    pub mean_remaining_overhead_ms: f64,
    pub maximum_absolute_error: f32,
    pub last_route: Vec<Hop>,
}

pub fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * fraction).ceil() as usize).saturating_sub(1);
    values[index.min(values.len() - 1)]
}

pub async fn measure(
    address: SocketAddr,
    token: &str,
    model: ModelSpec,
    rounds: usize,
) -> Result<Report> {
    validate_address(address)?;
    server::validate_token(token)?;
    ensure!((1..=1000).contains(&rounds), "rounds must be 1..=1000");
    let local = Shard::new(model.clone(), 0, model.layers, 1024)?;
    let input = fixture_input(model.width);
    let expected = local.forward(input.clone())?;
    let http = server::client()?;
    let request = RunRequest {
        model: model.clone(),
        activation: input.clone(),
    };
    let mut local_times = Vec::new();
    let mut distributed_times = Vec::new();
    let mut maximum_error = 0.0_f32;
    let mut compute = 0.0;
    let mut injected = 0.0;
    let mut last_route = vec![];
    // Two warmups are excluded from the reported timings.
    for iteration in 0..rounds + 2 {
        let start = Instant::now();
        std::hint::black_box(local.forward(input.clone())?);
        let local_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let response = http
            .post(url(address, "/v1/run"))
            .bearer_auth(token)
            .json(&request)
            .send()
            .await?;
        let output: RunResponse = server::decode(response).await?;
        let distributed_ms = start.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            output.model_id == model.id() && output.activation.len() == expected.len(),
            "invalid response shape or model"
        );
        let mut next = 0;
        for hop in &output.hops {
            ensure!(
                hop.start == next && hop.end > hop.start && hop.end <= model.layers,
                "invalid response coverage"
            );
            ensure!(
                hop.compute_ms.is_finite() && hop.compute_ms >= 0.0,
                "invalid compute timing"
            );
            next = hop.end;
        }
        ensure!(next == model.layers, "incomplete execution trace");
        for (got, want) in output.activation.iter().zip(&expected) {
            ensure!(got.is_finite(), "non-finite output");
            maximum_error = maximum_error.max((got - want).abs());
        }
        ensure!(
            maximum_error <= 1e-5,
            "distributed result differs from local baseline: {maximum_error}"
        );
        if iteration >= 2 {
            local_times.push(local_ms);
            distributed_times.push(distributed_ms);
            compute += output.hops.iter().map(|h| h.compute_ms).sum::<f64>();
            injected += output
                .hops
                .iter()
                .map(|h| h.simulated_delay_ms as f64)
                .sum::<f64>();
            last_route = output.hops;
        }
    }
    let total_ms: f64 = distributed_times.iter().sum();
    Ok(Report {
        workload: "deterministic dense residual numerical fixture (not an LLM)",
        execution_note: "one activation vector per sequential pass; rates are NOT tokens/second; remaining overhead includes HTTP/JSON, queueing, transport, and timer overshoot",
        model,
        samples: rounds,
        local_p50_ms: percentile(&local_times, 0.5),
        distributed_p50_ms: percentile(&distributed_times, 0.5),
        distributed_p95_ms: percentile(&distributed_times, 0.95),
        sequential_passes_per_second: 1000.0 * rounds as f64 / total_ms,
        mean_worker_compute_ms: compute / rounds as f64,
        mean_injected_delay_ms: injected / rounds as f64,
        mean_remaining_overhead_ms: (total_ms - compute - injected) / rounds as f64,
        maximum_absolute_error: maximum_error,
        last_route,
    })
}

pub async fn demo(
    model: ModelSpec,
    workers: usize,
    rounds: usize,
    delay_ms: u64,
) -> Result<Report> {
    model.validate()?;
    ensure!(
        (1..=16).contains(&workers) && workers <= model.layers,
        "workers must be 1..=16 and no more than the layer count"
    );
    ensure!(delay_ms <= 1000, "delay must be 0..=1000 ms per worker");
    ensure!((1..=1000).contains(&rounds), "rounds must be 1..=1000");
    let token = uuid::Uuid::new_v4().to_string();
    let coordinator =
        server::coordinator(TcpListener::bind("127.0.0.1:0").await?, token.clone()).await?;
    let mut services = Vec::new();
    for i in 0..workers {
        let shard = Shard::new(
            model.clone(),
            i * model.layers / workers,
            (i + 1) * model.layers / workers,
            1024,
        )?;
        services.push(
            server::worker(
                TcpListener::bind("127.0.0.1:0").await?,
                WorkerConfig {
                    id: format!("demo-{}", i + 1),
                    advertise: None,
                    coordinator: coordinator.address,
                    token: token.clone(),
                    shard,
                    delay_ms,
                },
            )
            .await?,
        );
    }
    let report = measure(coordinator.address, &token, model, rounds).await?;
    drop(services);
    drop(coordinator);
    Ok(report)
}
