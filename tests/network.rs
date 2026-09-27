use sangama::{
    benchmark,
    kernel::{Shard, fixture_input},
    protocol::*,
    server::{self, WorkerConfig},
};
use tokio::net::TcpListener;

const TOKEN: &str = "integration-test-token-only";

async fn coordinator() -> server::Service {
    server::coordinator(
        TcpListener::bind("127.0.0.1:0").await.unwrap(),
        TOKEN.into(),
    )
    .await
    .unwrap()
}

async fn worker(
    coordinator: &server::Service,
    id: &str,
    start: usize,
    end: usize,
) -> server::Service {
    server::worker(
        TcpListener::bind("127.0.0.1:0").await.unwrap(),
        WorkerConfig {
            id: id.into(),
            advertise: None,
            coordinator: coordinator.address,
            token: TOKEN.into(),
            shard: Shard::new(
                ModelSpec {
                    layers: 4,
                    width: 16,
                },
                start,
                end,
                1,
            )
            .unwrap(),
            delay_ms: 0,
        },
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn distributed_pipeline_matches_local_over_real_sockets() {
    let coordinator = coordinator().await;
    let _a = worker(&coordinator, "a", 0, 2).await;
    let _b = worker(&coordinator, "b", 2, 4).await;
    let result = benchmark::measure(
        coordinator.address,
        TOKEN,
        ModelSpec {
            layers: 4,
            width: 16,
        },
        3,
    )
    .await
    .unwrap();
    assert!(result.maximum_absolute_error <= 1e-5);
    assert_eq!(
        result
            .last_route
            .iter()
            .map(|x| x.worker_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert!(result.distributed_p95_ms >= result.distributed_p50_ms);
}

#[tokio::test]
async fn unauthorized_requests_and_incomplete_routes_fail_explicitly() {
    let coordinator = coordinator().await;
    let client = server::client().unwrap();
    let response = client
        .get(url(coordinator.address, "/v1/workers"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    let _worker = worker(&coordinator, "a", 0, 2).await;
    let response = client
        .post(url(coordinator.address, "/v1/run"))
        .bearer_auth(TOKEN)
        .json(&RunRequest {
            model: ModelSpec {
                layers: 4,
                width: 16,
            },
            activation: fixture_input(16),
        })
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    assert!(response.text().await.unwrap().contains("no live route"));
}

#[tokio::test]
async fn workers_reject_overlapping_routes_and_invalid_activations() {
    let coordinator = coordinator().await;
    let worker = worker(&coordinator, "a", 0, 4).await;
    let client = server::client().unwrap();
    let response = client
        .get(url(worker.address, "/v1/info"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    let info: WorkerInfo = server::decode(response).await.unwrap();
    let response = client
        .post(url(worker.address, "/v1/execute"))
        .bearer_auth(TOKEN)
        .json(&ExecuteRequest {
            model: info.model.clone(),
            activation: fixture_input(16),
            route: vec![info.clone(), info.clone()],
        })
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let response = client
        .post(url(worker.address, "/v1/execute"))
        .bearer_auth(TOKEN)
        .json(&ExecuteRequest {
            model: info.model.clone(),
            activation: fixture_input(8),
            route: vec![info],
        })
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn worker_departure_is_reported_as_failure_not_a_fake_result() {
    let coordinator = coordinator().await;
    let a = worker(&coordinator, "a", 0, 4).await;
    drop(a);
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let result = benchmark::measure(
        coordinator.address,
        TOKEN,
        ModelSpec {
            layers: 4,
            width: 16,
        },
        1,
    )
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn demo_validates_inputs_and_measures_injected_delay() {
    let model = ModelSpec {
        layers: 4,
        width: 16,
    };
    assert!(benchmark::demo(model.clone(), 0, 1, 0).await.is_err());
    assert!(benchmark::demo(model.clone(), 2, 0, 0).await.is_err());
    let result = benchmark::demo(model, 2, 2, 5).await.unwrap();
    assert_eq!(result.mean_injected_delay_ms, 10.0);
    assert!(result.distributed_p50_ms >= 10.0);
}
