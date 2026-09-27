use p2p_inference::{
    kernel::{Shard, fixture_input},
    planner::plan,
    protocol::*,
};

fn peer(id: &str, start: usize, end: usize, cost: f64, model: &ModelSpec) -> Peer {
    Peer {
        worker: WorkerInfo {
            protocol: PROTOCOL_VERSION,
            id: id.into(),
            address: "127.0.0.1:9001".parse().unwrap(),
            model: model.clone(),
            start,
            end,
            weight_bytes: (end - start) * model.width * model.width * 4,
            estimated_compute_ms: cost,
            simulated_delay_ms: 0,
        },
        age_seconds: 0.0,
        probe_ms: 0.0,
    }
}

#[test]
fn shard_boundaries_preserve_numerical_result() {
    let model = ModelSpec {
        layers: 12,
        width: 32,
    };
    let input = fixture_input(model.width);
    let expected = Shard::new(model.clone(), 0, 12, 16)
        .unwrap()
        .forward(input.clone())
        .unwrap();
    for split in 1..model.layers {
        let a = Shard::new(model.clone(), 0, split, 16).unwrap();
        let b = Shard::new(model.clone(), split, 12, 16).unwrap();
        let actual = b.forward(a.forward(input.clone()).unwrap()).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn allocation_and_activation_validation_happen_before_execution() {
    assert!(
        Shard::new(
            ModelSpec {
                layers: 64,
                width: 1024
            },
            0,
            64,
            1
        )
        .is_err()
    );
    assert!(
        Shard::new(
            ModelSpec {
                layers: usize::MAX,
                width: usize::MAX
            },
            0,
            1,
            64
        )
        .is_err()
    );
    let shard = Shard::new(
        ModelSpec {
            layers: 1,
            width: 8,
        },
        0,
        1,
        1,
    )
    .unwrap();
    assert!(shard.forward(vec![0.0; 7]).is_err());
    assert!(shard.forward(vec![f32::NAN; 8]).is_err());
}

#[test]
fn planner_finds_fast_complete_coverage_and_rejects_gaps() {
    let model = ModelSpec {
        layers: 6,
        width: 8,
    };
    let peers = vec![
        peer("all-slow", 0, 6, 100.0, &model),
        peer("a", 0, 2, 1.0, &model),
        peer("b", 2, 6, 1.0, &model),
    ];
    let route = plan(&model, &peers).unwrap();
    assert_eq!(
        route.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert!(plan(&model, &peers[1..2]).is_err());
    let other = ModelSpec {
        layers: 6,
        width: 16,
    };
    assert!(plan(&other, &peers).is_err());
}

#[test]
fn planner_retains_higher_cost_paths_that_fit_the_hop_limit() {
    let model = ModelSpec {
        layers: 33,
        width: 8,
    };
    let mut peers: Vec<_> = (0..33)
        .map(|i| peer(&format!("s{i}"), i, i + 1, 0.1, &model))
        .collect();
    peers.push(peer("wide", 0, 32, 100.0, &model));
    let route = plan(&model, &peers).unwrap();
    assert_eq!(route.len(), 2);
    assert_eq!(route[0].id, "wide");
}

#[test]
fn only_private_numeric_worker_addresses_are_accepted() {
    assert!(validate_address("127.0.0.1:1".parse().unwrap()).is_ok());
    assert!(validate_address("192.168.1.2:9001".parse().unwrap()).is_ok());
    assert!(validate_address("[::1]:9001".parse().unwrap()).is_ok());
    assert!(validate_address("8.8.8.8:80".parse().unwrap()).is_err());
    assert!(validate_address("169.254.169.254:80".parse().unwrap()).is_err());
    assert!(validate_address("0.0.0.0:9001".parse().unwrap()).is_err());
    assert!(validate_address("127.0.0.1:0".parse().unwrap()).is_err());
}
