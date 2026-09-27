use sangama::{protocol::*, server};
use std::{
    net::TcpListener,
    process::{Child, Command, Stdio},
    time::Duration,
};

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start(args: &[&str]) -> Process {
    Process(
        Command::new(env!("CARGO_BIN_EXE_sangama"))
            .args(args)
            .env("P2P_TOKEN", "process-test-token-only")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    )
}

#[tokio::test]
async fn standalone_coordinator_workers_and_bench_interoperate() {
    let reservations: Vec<_> = (0..3)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addresses: Vec<_> = reservations
        .iter()
        .map(|l| l.local_addr().unwrap())
        .collect();
    let mut reservations = reservations.into_iter();
    drop(reservations.next());
    let coordinator = addresses[0].to_string();
    let _coordinator = start(&["coordinator", "--listen", &coordinator]);
    let http = server::client().unwrap();
    let mut ready = false;
    for _ in 0..100 {
        if http
            .get(url(addresses[0], "/health"))
            .bearer_auth("process-test-token-only")
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(ready, "coordinator did not start");
    drop(reservations.next());
    let _a = start(&[
        "worker",
        "--id",
        "process-a",
        "--listen",
        &addresses[1].to_string(),
        "--coordinator",
        &coordinator,
        "--start",
        "0",
        "--end",
        "6",
    ]);
    drop(reservations.next());
    let _b = start(&[
        "worker",
        "--id",
        "process-b",
        "--listen",
        &addresses[2].to_string(),
        "--coordinator",
        &coordinator,
        "--start",
        "6",
        "--end",
        "12",
    ]);
    let mut registered = false;
    for _ in 0..100 {
        let response = http
            .get(url(addresses[0], "/v1/workers"))
            .bearer_auth("process-test-token-only")
            .send()
            .await
            .unwrap();
        let peers: Vec<Peer> = server::decode(response).await.unwrap();
        if peers.len() == 2 {
            registered = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(registered, "workers did not register");
    let output = Command::new(env!("CARGO_BIN_EXE_sangama"))
        .args(["bench", "--coordinator", &coordinator, "--rounds", "2"])
        .env("P2P_TOKEN", "process-test-token-only")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["last_route"].as_array().unwrap().len(), 2);
    assert!(report["maximum_absolute_error"].as_f64().unwrap() <= 1e-5);
}
