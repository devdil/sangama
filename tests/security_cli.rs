use std::process::Command;

#[test]
fn refuses_nonloopback_before_loading_any_model() {
    for address in ["0.0.0.0:7901", "192.168.1.2:7901", "100.64.1.2:7901"] {
        let output = Command::new(env!("CARGO_BIN_EXE_sangama"))
            .env_remove("P2P_TOKEN_FILE")
            .env("P2P_TOKEN", "test-only-not-a-real-token")
            .args([
                "qwen-worker",
                "--shard",
                "0",
                "--listen",
                address,
                "--model-dir",
                "/nonexistent-model",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("loopback"));
    }
    let output = Command::new(env!("CARGO_BIN_EXE_sangama"))
        .env_remove("P2P_TOKEN_FILE")
        .env("P2P_TOKEN", "test-only-not-a-real-token")
        .args([
            "qwen-test",
            "--peers",
            "192.168.1.2:7901",
            "--model-dir",
            "/nonexistent-model",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("loopback"));
}
