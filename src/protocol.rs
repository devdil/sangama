use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_HOPS: usize = 32;
pub const LEASE_SECONDS: u64 = 15;

/// A deterministic numerical fixture, NOT a trained language model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub layers: usize,
    pub width: usize,
}

impl ModelSpec {
    pub fn validate(&self) -> Result<()> {
        ensure!((1..=64).contains(&self.layers), "layers must be 1..=64");
        ensure!((8..=1024).contains(&self.width), "width must be 8..=1024");
        Ok(())
    }

    pub fn id(&self) -> String {
        format!("numerical-fixture-v1-{}x{}", self.layers, self.width)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerInfo {
    pub protocol: u32,
    pub id: String,
    pub address: SocketAddr,
    pub model: ModelSpec,
    pub start: usize,
    pub end: usize,
    pub weight_bytes: usize,
    pub estimated_compute_ms: f64,
    pub simulated_delay_ms: u64,
}

impl WorkerInfo {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        ensure!(
            self.protocol == PROTOCOL_VERSION,
            "incompatible protocol version"
        );
        ensure!(
            !self.id.is_empty() && self.id.len() <= 64,
            "invalid worker id"
        );
        ensure!(
            self.start < self.end && self.end <= self.model.layers,
            "invalid layer range"
        );
        ensure!(
            self.estimated_compute_ms.is_finite() && self.estimated_compute_ms >= 0.0,
            "invalid execution estimate"
        );
        ensure!(self.simulated_delay_ms <= 1000, "delay exceeds 1000 ms");
        ensure!(
            self.weight_bytes == (self.end - self.start) * self.model.width * self.model.width * 4,
            "weight size mismatch"
        );
        validate_address(self.address)
    }
}

/// This first transport is intended for a trusted private LAN or VPN only.
pub fn validate_address(address: SocketAddr) -> Result<()> {
    ensure!(address.port() > 0, "advertise a nonzero port");
    let valid = match address.ip() {
        IpAddr::V4(ip) => ip.is_loopback() || ip.is_private(),
        IpAddr::V6(ip) => ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00,
    };
    if !valid {
        bail!("worker addresses must be numeric loopback/private LAN/VPN addresses");
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Peer {
    pub worker: WorkerInfo,
    pub age_seconds: f64,
    pub probe_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    pub model: ModelSpec,
    pub activation: Vec<f32>,
}

impl RunRequest {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        ensure!(
            self.activation.len() == self.model.width,
            "activation width mismatch"
        );
        ensure!(
            self.activation.iter().all(|x| x.is_finite()),
            "activation must be finite"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecuteRequest {
    pub model: ModelSpec,
    pub activation: Vec<f32>,
    pub route: Vec<WorkerInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hop {
    pub worker_id: String,
    pub start: usize,
    pub end: usize,
    pub compute_ms: f64,
    pub simulated_delay_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResponse {
    pub model_id: String,
    pub activation: Vec<f32>,
    pub hops: Vec<Hop>,
}

pub fn url(address: SocketAddr, path: &str) -> String {
    format!("http://{address}{path}")
}
