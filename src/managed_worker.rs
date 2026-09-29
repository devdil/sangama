//! Opt-in remote placement of prepared, hash-verified Qwen shards.
use crate::{
    qwen::{Manifest, network::Info},
    resources,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    path::PathBuf,
    time::{Duration, Instant},
};
use tokio::{
    process::{Child, Command},
    sync::Mutex,
};
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model_dir: PathBuf,
    pub device: String,
    pub memory_budget_mib: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardCapacity {
    pub index: usize,
    pub required_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capacity {
    pub peer: String,
    pub model_hash: String,
    pub device: String,
    pub available_bytes: u64,
    pub budget_bytes: u64,
    pub shards: Vec<ShardCapacity>,
    pub loaded_shard: Option<usize>,
    pub busy: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub lease: String,
    pub model_hash: String,
    pub shard: usize,
}
struct State {
    lease: Option<(Assignment, Instant)>,
    child: Option<Child>,
    loaded: Option<usize>,
}
pub struct Manager {
    config: Config,
    manifest: Manifest,
    hash: String,
    peer: String,
    address: SocketAddr,
    token: String,
    token_file: PathBuf,
    next: Vec<SocketAddr>,
    http: reqwest::Client,
    state: Mutex<State>,
}
impl Manager {
    pub fn new(
        config: Config,
        peer: String,
        address: SocketAddr,
        token: String,
        token_file: PathBuf,
        next: Vec<SocketAddr>,
    ) -> Result<Self> {
        ensure!(
            crate::qwen::DEVICES.contains(&config.device.as_str()),
            "unsupported managed device"
        );
        let (manifest, hash) = crate::qwen::load_manifest(&config.model_dir)?;
        Ok(Self {
            config,
            manifest,
            hash,
            peer,
            address,
            token,
            token_file,
            next,
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(3))
                .build()?,
            state: Mutex::new(State {
                lease: None,
                child: None,
                loaded: None,
            }),
        })
    }
    async fn info(&self) -> Result<Info> {
        Ok(self
            .http
            .get(format!("http://{}/v1/qwen/info", self.address))
            .bearer_auth(&self.token)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    fn capacity(&self, state: &State, busy: bool) -> Result<Capacity> {
        let available =
            resources::available_for(&self.config.device).context("cannot measure memory")?;
        let budget = self
            .config
            .memory_budget_mib
            .map(|m| m.saturating_mul(1024 * 1024))
            .unwrap_or(available / 5 * 4)
            .min(available);
        let shards = self
            .manifest
            .shards
            .iter()
            .filter(|s| self.config.model_dir.join(&s.file).is_file())
            .map(|s| ShardCapacity {
                index: s.index,
                required_bytes: resources::estimate(s.file_bytes, s.end - s.start),
            })
            .collect();
        Ok(Capacity {
            peer: self.peer.clone(),
            model_hash: self.hash.clone(),
            device: self.config.device.clone(),
            available_bytes: available,
            budget_bytes: budget,
            shards,
            loaded_shard: state.loaded,
            busy,
        })
    }
    async fn reset_lease(&self, lease: &str) {
        let _ = self
            .http
            .post(format!("http://{}/v1/qwen/reset", self.address))
            .bearer_auth(&self.token)
            .json(&serde_json::json!({"session":lease}))
            .send()
            .await;
    }
    async fn expire(&self, state: &mut State) {
        if state
            .lease
            .as_ref()
            .is_some_and(|(_, t)| t.elapsed() > Duration::from_secs(120))
            && let Some((a, _)) = state.lease.take()
        {
            self.reset_lease(&a.lease).await;
        }
    }
    pub async fn ready(&self) -> bool {
        let Ok(mut state) = self.state.try_lock() else {
            return false;
        };
        self.expire(&mut state).await;
        state.loaded.is_some() && state.lease.is_none()
    }
    async fn claim_worker(&self, lease: &str) -> Result<()> {
        self.http
            .post(format!("http://{}/v1/qwen/reserve", self.address))
            .bearer_auth(&self.token)
            .json(&serde_json::json!({"session":lease}))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
    pub async fn handle(&self, path: &str, body: &[u8]) -> Result<serde_json::Value> {
        ensure!(body.len() <= 8192, "placement request too large");
        let mut state = self.state.try_lock().context("placement operation busy")?;
        if state
            .child
            .as_mut()
            .is_some_and(|c| c.try_wait().ok().flatten().is_some())
        {
            state.child = None;
            state.loaded = None;
        }
        self.expire(&mut state).await;
        let busy = if state.loaded.is_some() {
            self.info().await.map(|i| i.busy).unwrap_or(true)
        } else {
            false
        };
        if path == "/v1/node/capacity" {
            return Ok(serde_json::to_value(
                self.capacity(&state, busy || state.lease.is_some())?,
            )?);
        }
        let request: Assignment = serde_json::from_slice(body)?;
        ensure!(
            uuid::Uuid::parse_str(&request.lease).is_ok()
                && request.model_hash == self.hash
                && request.shard < self.manifest.shards.len(),
            "invalid assignment"
        );
        let owned = state
            .lease
            .as_ref()
            .is_some_and(|(a, _)| a.lease == request.lease && a.shard == request.shard);
        match path {
            "/v1/node/reserve" => {
                ensure!(
                    (!busy || owned) && (state.lease.is_none() || owned),
                    "worker is reserved or serving inference"
                );
                let capacity = self.capacity(&state, false)?;
                let shard = capacity
                    .shards
                    .iter()
                    .find(|s| s.index == request.shard)
                    .context("assigned shard file is not prepared on this node")?;
                ensure!(
                    state.loaded == Some(request.shard)
                        || shard.required_bytes <= capacity.budget_bytes,
                    "insufficient memory budget"
                );
                if state.loaded.is_some() {
                    self.claim_worker(&request.lease).await?;
                }
                state.lease = Some((request, Instant::now()));
                Ok(serde_json::json!({"reserved":true,"lease_seconds":120}))
            }
            "/v1/node/load" => {
                ensure!(owned, "assignment lease unavailable");
                if state.loaded.is_some() {
                    self.claim_worker(&request.lease).await?;
                }
                if state.loaded != Some(request.shard) {
                    if let Some(mut child) = state.child.take() {
                        let _ = child.kill().await;
                        let _ = child.wait().await;
                        state.loaded = None;
                    }
                    let shard = &self.manifest.shards[request.shard];
                    resources::check(
                        shard.file_bytes,
                        shard.end - shard.start,
                        self.config.memory_budget_mib,
                        &self.config.device,
                    )?;
                    let mut cmd = Command::new(std::env::current_exe()?);
                    cmd.arg("--token-file")
                        .arg(&self.token_file)
                        .arg("qwen-worker")
                        .arg("--model-dir")
                        .arg(&self.config.model_dir)
                        .arg("--device")
                        .arg(&self.config.device)
                        .arg("--shard")
                        .arg(request.shard.to_string())
                        .arg("--listen")
                        .arg(self.address.to_string())
                        .kill_on_drop(true);
                    if !self.next.is_empty() {
                        cmd.arg("--allow-next").arg(
                            self.next
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                    }
                    state.child = Some(cmd.spawn()?);
                    let started = Instant::now();
                    loop {
                        if let Ok(info) = self.info().await
                            && Some(info.pid) == state.child.as_ref().and_then(Child::id)
                        {
                            ensure!(
                                info.model_hash == self.hash && info.shard.index == request.shard,
                                "loaded worker identity mismatch"
                            );
                            state.loaded = Some(request.shard);
                            break;
                        }
                        ensure!(
                            state.child.as_mut().unwrap().try_wait()?.is_none(),
                            "managed worker exited during load"
                        );
                        if started.elapsed() > Duration::from_secs(45) {
                            if let Some(mut child) = state.child.take() {
                                let _ = child.kill().await;
                                let _ = child.wait().await;
                            }
                            anyhow::bail!("managed worker load timed out");
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
                self.claim_worker(&request.lease).await?;
                state.lease = Some((request, Instant::now()));
                Ok(serde_json::json!({"loaded":true,"info":self.info().await?}))
            }
            "/v1/node/release" => {
                ensure!(owned, "assignment lease belongs to another request");
                self.reset_lease(&request.lease).await;
                state.lease = None;
                // Keep verified weights resident for subsequent inference; release only the placement lease.
                Ok(serde_json::json!({"released":true}))
            }
            _ => anyhow::bail!("unsupported placement operation"),
        }
    }
}
