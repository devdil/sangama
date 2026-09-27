use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use sangama::{
    benchmark,
    kernel::Shard,
    protocol::*,
    server::{self, WorkerConfig},
};
use std::{net::SocketAddr, path::PathBuf};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    version,
    about = "Sangama: distributed inference with Qwen layer sharding and Kademlia discovery."
)]
struct Cli {
    #[arg(long, global = true, env = "P2P_TOKEN", hide_env_values = true)]
    token: Option<String>,
    /// Read a private token file instead of putting a secret in command arguments.
    #[arg(long, global = true, env = "P2P_TOKEN_FILE", conflicts_with = "token")]
    token_file: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct ModelArgs {
    #[arg(long, default_value_t = 12)]
    layers: usize,
    #[arg(long, default_value_t = 128)]
    width: usize,
}
impl ModelArgs {
    fn spec(&self) -> ModelSpec {
        ModelSpec {
            layers: self.layers,
            width: self.width,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Run a persistent encrypted Kademlia discovery node.
    DhtNode {
        #[arg(long, default_value = ".mesh/node")]
        state_dir: PathBuf,
        #[arg(long, default_value = "/ip4/127.0.0.1/tcp/9000")]
        listen: libp2p::Multiaddr,
        #[arg(long)]
        bootstrap: Vec<libp2p::Multiaddr>,
        #[arg(long)]
        model_hash: Option<String>,
        #[arg(long, default_value_t = 0)]
        start: usize,
        #[arg(long, default_value_t = 12)]
        end: usize,
        /// Look up a manifest hash, print verified advertisements, then exit.
        #[arg(long)]
        find_model: Option<String>,
        #[arg(long, default_value_t = 25)]
        wait_seconds: u64,
    },
    /// Open a protected localhost control panel for real Qwen tests.
    Ui {
        #[arg(long, default_value = ".mesh/ui")]
        dht_dir: PathBuf,
        #[arg(long, default_value = "/ip4/127.0.0.1/tcp/0")]
        dht_listen: libp2p::Multiaddr,
        #[arg(long)]
        bootstrap: Vec<libp2p::Multiaddr>,
        #[arg(long, default_value = "127.0.0.1:8088")]
        listen: SocketAddr,
        #[arg(long, default_value = ".models/qwen2.5-0.5b-instruct")]
        model_dir: PathBuf,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "metal"])]
        device: String,
        #[arg(long, value_delimiter = ',')]
        peers: Vec<SocketAddr>,
    },
    /// Compare real Qwen generation against an unsplit baseline using separate workers.
    QwenTest {
        #[arg(long, default_value = ".models/qwen2.5-0.5b-instruct")]
        model_dir: PathBuf,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "metal"])]
        device: String,
        #[arg(
            long,
            default_value = "Explain peer-to-peer computing in one short sentence."
        )]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        /// Existing workers in shard order; omit to launch separate local processes.
        #[arg(long, value_delimiter = ',')]
        peers: Vec<SocketAddr>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Load only one real Qwen weight shard and retain its attention cache.
    QwenWorker {
        #[arg(long, default_value = ".models/qwen2.5-0.5b-instruct")]
        model_dir: PathBuf,
        #[arg(long)]
        shard: usize,
        #[arg(long, default_value = "cpu", value_parser = ["cpu", "metal"])]
        device: String,
        #[arg(long, default_value = "127.0.0.1:7901")]
        listen: SocketAddr,
        /// Explicit permitted downstream loopback/tunnel endpoints.
        #[arg(long, value_delimiter = ',')]
        allow_next: Vec<SocketAddr>,
    },
    /// Run coordinator + workers on ephemeral loopback ports; verify and benchmark.
    Demo {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 3)]
        workers: usize,
        #[arg(long, default_value_t = 20)]
        rounds: usize,
        /// Artificial wait once per worker per pass. Not a WAN/network simulator.
        #[arg(long, default_value_t = 0)]
        delay_ms: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run an authenticated registry and planner. Loopback by default.
    Coordinator {
        #[arg(long, default_value = "127.0.0.1:7800")]
        listen: SocketAddr,
    },
    /// Host a resident fixture shard and register its contiguous layer range.
    Worker {
        #[arg(long)]
        id: String,
        #[arg(long, default_value = "127.0.0.1:7801")]
        listen: SocketAddr,
        /// Routable private address; required when binding to 0.0.0.0.
        #[arg(long)]
        advertise: Option<SocketAddr>,
        #[arg(long, default_value = "127.0.0.1:7800")]
        coordinator: SocketAddr,
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        start: usize,
        /// Exclusive layer boundary.
        #[arg(long)]
        end: usize,
        #[arg(long, default_value_t = 64)]
        weight_budget_mib: usize,
        #[arg(long, default_value_t = 0)]
        delay_ms: u64,
    },
    /// List live workers. Expired leases disappear after 15 seconds.
    Peers {
        #[arg(long, default_value = "127.0.0.1:7800")]
        coordinator: SocketAddr,
    },
    /// Inspect a complete route selected by the planner.
    Plan {
        #[arg(long, default_value = "127.0.0.1:7800")]
        coordinator: SocketAddr,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Benchmark registered workers and check against local numerical execution.
    Bench {
        #[arg(long, default_value = "127.0.0.1:7800")]
        coordinator: SocketAddr,
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 20)]
        rounds: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Print a non-sensitive host summary. Does not change system settings.
    Doctor,
}

fn token(value: Option<String>) -> Result<String> {
    let value =
        value.context("set P2P_TOKEN (at least 16 characters); demo creates its own token")?;
    server::validate_token(&value)?;
    Ok(value)
}

fn print_json(value: &impl serde::Serialize, output: Option<PathBuf>) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, format!("{json}\n"))?;
        tracing::info!(path = %path.display(), "report written");
    }
    println!("{json}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let mut cli = Cli::parse();
    if let Some(path) = &cli.token_file {
        cli.token = Some(sangama::security::read_token(path)?);
    }
    match cli.command {
        Command::DhtNode {
            state_dir,
            listen,
            bootstrap,
            model_hash,
            start,
            end,
            find_model,
            wait_seconds,
        } => {
            anyhow::ensure!(
                (1..=120).contains(&wait_seconds),
                "wait-seconds must be 1..120"
            );
            let node = sangama::dht::start(state_dir, listen, bootstrap).await?;
            println!("{}", serde_json::to_string(&node.handle.snapshot().await)?);
            if let Some(model_hash) = model_hash {
                node.handle
                    .publish(sangama::dht::Offer {
                        model_hash,
                        start,
                        end,
                    })
                    .await?;
            }
            if let Some(hash) = find_model {
                let deadline =
                    tokio::time::Instant::now() + std::time::Duration::from_secs(wait_seconds);
                loop {
                    if node.handle.snapshot().await.search != "searching" {
                        node.handle.find(hash.clone()).await?;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    let snapshot = node.handle.snapshot().await;
                    if !snapshot.discoveries.is_empty() {
                        print_json(&snapshot, None)?;
                        break;
                    }
                    anyhow::ensure!(
                        tokio::time::Instant::now() < deadline,
                        "no verified providers found before timeout"
                    );
                }
            } else {
                tokio::signal::ctrl_c().await?;
            }
        }
        Command::Ui {
            dht_dir,
            dht_listen,
            bootstrap,
            listen,
            model_dir,
            device,
            peers,
        } => {
            let node = sangama::dht::start(dht_dir, dht_listen, bootstrap).await?;
            sangama::ui::serve(
                listen,
                model_dir,
                device,
                peers,
                cli.token,
                node.handle.clone(),
            )
            .await?;
        }
        Command::QwenTest {
            model_dir,
            device,
            prompt,
            max_tokens,
            peers,
            output,
        } => {
            let token = if peers.is_empty() {
                cli.token
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
            } else {
                token(cli.token)?
            };
            let report = sangama::qwen::runner::run(sangama::qwen::runner::Options {
                model_dir,
                device,
                prompt,
                max_tokens,
                peers,
                token,
            })
            .await?;
            print_json(&report, output)?;
            anyhow::ensure!(
                report.passed,
                "Qwen distributed verification failed; inspect the report"
            );
        }
        Command::QwenWorker {
            model_dir,
            shard,
            device,
            listen,
            allow_next,
        } => {
            sangama::qwen::network::serve(
                &model_dir,
                shard,
                &device,
                listen,
                token(cli.token)?,
                allow_next,
            )
            .await?;
        }
        Command::Demo {
            model,
            workers,
            rounds,
            delay_ms,
            output,
        } => {
            let report = benchmark::demo(model.spec(), workers, rounds, delay_ms).await?;
            print_json(&report, output)?;
        }
        Command::Coordinator { listen } => {
            let service =
                server::coordinator(TcpListener::bind(listen).await?, token(cli.token)?).await?;
            tracing::info!(address = %service.address, "coordinator ready; Ctrl-C to stop");
            tokio::signal::ctrl_c().await?;
            drop(service);
        }
        Command::Worker {
            id,
            listen,
            advertise,
            coordinator,
            model,
            start,
            end,
            weight_budget_mib,
            delay_ms,
        } => {
            let shard = Shard::new(model.spec(), start, end, weight_budget_mib)?;
            let service = server::worker(
                TcpListener::bind(listen).await?,
                WorkerConfig {
                    id,
                    advertise,
                    coordinator,
                    token: token(cli.token)?,
                    shard,
                    delay_ms,
                },
            )
            .await?;
            tracing::info!(address = %service.address, "worker ready; Ctrl-C to stop");
            tokio::signal::ctrl_c().await?;
            drop(service);
        }
        Command::Peers { coordinator } => {
            validate_address(coordinator)?;
            let response = server::client()?
                .get(url(coordinator, "/v1/workers"))
                .bearer_auth(token(cli.token)?)
                .send()
                .await?;
            let peers: Vec<Peer> = server::decode(response).await?;
            print_json(&peers, None)?;
        }
        Command::Plan { coordinator, model } => {
            validate_address(coordinator)?;
            model.spec().validate()?;
            let response = server::client()?
                .post(url(coordinator, "/v1/plan"))
                .bearer_auth(token(cli.token)?)
                .json(&model.spec())
                .send()
                .await?;
            let route: Vec<WorkerInfo> = server::decode(response).await?;
            print_json(&route, None)?;
        }
        Command::Bench {
            coordinator,
            model,
            rounds,
            output,
        } => {
            let report =
                benchmark::measure(coordinator, &token(cli.token)?, model.spec(), rounds).await?;
            print_json(&report, output)?;
        }
        Command::Doctor => {
            let memory = if cfg!(target_os = "macos") {
                std::process::Command::new("sysctl")
                    .args(["-n", "hw.memsize"])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .and_then(|s| s.trim().parse::<u64>().ok())
            } else {
                None
            };
            print_json(
                &serde_json::json!({
                    "os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
                    "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
                    "physical_memory_bytes": memory,
                    "backends": ["CPU numerical fixture", "Candle Qwen2.5"], "metal_compiled": cfg!(feature = "metal"),
                    "model_download_required": "Qwen: yes; numerical fixture: no",
                    "next_step": "Run demo, or fetch the pinned Qwen checkpoint and run qwen-test"
                }),
                None,
            )?;
        }
    }
    Ok(())
}
