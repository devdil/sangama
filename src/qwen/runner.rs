use super::{
    CONTEXT_LIMIT, Manifest, check_hash, config, device, load_manifest, network::Info, weights,
    wire::*,
};
use crate::{
    protocol::{url, validate_address},
    server,
};
use anyhow::{Context, Result, ensure};
use candle::Tensor;
use candle_transformers::models::qwen2::ModelForCausalLM;
use serde::{Deserialize, Serialize};
use std::{
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

pub struct Options {
    pub model_dir: PathBuf,
    pub device: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub peers: Vec<SocketAddr>,
    pub token: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn format_chat(messages: &[ChatMessage]) -> Result<String> {
    ensure!(
        !messages.is_empty() && messages.len() <= 64,
        "expected 1..64 messages"
    );
    let mut formatted = String::new();
    for message in messages {
        ensure!(
            matches!(message.role.as_str(), "system" | "user" | "assistant"),
            "unsupported message role"
        );
        ensure!(
            !message.content.contains("<|im_") && !message.content.contains("<|endoftext|>"),
            "chat control tokens are forbidden"
        );
        formatted.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role, message.content
        ));
    }
    ensure!(
        messages.iter().any(|message| message.role == "user"),
        "conversation must contain a user message"
    );
    ensure!(formatted.len() <= 128 * 1024, "conversation too large");
    formatted.push_str("<|im_start|>assistant\n");
    Ok(formatted)
}

/// Text-only chat through the same validated shard route. A closed stream cancels at the next token.
pub async fn chat(
    options: Options,
    messages: Vec<ChatMessage>,
    deltas: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<Report> {
    let formatted = format_chat(&messages)?;
    execute_chat(options, false, Some(formatted), deltas).await
}

#[derive(Serialize)]
pub struct Timing {
    pub first_token_ms: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub total_ms: f64,
}
impl Timing {
    fn from_samples(times: &[f64]) -> Self {
        let decode_ms: f64 = times.iter().skip(1).sum();
        Self {
            first_token_ms: times[0],
            decode_tokens_per_second: if times.len() > 1 {
                Some(1000.0 * (times.len() - 1) as f64 / decode_ms)
            } else {
                None
            },
            total_ms: times.iter().sum(),
        }
    }
}

#[derive(Serialize)]
pub struct Report {
    pub operation: &'static str,
    pub finish_reason: &'static str,
    pub model_id: String,
    pub revision: String,
    pub manifest_hash: String,
    pub precision: &'static str,
    pub device: String,
    pub topology: String,
    pub prompt: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub local_text: Option<String>,
    pub distributed_text: String,
    pub local_token_ids: Option<Vec<u32>>,
    pub distributed_token_ids: Vec<u32>,
    pub tokens_match: Option<bool>,
    pub maximum_logit_absolute_error: Option<f32>,
    pub logit_tolerance: Option<f32>,
    pub passed: Option<bool>,
    pub local: Option<Timing>,
    pub distributed: Timing,
    pub workers: Vec<Info>,
    pub last_trace: Vec<Trace>,
    pub notes: Vec<&'static str>,
}

struct Processes(Vec<Child>);
impl Drop for Processes {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct Baseline {
    ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    times: Vec<f64>,
}

fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
        "invalid logits"
    );
    // Use the first maximum, matching argmax for a tied distribution.
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

fn baseline(
    dir: &Path,
    manifest: &Manifest,
    backend: &str,
    prompt: &[u32],
    limit: usize,
) -> Result<Baseline> {
    check_hash(&dir.join("model.safetensors"), &manifest.weights_sha256)?;
    let device = device(backend)?;
    let cfg = config(dir)?;
    let mut model = ModelForCausalLM::new(&cfg, weights(&dir.join("model.safetensors"), &device)?)?;
    // Warm up prompt kernels before timing; reset all KV caches afterward.
    let input = Tensor::new(prompt, &device)?.unsqueeze(0)?;
    let _ = model.forward(&input, 0)?.flatten_all()?.to_vec1::<f32>()?;
    model.clear_kv_cache();
    device.synchronize()?;
    let mut result = Baseline {
        ids: vec![],
        logits: vec![],
        times: vec![],
    };
    let mut position = 0;
    for _ in 0..limit {
        let context = if result.ids.is_empty() {
            prompt
        } else {
            &result.ids[result.ids.len() - 1..]
        };
        let now = Instant::now();
        let input = Tensor::new(context, &device)?.unsqueeze(0)?;
        let logits = model
            .forward(&input, position)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let id = greedy(&logits)?;
        result.times.push(now.elapsed().as_secs_f64() * 1000.0);
        position += context.len();
        result.ids.push(id);
        result.logits.push(logits);
        if id == 151645 || id == 151643 {
            break;
        }
    }
    tracing::info!(
        tokens = result.ids.len(),
        "unsplit Candle baseline complete"
    );
    Ok(result)
}

async fn start_workers(
    dir: &Path,
    manifest: &Manifest,
    backend: &str,
    token: &str,
) -> Result<(Processes, Vec<SocketAddr>)> {
    let reserved: Vec<_> = (0..manifest.shards.len())
        .map(|_| TcpListener::bind("127.0.0.1:0"))
        .collect::<std::io::Result<_>>()?;
    let addresses: Vec<_> = reserved
        .iter()
        .map(|l| l.local_addr())
        .collect::<std::io::Result<_>>()?;
    let logs = PathBuf::from("runs").join(format!("qwen-workers-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&logs)?;
    let mut processes = Processes(vec![]);
    for (index, listener) in reserved.into_iter().enumerate() {
        drop(listener);
        let stderr = std::fs::File::create(logs.join(format!("worker-{index}.log")))?;
        let mut command = Command::new(std::env::current_exe()?);
        command.arg("qwen-worker");
        if let Some(next) = addresses.get(index + 1) {
            command.arg("--allow-next").arg(next.to_string());
        }
        let child = command
            .arg("--model-dir")
            .arg(dir)
            .args([
                "--shard",
                &index.to_string(),
                "--device",
                backend,
                "--listen",
                &addresses[index].to_string(),
            ])
            .env_remove("P2P_TOKEN_FILE")
            .env("P2P_TOKEN", token)
            .env("RUST_LOG", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()?;
        processes.0.push(child);
    }
    let http = server::client()?;
    for address in &addresses {
        let start = Instant::now();
        loop {
            for child in &mut processes.0 {
                if let Some(status) = child.try_wait()? {
                    anyhow::bail!("Qwen worker exited ({status}); see {}", logs.display());
                }
            }
            if http
                .get(url(*address, "/v1/qwen/info"))
                .bearer_auth(token)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                break;
            }
            ensure!(
                start.elapsed() < Duration::from_secs(120),
                "worker startup timed out; see {}",
                logs.display()
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    Ok((processes, addresses))
}

async fn reset(
    http: &reqwest::Client,
    addresses: &[SocketAddr],
    token: &str,
    session: &str,
) -> Result<()> {
    for address in addresses {
        let response = http
            .post(url(*address, "/v1/qwen/reset"))
            .bearer_auth(token)
            .json(&serde_json::json!({"session":session}))
            .send()
            .await?;
        let _: serde_json::Value = server::decode(response).await?;
    }
    Ok(())
}

fn request(
    model_hash: &str,
    session: &str,
    route: &[Endpoint],
    tokens: &[u32],
    position: usize,
    sample: bool,
) -> Frame {
    Frame {
        header: Header {
            protocol: 1,
            sample,
            model_hash: model_hash.into(),
            session: session.into(),
            position,
            seq_len: tokens.len(),
            kind: Kind::Tokens,
            tokens: tokens.to_vec(),
            route: route.to_vec(),
            trace: vec![],
        },
        values: vec![],
    }
}

async fn step(http: &reqwest::Client, token: &str, frame: &Frame) -> Result<Frame> {
    let response = http
        .post(url(frame.header.route[0].address, "/v1/qwen/forward"))
        .bearer_auth(token)
        .header("content-type", "application/octet-stream")
        .body(frame.encode()?)
        .send()
        .await?;
    let response = super::wire::response(response).await?;
    ensure!(
        response.valid_output()
            && response.header.sample == frame.header.sample
            && response.header.model_hash == frame.header.model_hash
            && response.header.session == frame.header.session
            && response.header.position == frame.header.position
            && response.header.seq_len == frame.header.seq_len
            && response.header.route.is_empty()
            && response.header.trace.len() == frame.header.route.len(),
        "invalid final response"
    );
    Ok(response)
}

pub async fn run(options: Options) -> Result<Report> {
    execute(options, true).await
}

/// Generate using shard workers only; never open the complete checkpoint or create a local model.
pub async fn generate(options: Options) -> Result<Report> {
    execute(options, false).await
}

async fn execute(options: Options, verify: bool) -> Result<Report> {
    execute_chat(options, verify, None, None).await
}

async fn execute_chat(
    options: Options,
    verify: bool,
    conversation: Option<String>,
    deltas: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<Report> {
    ensure!(
        (1..=super::OUTPUT_LIMIT).contains(&options.max_tokens),
        "max-tokens must be 1..={}",
        super::OUTPUT_LIMIT
    );
    ensure!(
        !options.prompt.trim().is_empty() && options.prompt.len() <= 16 * 1024,
        "prompt must be 1..=16384 bytes"
    );
    ensure!(
        !options.prompt.contains("<|im_start|>") && !options.prompt.contains("<|im_end|>"),
        "prompt must not contain chat control tokens"
    );
    server::validate_token(&options.token)?;
    for address in &options.peers {
        crate::security::loopback(*address)?;
    }
    let dir = options
        .model_dir
        .canonicalize()
        .context("run python3 scripts/fetch-qwen.py first")?;
    let (manifest, model_hash) = load_manifest(&dir)?;
    check_hash(&dir.join("tokenizer.json"), &manifest.tokenizer_sha256)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let is_chat = conversation.is_some();
    let formatted = conversation.unwrap_or_else(|| format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        options.prompt
    ));
    let prompt = tokenizer
        .encode(formatted, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    ensure!(
        (is_chat || prompt.len() <= 512) && prompt.len() + options.max_tokens <= CONTEXT_LIMIT,
        "prompt/context too long for this prototype"
    );
    let local = if verify {
        let dir = dir.clone();
        let manifest = manifest.clone();
        let backend = options.device.clone();
        let prompt = prompt.clone();
        tracing::info!(device = %backend, "loading real Qwen checkpoint for unsplit baseline");
        Some(
            tokio::task::spawn_blocking(move || {
                baseline(&dir, &manifest, &backend, &prompt, options.max_tokens)
            })
            .await??,
        )
    } else {
        None
    };
    // The full model is dropped before any shard processes are launched.
    let (processes, addresses) = if options.peers.is_empty() {
        let (processes, addresses) =
            start_workers(&dir, &manifest, &options.device, &options.token).await?;
        (Some(processes), addresses)
    } else {
        (None, options.peers.clone())
    };
    ensure!(
        addresses.len() == manifest.shards.len(),
        "supply one peer for each manifest shard, in layer order"
    );
    let http = server::client()?;
    let mut infos = Vec::new();
    for (index, address) in addresses.iter().enumerate() {
        validate_address(*address)?;
        let response = http
            .get(url(*address, "/v1/qwen/info"))
            .bearer_auth(&options.token)
            .send()
            .await?;
        let info: Info = server::decode(response).await?;
        ensure!(
            info.model_hash == model_hash
                && info.shard.index == index
                && info.shard.sha256 == manifest.shards[index].sha256
                && info.shard.start == manifest.shards[index].start
                && info.shard.end == manifest.shards[index].end
                && info.device == options.device
                && info.precision == "f32",
            "peer model/shard/device mismatch"
        );
        infos.push(info);
    }
    let route: Vec<_> = addresses
        .iter()
        .enumerate()
        .map(|(shard, address)| Endpoint {
            shard,
            address: *address,
        })
        .collect();
    let session = uuid::Uuid::new_v4().to_string();
    let measurement: Result<_> = async {
        // Acquire the whole route before advancing any KV cache. Cleanup below releases
        // every successfully acquired lease if a later shard refuses the session.
        for (index, address) in addresses.iter().enumerate() {
            // Name the previous shard so an admitted mesh accepts forwards only from that peer.
            // Workers reached directly on loopback ignore it.
            let mut body = serde_json::json!({"session":session});
            if index > 0 {
                body["upstream"] = addresses[index - 1].to_string().into();
            }
            http.post(url(*address, "/v1/qwen/reserve"))
                .bearer_auth(&options.token)
                .json(&body)
                .send()
                .await?
                .error_for_status()?;
        }
        if verify {
            let _ = step(
                &http,
                &options.token,
                &request(&model_hash, &session, &route, &prompt, 0, false),
            )
            .await?;
            reset(&http, &addresses, &options.token, &session).await?;
        }
        let mut decoder = tokenizer.decode_stream(true);
        let mut ids = Vec::new();
        let mut times = Vec::new();
        let mut position = 0;
        let mut maximum_error = 0.0_f32;
        let mut last_trace = Vec::new();
        let mut finish_reason = "max_tokens";
        let limit = local.as_ref().map_or(options.max_tokens, |b| b.ids.len());
        for index in 0..limit {
            let context = if ids.is_empty() {
                prompt.as_slice()
            } else {
                &ids[ids.len() - 1..]
            };
            let now = Instant::now();
            let mut last = None;
            for chunk in context.chunks(512) {
                if let Some(sender) = &deltas {
                    ensure!(!sender.is_closed(), "client disconnected");
                }
                last = Some(
                    step(
                        &http,
                        &options.token,
                        &request(&model_hash, &session, &route, chunk, position, !verify),
                    )
                    .await?,
                );
                position += chunk.len();
            }
            let result = last.context("empty context")?;
            let id = if result.header.sample {
                result.header.tokens[0]
            } else {
                greedy(&result.values)?
            };
            times.push(now.elapsed().as_secs_f64() * 1000.0);
            last_trace = result.header.trace;
            ids.push(id);
            if let Some(sender) = &deltas
                && let Some(delta) = decoder
                    .step(id)
                    .map_err(|e| anyhow::anyhow!("stream decode: {e}"))?
            {
                sender
                    .send(delta)
                    .await
                    .map_err(|_| anyhow::anyhow!("client disconnected"))?;
            }
            if let Some(local) = &local {
                for (got, expected) in result.values.iter().zip(&local.logits[index]) {
                    maximum_error = maximum_error.max((got - expected).abs());
                }
                if id != local.ids[index] {
                    finish_reason = "verification_mismatch";
                    break;
                }
            }
            if id == 151645 || id == 151643 {
                finish_reason = "eos";
                break;
            }
        }
        Ok((ids, times, maximum_error, last_trace, finish_reason))
    }
    .await;
    let cleanup = reset(&http, &addresses, &options.token, &session).await;
    let (ids, times, maximum_error, trace, finish_reason) = measurement?;
    cleanup?;
    let tokens_match = local.as_ref().map(|local| ids == local.ids);
    let report = Report {
        operation: if verify { "verify" } else { "generate" },
        finish_reason,
        model_id: manifest.model_id,
        revision: manifest.revision,
        manifest_hash: model_hash,
        precision: "BF16 checkpoint converted to F32; not quantized",
        device: options.device,
        topology: if processes.is_some() {
            "separate worker processes over loopback on one physical computer".into()
        } else {
            "explicit supplied peer addresses; physical topology not independently verified".into()
        },
        prompt: options.prompt,
        prompt_tokens: prompt.len(),
        generated_tokens: ids.len(),
        local_text: local
            .as_ref()
            .map(|local| {
                tokenizer
                    .decode(&local.ids, true)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))
            })
            .transpose()?,
        distributed_text: tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?,
        tokens_match,
        maximum_logit_absolute_error: verify.then_some(maximum_error),
        logit_tolerance: verify.then_some(1e-3),
        passed: tokens_match.map(|matched| matched && maximum_error <= 1e-3),
        local: local
            .as_ref()
            .map(|local| Timing::from_samples(&local.times)),
        distributed: Timing::from_samples(&times),
        local_token_ids: local.map(|local| local.ids),
        distributed_token_ids: ids,
        workers: infos,
        last_trace: trace,
        notes: vec![
            if verify {
                "Every distributed logit is compared with upstream Candle's unsplit Qwen implementation until any token divergence."
            } else {
                "Standalone greedy generation: no complete checkpoint is read or loaded on the client. No baseline comparison is performed; verification fields are null."
            },
            if verify {
                "Warm timings: checkpoint loading, worker startup, tokenizer initialization, and one prompt warmup per mode are excluded."
            } else {
                "Generation timing includes the first prompt forward with no extra warmup; worker startup, file loading, and tokenizer initialization are excluded."
            },
            "Decode rate excludes the first generated token; token counts include EOS if emitted. This is one run, not a statistical performance benchmark.",
            "Worker forward_ms includes host/device tensor transfers. Generation samples greedily on the final worker and returns a token; verification returns complete logits.",
            "Tied embeddings are duplicated at the first and last stages. Each worker reads only its physical shard file, not the complete checkpoint.",
            "One active session per worker, 4096-token context cap, fixed routes, loopback HTTP (use SSH tunnels between hosts); no automatic KV failover.",
        ],
    };
    drop(processes);
    Ok(report)
}
