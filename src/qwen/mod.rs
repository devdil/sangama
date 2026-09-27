pub mod model;
pub mod network;
pub mod runner;
pub mod wire;

use anyhow::{Context, Result, ensure};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

pub const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
pub const REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
pub const WEIGHTS_SHA256: &str = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe";
pub const CONTEXT_LIMIT: usize = 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShardSpec {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub file: String,
    pub sha256: String,
    pub file_bytes: u64,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_id: String,
    pub revision: String,
    pub weights_sha256: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub shards: Vec<ShardSpec>,
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut input =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let len = input.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        hasher.update(&buffer[..len]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn check_hash(path: &Path, expected: &str) -> Result<()> {
    ensure!(
        sha256(path)? == expected,
        "SHA256 mismatch for {}",
        path.display()
    );
    Ok(())
}

pub fn load_manifest(dir: &Path) -> Result<(Manifest, String)> {
    let path = dir.join("manifest.json");
    let bytes = std::fs::read(&path).context("run python3 scripts/fetch-qwen.py first")?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.model_id == MODEL_ID
            && manifest.revision == REVISION
            && manifest.weights_sha256 == WEIGHTS_SHA256,
        "unsupported checkpoint/revision"
    );
    ensure!(
        (1..=8).contains(&manifest.shards.len()),
        "expected 1..=8 shards"
    );
    let mut next = 0;
    for (index, shard) in manifest.shards.iter().enumerate() {
        ensure!(
            shard.index == index
                && shard.start == next
                && shard.end > shard.start
                && shard.end <= 24,
            "manifest layer gap/overlap"
        );
        ensure!(
            Path::new(&shard.file).components().count() == 1
                && shard.file.ends_with(".safetensors"),
            "invalid shard filename"
        );
        next = shard.end;
    }
    ensure!(next == 24, "manifest must cover all 24 layers");
    check_hash(&dir.join("config.json"), &manifest.config_sha256)?;
    Ok((manifest, format!("{:x}", Sha256::digest(&bytes))))
}

pub fn config(dir: &Path) -> Result<model::Config> {
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    // Candle expects an integer; this checkpoint disables sliding attention and uses null.
    if value["sliding_window"].is_null() {
        value["sliding_window"] = serde_json::json!(32768);
    }
    let mut cfg: model::Config = serde_json::from_value(value)?;
    ensure!(
        cfg.hidden_size == 896
            && cfg.num_hidden_layers == 24
            && cfg.vocab_size == 151936
            && !cfg.use_sliding_window,
        "unexpected Qwen configuration"
    );
    // Retain exact RoPE values for the supported prefix while bounding KV/session memory.
    cfg.max_position_embeddings = CONTEXT_LIMIT;
    Ok(cfg)
}

pub fn device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0).context("initialize Metal")?)
            }
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("rebuild with --features metal to use your Mac GPU")
            }
        }
        _ => anyhow::bail!("device must be cpu or metal"),
    }
}

pub fn weights(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    // Owned buffers avoid unsafe mmap lifetimes or mutable-on-disk mappings.
    Ok(VarBuilder::from_buffered_safetensors(
        std::fs::read(path)?,
        DType::F32,
        device,
    )?)
}

pub fn default_dir() -> PathBuf {
    PathBuf::from(".models/qwen2.5-0.5b-instruct")
}
