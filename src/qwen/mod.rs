pub mod engine;
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
/// Decoder layers in the supported checkpoint.
pub const LAYERS: usize = 24;
pub const CONTEXT_LIMIT: usize = 4096;
/// Execution backends a worker may use. Workers in one route may use different backends.
/// Vulkan and ROCm are available only with the llama.cpp engine.
pub const DEVICES: [&str; 5] = ["cpu", "metal", "cuda", "vulkan", "rocm"];
/// Largest completion one request may ask for; prompt plus output must still fit CONTEXT_LIMIT.
pub const OUTPUT_LIMIT: usize = 512;

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
    /// Present for a model published as per-layer GGUF slices; absent for the pinned Qwen2.5
    /// checkpoint, whose shape is fixed below.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliced: Option<SlicedModel>,
}

/// A model published as per-layer GGUF slices (scripts/split-gguf.py) and run with llama.cpp.
/// Each shard's `sha256` is the digest of its slices' checksums (scripts/prepare-stage.py), so a
/// stage is pinned to exact slice files just as a Candle shard is pinned to its file.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SlicedModel {
    pub architecture: String,
    pub layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    /// Tokens that end generation, by their text in the tokenizer.
    pub eos_tokens: Vec<String>,
    /// Text after the assistant header, e.g. `<think>\n\n</think>\n\n` to turn off Qwen3.5
    /// thinking.
    #[serde(default)]
    pub assistant_prefix: String,
    pub slices_repo: String,
    pub slices_revision: String,
    pub slices_manifest_sha256: String,
}

impl Manifest {
    pub fn layers(&self) -> usize {
        self.sliced.as_ref().map_or(LAYERS, |m| m.layers)
    }
    pub fn hidden_size(&self) -> usize {
        self.sliced.as_ref().map_or(896, |m| m.hidden_size)
    }
    pub fn vocab_size(&self) -> usize {
        self.sliced.as_ref().map_or(151936, |m| m.vocab_size)
    }
    pub fn eos_tokens(&self) -> Vec<&str> {
        self.sliced.as_ref().map_or_else(
            || vec!["<|im_end|>", "<|endoftext|>"],
            |m| m.eos_tokens.iter().map(String::as_str).collect(),
        )
    }
    pub fn assistant_prefix(&self) -> &str {
        self.sliced
            .as_ref()
            .map_or("", |m| m.assistant_prefix.as_str())
    }
    /// The llama.cpp architecture a GGUF for this model must declare.
    pub fn architecture(&self) -> &str {
        self.sliced
            .as_ref()
            .map_or("qwen2", |m| m.architecture.as_str())
    }
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
    let extension = if let Some(m) = &manifest.sliced {
        ensure!(
            (1..=1024).contains(&m.layers)
                && (1..=65536).contains(&m.hidden_size)
                && (1..=1 << 20).contains(&m.vocab_size)
                && !m.eos_tokens.is_empty()
                && m.slices_manifest_sha256.len() == 64,
            "invalid sliced-model description"
        );
        ".gguf"
    } else {
        ensure!(
            manifest.model_id == MODEL_ID
                && manifest.revision == REVISION
                && manifest.weights_sha256 == WEIGHTS_SHA256,
            "unsupported checkpoint/revision"
        );
        ".safetensors"
    };
    ensure!(
        (1..=8).contains(&manifest.shards.len()),
        "expected 1..=8 shards"
    );
    let layers = manifest.layers();
    let mut next = 0;
    for (index, shard) in manifest.shards.iter().enumerate() {
        ensure!(
            shard.index == index
                && shard.start == next
                && shard.end > shard.start
                && shard.end <= layers,
            "manifest layer gap/overlap"
        );
        ensure!(
            Path::new(&shard.file).components().count() == 1 && shard.file.ends_with(extension),
            "invalid shard filename"
        );
        next = shard.end;
    }
    ensure!(next == layers, "manifest must cover all {layers} layers");
    if manifest.sliced.is_none() {
        check_hash(&dir.join("config.json"), &manifest.config_sha256)?;
    }
    Ok((manifest, format!("{:x}", Sha256::digest(&bytes))))
}

/// A GGUF conversion of the pinned checkpoint, approved for the llama.cpp engine.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufSpec {
    pub file: String,
    pub sha256: String,
    pub precision: String,
    pub file_bytes: u64,
}

/// `gguf.json` beside the manifest, written by scripts/prepare-gguf.py. It names the manifest
/// hash it was made from, so a GGUF can only serve the model its route expects.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GgufIndex {
    pub model_hash: String,
    pub files: Vec<GgufSpec>,
}

/// Finds `name` in the model directory's GGUF approvals and verifies its hash.
pub fn approved_gguf(dir: &Path, name: &str, model_hash: &str) -> Result<GgufSpec> {
    ensure!(
        Path::new(name).components().count() == 1 && name.ends_with(".gguf"),
        "GGUF must be a .gguf file name inside the model directory"
    );
    let index: GgufIndex = serde_json::from_slice(
        &std::fs::read(dir.join("gguf.json"))
            .context("run python3 scripts/prepare-gguf.py first")?,
    )?;
    ensure!(
        index.model_hash == model_hash,
        "gguf.json was made from a different manifest"
    );
    let spec = index
        .files
        .into_iter()
        .find(|f| f.file == name)
        .ok_or_else(|| anyhow::anyhow!("{name} is not listed in gguf.json"))?;
    check_hash(&dir.join(&spec.file), &spec.sha256)?;
    Ok(spec)
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
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                // CUDA_VISIBLE_DEVICES selects the GPU; the worker uses the first visible one.
                Ok(Device::new_cuda(0).context("initialize CUDA")?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("rebuild with --features cuda to use an NVIDIA GPU")
            }
        }
        "vulkan" | "rocm" => anyhow::bail!("{name} is available only with --engine llamacpp"),
        _ => anyhow::bail!("device must be cpu, metal or cuda"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("sangama-gguf-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), b"weights").unwrap();
        let hash = sha256(&dir.join("model.gguf")).unwrap();
        (dir, hash)
    }

    fn write_index(dir: &Path, model_hash: &str, sha: &str) {
        let index = GgufIndex {
            model_hash: model_hash.into(),
            files: vec![GgufSpec {
                file: "model.gguf".into(),
                sha256: sha.into(),
                precision: "f32".into(),
                file_bytes: 7,
            }],
        };
        std::fs::write(dir.join("gguf.json"), serde_json::to_vec(&index).unwrap()).unwrap();
    }

    fn sliced_manifest(shards: &[(usize, usize)], layers: usize) -> String {
        let shards: Vec<_> = shards
            .iter()
            .enumerate()
            .map(|(i, (start, end))| {
                serde_json::json!({"index": i, "start": start, "end": end,
                    "file": format!("stage-{i}.gguf"), "sha256": "0".repeat(64),
                    "file_bytes": 1, "tensor_count": 1})
            })
            .collect();
        serde_json::json!({"model_id": "m", "revision": "r", "weights_sha256": "w",
            "config_sha256": "", "tokenizer_sha256": "t", "shards": shards,
            "sliced": {"architecture": "qwen35moe", "layers": layers, "hidden_size": 4096,
                "vocab_size": 248320, "eos_tokens": ["<|im_end|>"],
                "assistant_prefix": "<think>\n\n</think>\n\n", "slices_repo": "a/b",
                "slices_revision": "c", "slices_manifest_sha256": "1".repeat(64)}})
        .to_string()
    }

    #[test]
    fn sliced_manifest_sets_model_shape() {
        let (dir, _) = fixture();
        std::fs::write(
            dir.join("manifest.json"),
            sliced_manifest(&[(0, 20), (20, 40), (40, 60)], 60),
        )
        .unwrap();
        let (manifest, _) = load_manifest(&dir).unwrap();
        assert_eq!(manifest.layers(), 60);
        assert_eq!(manifest.hidden_size(), 4096);
        assert_eq!(manifest.vocab_size(), 248320);
        assert_eq!(manifest.architecture(), "qwen35moe");
        assert_eq!(manifest.eos_tokens(), vec!["<|im_end|>"]);
        // Gaps, overlaps and partial coverage are refused as for the pinned model.
        for shards in [
            vec![(0, 20), (21, 60)],
            vec![(0, 30), (20, 60)],
            vec![(0, 20), (20, 50)],
        ] {
            std::fs::write(dir.join("manifest.json"), sliced_manifest(&shards, 60)).unwrap();
            assert!(load_manifest(&dir).is_err(), "{shards:?}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn approves_only_listed_gguf_for_this_manifest() {
        let (dir, sha) = fixture();
        write_index(&dir, "manifest", &sha);
        assert_eq!(
            approved_gguf(&dir, "model.gguf", "manifest")
                .unwrap()
                .precision,
            "f32"
        );
        let refused = |name: &str, hash: &str| approved_gguf(&dir, name, hash).is_err();
        assert!(refused("other.gguf", "manifest"));
        assert!(refused("../model.gguf", "manifest"));
        assert!(refused("model.safetensors", "manifest"));
        assert!(refused("model.gguf", "another-manifest"));
        write_index(&dir, "manifest", &"0".repeat(64));
        assert!(refused("model.gguf", "manifest"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
