# Per-layer model slices

Large models are published as one file per layer, so each worker downloads only the layers it serves. `scripts/split-gguf.py` makes the slices and a checksummed `model.json`; the same script assembles a worker's layer range into one GGUF that Sangama's layer-range llama.cpp loads.

## Format

`split` turns a GGUF model (all of its parts) into:

| File | Contents |
|---|---|
| `embed.gguf` | Token embedding; used only by the first stage |
| `layer-000.gguf` … `layer-NNN.gguf` | One transformer layer each |
| `head.gguf` | Final norm and LM head; used only by the last stage |
| `extra.gguf` | Anything else (none so far) |
| `model.json` | Model id, architecture, layer count, quantization, whether the LM head is separate from the embedding, and each slice's size and SHA-256 |

Every slice keeps the model's full metadata, so each is a valid GGUF on its own.

`assemble --start A --end B` checks the slices against `model.json` and writes one GGUF with:
- layers `[A, B)`;
- the embedding if `A == 0`;
- the head if `B` is the last layer;
- the embedding for the last stage too if the model reuses it as its LM head (tied weights).

```sh
PYTHONPATH=.tools/llama.cpp/gguf-py python3 scripts/split-gguf.py split model-*.gguf \
  --out slices --model-id unsloth/Qwen3.5-397B-A17B-GGUF --quantization Q4_K_M
PYTHONPATH=.tools/llama.cpp/gguf-py python3 scripts/split-gguf.py assemble slices/model.json \
  --start 20 --end 40 --out stage.gguf
```

## llama.cpp changes this needs

Both patches are in `crates/llama-stage/patches/`, applied by `scripts/fetch-llama-cpp.sh`.

- **`stage-token-embd.patch`.** A stage that starts mid-model never reads the token embedding, so the loader no longer requires it. The tied LM-head alias is optional before the final stage. Per-stage files can therefore leave out the 1 GB embedding.
- **`qwen35moe-layer-split.patch`.** Wires layer ranges into the Qwen3.5 MoE graph (the upstream PR covered `qwen35` and `qwen3moe`, not `qwen35moe`).

The worker shim also had to change. Qwen3.5 uses M-RoPE: when a batch carries hidden states instead of tokens, llama.cpp reads four position sections per token. Supplying one left three uninitialised, so every non-first stage changed its output from run to run. The shim now fills all four sections.

## Qwen3.5-397B-A17B, Q4_K_M (2026-09-29)

- **Source:** `unsloth/Qwen3.5-397B-A17B-GGUF`, Q4_K_M, 6 parts, 244.1 GB, Apache 2.0.
- **Architecture:** `qwen35moe`: 60 layers, width 4096, 512 experts with 10 active, full attention every fourth layer.
- **Slices:** 62 files totalling 244.8 GB. Each layer is 4.04–4.05 GB, the embedding 1.09 GB and the head 0.83 GB.
- **Where it ran:** a CPU-only cloud instance (16 vCPU, 129 GB RAM, 700 GB disk).
  - Download: 5 min 49 s.
  - Split, including hashing: about 10 min.
  - Assembling a 4-layer, 17 GB range: 80–100 s.

The whole model does not fit in that machine's memory, so correctness was checked range by range: 8 tokens in, compared with `spikes/llamacpp/stage-compare.cpp`.

| Check | Ranges | Result |
|---|---|---|
| Same stage twice (determinism, after the M-RoPE fix) | [28, 32) | Identical |
| One stage vs two chained stages (layer-range wiring) | [0,2)→[2,4) · [28,30)→[30,32) · [56,58)→[58,60) with logits | Bit-exact, all three |
| Assembled slice file vs original parts | [0, 4) · [28, 32) · [56, 60) with logits | Bit-exact, all three |

Before the M-RoPE fix, the same middle stage differed from itself by up to 2.6, and the chains by up to 2.7.

**Published:** [diljitpr/Qwen3.5-397B-A17B-Q4_K_M-slices](https://huggingface.co/diljitpr/Qwen3.5-397B-A17B-Q4_K_M-slices) on Hugging Face (public, Apache 2.0, with a model card and the licence). It has 66 files and 244.8 GB, and every size matches `model.json`. Two slices downloaded anonymously matched their SHA-256.

Hugging Face deduplicates storage in chunks, and the slices hold the same tensor bytes as Unsloth's upload, so 245 GB uploaded in a few minutes. A free account's private-storage limit stopped the upload at 79 GB; the repo was made public to finish.

## Still to do

- Have workers download and verify only their own layers from the published repo.
- Generalise Sangama's pinned manifest and worker checks, which are still specific to Qwen2.5-0.5B.
- Run the full model across machines. Three 128 GB+ devices hold one copy at Q4_K_M.
