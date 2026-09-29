# llama.cpp engine

A worker can run its layers with llama.cpp instead of Candle. llama.cpp brings GPU backends Candle lacks (Vulkan for AMD, Intel and Qualcomm GPUs; ROCm/HIP for AMD) and quantized GGUF weights. Both engines take tokens or F32 hidden states and return F32 hidden states or logits, so one route can mix them.

## How it works

Each worker loads layers `[start, end)` of its manifest shard from a GGUF file. llama.cpp's public API has no layer ranges yet, so Sangama builds a pinned llama.cpp with [unslothai/llama.cpp PR #180](https://github.com/unslothai/llama.cpp/pull/180) plus a Qwen2 patch (`crates/llama-stage/patches/`). `crates/llama-stage` wraps one stage behind a small C shim.

A worker only loads a GGUF that `gguf.json` in the model directory approves: the file name, its SHA-256, its weight precision, and the manifest hash it was converted from. This keeps the pinned-model guarantee: a route cannot silently run a different model.

A route may mix engines and devices, but every worker must use the same weight precision. A route with Q4_K_M and F32 workers is refused, because it would change the model's output without saying so.

## Build

Needs CMake, a C/C++ compiler, and the toolkit of the chosen GPU backend.

```sh
./scripts/fetch-llama-cpp.sh          # pinned source, checksum-verified and patched, in .tools/llama.cpp
./scripts/cargo build --release --locked --features llamacpp-metal    # Apple Silicon
./scripts/cargo build --release --locked --features llamacpp-cuda     # NVIDIA (CUDA toolkit)
./scripts/cargo build --release --locked --features llamacpp-vulkan   # AMD, Intel, NVIDIA, Qualcomm (Vulkan SDK)
./scripts/cargo build --release --locked --features llamacpp-hip      # AMD ROCm (ROCM_PATH)
./scripts/cargo build --release --locked --features llamacpp          # CPU only
```

Pick at most one llama.cpp GPU feature. It can be combined with Candle's `metal` or `cuda` features. `sangama doctor` reports `llamacpp_compiled`.

## Prepare GGUF weights

```sh
python3 -m venv .tools/convert-venv
.tools/convert-venv/bin/pip install -r .tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.tools/convert-venv/bin/python scripts/prepare-gguf.py --quantize Q4_K_M
```

This writes `qwen2.5-0.5b-instruct-f32.gguf` (the same values Candle uses) and, with `--quantize`, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (5× smaller), and approves both in `gguf.json`.

## Run workers

```sh
./target/release/sangama qwen-worker --shard 0 --device metal --engine llamacpp \
  --gguf qwen2.5-0.5b-instruct-f32.gguf --listen 127.0.0.1:7901 --allow-next 127.0.0.1:7902
./target/release/sangama qwen-worker --shard 1 --device metal --listen 127.0.0.1:7902   # Candle
./target/release/sangama generate --peers 127.0.0.1:7901,127.0.0.1:7902
```

`--device` is `cpu` or the backend llama.cpp was built for: `metal`, `cuda`, `vulkan` or `rocm`. A worker refuses a device its build lacks. The generation report lists each worker's engine, device and precision.

## Verified (Apple M5 Pro, macOS 26.3, 2026-09-29)

| Route (shard 0 + shard 1) | 20-token greedy output |
|---|---|
| Candle Metal + Candle Metal | Reference |
| llama.cpp Metal + llama.cpp Metal, F32 | Identical to reference |
| llama.cpp Metal + Candle Metal, F32 | Identical to reference |
| Candle CPU + llama.cpp CPU, F32 | Identical to reference |
| llama.cpp Metal + llama.cpp Metal, Q4_K_M | Runs; slightly different sentence, as quantization is expected to give |
| llama.cpp Q4_K_M + Candle F32 | Refused: route mixes weight precisions |

Workers also refuse a GGUF missing from `gguf.json`, a hash mismatch, a `gguf.json` made from another manifest, a precision that differs from `gguf.json`, and a device the build lacks. `crates/llama-stage/tests/split.rs` checks split against unsplit generation when `SANGAMA_TEST_GGUF` points to a GGUF.

## Not done yet

- **CUDA, Vulkan and ROCm builds are untested.** Only Metal and CPU have run. The ROCm device-name check assumes llama.cpp names HIP devices `ROCm0`.
- **Memory budget.** A worker budgets for the whole GGUF, and each stage allocates KV cache for every layer. Per-worker GGUF slices would fix both.
- **Managed workers and the admitted mesh** still run Candle and F32 only.
- **Maintaining the patch.** Track upstream llama.cpp for a layer-range API and rebase until one lands.
