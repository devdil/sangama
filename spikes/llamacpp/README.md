# llama.cpp layer-split spike

Can llama.cpp run a contiguous range of layers per worker, passing hidden states between stages, the way Sangama's Candle workers do? If so, Sangama gets llama.cpp's GPU backends (CUDA, Metal, Vulkan, ROCm/HIP, SYCL, OpenCL, and others) and quantized GGUF weights. This spike answered that for the pinned Qwen2.5-0.5B-Instruct model; the result is now the [llama.cpp engine](../../docs/llamacpp.md).

## What was tested

- **llama.cpp:** [unslothai/llama.cpp PR #180](https://github.com/unslothai/llama.cpp/pull/180) at `bc230ec`. It adds `LLAMA_PP_IL_BEG`/`LLAMA_PP_IL_END`: a stage runs layers `[beg, end)`, takes hidden states through `llama_batch.embd` when `beg > 0`, and emits the residual stream instead of logits when `end < n_layer`. The loader skips weights outside the range. The PR wires Qwen3 MoE and Qwen3.5; [`qwen2-layer-split.patch`](../../crates/llama-stage/patches/qwen2-layer-split.patch) applies the same change to Qwen2.
- **Weights:** the pinned checkpoint converted to F32 GGUF (the same values Candle uses), and a Q4_K_M quantization of it.
- **Checks:**
  - The PR's `test-layer-split` compares a split at layers 1, 12 and 23 with the unsplit model.
  - [`pipeline-gen.cpp`](pipeline-gen.cpp) runs greedy generation through two stages, each loading only its own layers and keeping its own KV cache, and compares the tokens with the unsplit model.

Run it all with `sh spikes/llamacpp/reproduce.sh`.

## Results (Apple M5 Pro, macOS 26.3, 2026-09-29)

| Check | Metal | CPU |
|---|---|---|
| Stage A output vs the full model's input to the split layer, F32, splits 1/12/23 | Bit-exact | Bit-exact |
| Split vs unsplit logits, F32 and Q4_K_M | Max difference 0 | Max difference 0 |
| Repeated runs of stage B | Identical | Identical |
| 20-token greedy generation, split at 12, F32 | Same token IDs as unsplit **and as Sangama's Candle Metal route** | Same |
| 26-token greedy generation, Q4_K_M | Same token IDs as unsplit | not run |

The F32 split reproduced Candle's output exactly: *"Peer-to-peer computing allows users to connect and share resources without the need for a central server."* Q4_K_M produces a slightly different sentence, as quantization is expected to.

The open report of non-deterministic `embd` input on CPU ([llama.cpp #28963](https://github.com/ggml-org/llama.cpp/issues/28963)) did not reproduce. The PR guards against it: a stage that starts mid-model no longer builds the token-embedding branch, which would otherwise read uninitialised token ids.

### Memory per stage (weights loaded without mmap)

| Weights | Unsplit (GPU) | Each stage, split at 12 (GPU) |
|---|---|---|
| F32 | 1,885 MiB | 1,202 MiB |
| Q4_K_M | 374 MiB | 256 MiB |

With the default mmap, llama.cpp reports the whole file mapped, but untouched layers are never read. The tied input/output embedding (519 MiB in F32) is held by both the first stage (input) and the last (LM head); it is a large share of this small model and a small share of large ones.

## Gaps before this can replace or sit beside Candle

1. **Rust integration.** Build the patched llama.cpp through `llama-cpp-sys-2` and expose a worker engine (`--engine llamacpp`) that speaks Sangama's frame protocol. Hidden states are F32 on both engines, so routes can mix them.
2. **Layer range as parameters.** The PR reads environment variables at model load; a worker needs them as model and context parameters.
3. **KV cache for skipped layers.** Each stage still allocates KV cache for all layers (12 MiB here at 1,024 context); it should cover only its own range.
4. **GGUF per worker.** Workers should download only their own layers, for example as split GGUF files per layer range.
5. **Maintaining the patch.** Upstream llama.cpp has no layer-range API yet. Track it, and rebase the patch until one lands.
6. **Other vendors.** Validate CUDA, ROCm/HIP, Vulkan (AMD, Intel, Qualcomm) and SYCL on real hardware. Only Metal and CPU were tested here.
