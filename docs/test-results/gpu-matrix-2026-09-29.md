# GPU, engine and cross-geography test run, 2026-09-29

This run tested Sangama on NVIDIA GPUs for the first time. It covered CUDA and Vulkan, both engines, routes between the maintainer's Mac and a US datacenter, and failure handling. Scenario definitions and statuses are in the [scenario matrix](scenario-matrix.md). Raw results are in [results-nvidia-2026-09-29.jsonl](results-nvidia-2026-09-29.jsonl) and [results-wan-2026-09-29.jsonl](results-wan-2026-09-29.jsonl), produced by `scripts/route-matrix.py`.

Code: `main` at `c4b8851`, plus the fixes listed under [Changes made during the run](#changes-made-during-the-run).

## Machines

| | Mac (E1) | GPU box (E2) |
|---|---|---|
| Hardware | Apple M5 Pro, 24 GB unified memory | Vast.ai instance 53337690: 2× RTX 3060 12 GB (compute 8.6), Xeon E5-2680 v4, 56 threads, 251 GB RAM |
| Location | Maintainer's home network (region not recorded) | Wisconsin, US (hosting provider) |
| OS | macOS 26.3 | Ubuntu 24.04 container, kernel 5.19 |
| GPU stack | Metal | NVIDIA driver 550.144 (CUDA 12.4), CUDA toolkit 12.4, Vulkan 1.3.277 |
| Builds | `metal,llamacpp-metal` | `cuda,llamacpp-cuda` and `llamacpp-vulkan` |

Network between them: TCP connect round trip **260 ms minimum, 371 ms median** (15 samples). The Mac's upload measured about **5.3 Mbit/s**: 398 MB took 9 min 55 s.

## Results on the GPU box

All routes run shard 0 on GPU 0 and shard 1 on GPU 1, in separate processes (except B5). Timings are from one warm run; first-token time includes the prompt (30 tokens).

| ID | Route | Result | First token | Decode |
|---|---|---|---|---|
| A6 | Candle CUDA + Candle CUDA | ✅ exact F32 reference | 107 ms | 55 tok/s |
| A7 | llama.cpp CUDA + llama.cpp CUDA, F32 | ✅ exact | 295 ms | 103 tok/s |
| A8 | llama.cpp CUDA, Q4_K_M made on the box | ⚠️ runs; differs from the Metal Q4 reference | 106 ms | 220 tok/s |
| A8b | llama.cpp CUDA, Q4_K_M made on the Mac | ⚠️ runs; differs from Metal Q4 and from A8 | 98 ms | 205 tok/s |
| A9 | llama.cpp **Vulkan** + llama.cpp Vulkan, F32 | ✅ exact | 55 ms | 106 tok/s |
| B4 | Candle CUDA → llama.cpp CUDA | ✅ exact | 198 ms | 74 tok/s |
| B5 | llama.cpp CUDA → Candle CPU | ✅ exact | 491 ms | 3.7 tok/s |
| B7 | llama.cpp Vulkan → Candle CUDA | ✅ exact | 83 ms | 86 tok/s |
| B8 | Q4_K_M + F32 | ✅ refused: route mixes weight precisions | | |
| B9 | Box-made Q4_K_M + Mac-made Q4_K_M | ✅ refused: route mixes different GGUF files | | |

The first run of A6 took 10.8 s to its first token, and of A9 2.7 s, while CUDA kernels and Vulkan shaders compiled; the numbers above are from the second run.

## Results between the Mac and the US box

Each hop between machines is an SSH tunnel with the same loopback port on both ends. The client runs on the Mac.

| ID | Route | Result | First token | Decode |
|---|---|---|---|---|
| C1 | Mac Candle Metal → US Candle CUDA | ✅ exact F32 reference | 1.6 s | 3.1 tok/s |
| C2 | US Candle CUDA → Mac Candle Metal | ✅ exact | 3.4 s | 1.2 tok/s |
| C3 | Mac llama.cpp Metal → US Candle CUDA | ✅ exact | 1.6 s | 1.2 tok/s |
| C4 | Mac llama.cpp Metal Q4 → US llama.cpp CUDA Q4, same GGUF file | ⚠️ runs and is accepted; differs from Metal-only Q4 | 1.8 s | 2.2 tok/s |

C1 costs one intercontinental round trip per token (about 320 ms per token against a 371 ms median RTT). C2 costs two: client → US → Mac → client. C3 has the same topology as C1, but its run coincided with round trips of up to 4.7 s, so treat its speed as one noisy sample.

## Failure handling

| ID | Check | Result |
|---|---|---|
| E-4 | Candle built with CUDA 12.8 on a 12.4 driver | ✅ worker refuses to start: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, now with a hint to rebuild with a toolkit no newer than the driver |
| E-5 | Shard 1 killed 4 s into a 60-token generation | ✅ client fails with `503`; after restarting shard 1 the next generation succeeds at once with exact tokens (no stale lease) |
| E-6 | Second client while a route is busy | ✅ second client gets `409 Conflict`; first client finishes with exact tokens |

## Findings

1. **F32 is exact everywhere tested.** Candle and llama.cpp on CPU, Metal, CUDA and Vulkan, mixed in any combination and across a 260–370 ms network path, produce the same 20 tokens.
2. **Quantized output depends on the GGUF file and on the GPU backend.**
   - Quantizing the same F32 GGUF on the Mac and on the Linux box produced different files (`2de87c…` vs `fea217…`). The F32 conversion is byte-identical on both.
   - One Q4 file gives different tokens on CUDA than on Metal, because the backends use different quantized matrix kernels.
   - All outputs were coherent. But a Q4 route is only reproducible on the same file and the same backend.
3. **Before this run, a route could silently mix two different Q4 files.** Workers now report the GGUF's SHA-256, and routes refuse mismatched files (B9).
4. **Weights should be downloaded from a published source, not converted on each peer or uploaded between peers.** The box fetched the checkpoint from Hugging Face in 90 s; sending 398 MB from the Mac took 10 minutes. Recommendation:
   - quantize once;
   - publish the GGUFs (later, per-layer-range slices) with their hashes in the pinned manifest;
   - have every worker download and verify them.
5. **Toolkit and driver must match for Candle CUDA.** Candle compiles its kernels to PTX, which a driver older than the toolkit rejects. On Vast.ai the image's CUDA 12.8 toolkit did not match the host's 12.4 driver; installing CUDA 12.4 from NVIDIA's Ubuntu 22.04 repository fixed it.
6. **Build dependencies found:**
   - The CUDA llama.cpp build links NCCL when present. Sangama now disables NCCL, since each worker uses one GPU.
   - The Vulkan build needs `spirv-headers`, `glslc` and `libvulkan-dev`.
7. **Distance dominates speed.** Local GPU routes decode at 55–106 tok/s (F32). The same route between the Mac and the US box decodes at 1–3 tok/s, limited by round trips, as `docs/large-models.md` predicted.

## Changes made during the run

- `crates/llama-stage/build.rs`: build llama.cpp with `GGML_CUDA_NCCL=OFF`.
- Worker info reports `weights_sha256` for llama.cpp workers; routes refuse mixed GGUF files.
- The generation report's `precision` reflects the route's weights instead of always saying F32.
- The Candle CUDA load error explains a toolkit newer than the driver.
- `scripts/route-matrix.py`: runs scenario files and records tokens and timings.

## Not covered by this run

AMD (ROCm, Vulkan), Intel, Qualcomm, Windows, Docker GPU slicing (Vast.ai instances cannot run Docker), MIG, the admitted mesh with a relay, and a second GPU host in another region. The [scenario matrix](scenario-matrix.md) lists each as Blocked or Planned.
