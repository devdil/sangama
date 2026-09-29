# Test scenario matrix

Every scenario Sangama should pass before it is called cross-platform, with its status. **Pass** means it ran and met its check; **Fail** means it ran and did not; **Blocked** says what is missing; **Planned** has not run yet. Update this file whenever a scenario runs.

All generation checks use the pinned Qwen2.5-0.5B-Instruct, greedy decoding and the prompt "Explain peer-to-peer computing in one short sentence." in Sangama's chat format. F32 routes must reproduce the Candle Metal reference token IDs exactly:

```
30888 4686 78597 24231 6147 3847 311 4564 323 4332 4963 2041 279 1184 369 264 8622 3538 13 151645
```

Q4_K_M routes must reproduce the Q4_K_M reference instead (they are not expected to match F32).

## Environments

| ID | Machine | Location | OS | GPU / driver | Notes |
|---|---|---|---|---|---|
| E1 | Apple M5 Pro, 24 GB | Maintainer's home network | macOS 26.3 | Metal (unified memory) | Development machine |
| E2 | Vast.ai instance 53337690, 2× RTX 3060 12 GB, Xeon E5-2680 v4 | Wisconsin, US (datacenter host); 260–370 ms from E1 | Ubuntu 24.04 container | Driver 550.144 (CUDA 12.4), CUDA toolkit 12.4 installed, Vulkan 1.3.277 | Instances are containers: no Docker-in-Docker |
| E3 | Linux + AMD Radeon (RDNA3/4) | — | Ubuntu | ROCm, Vulkan | Not yet available |
| E4 | Windows 11 PC | — | Windows 11 | NVIDIA or AMD | Not yet available |
| E5 | Intel Arc or Core Ultra | — | Linux or Windows | Vulkan | Not yet available |
| E6 | Snapdragon X laptop | — | Windows on ARM | Adreno, Vulkan | Not yet available |
| E7 | Docker host with 2+ NVIDIA GPUs or MIG | — | Linux (VM or bare metal) | NVIDIA Container Toolkit | For `docker/compose.gpu.yml` |

## A. Single-machine engine and device coverage

| ID | Scenario | Env | Check | Status |
|---|---|---|---|---|
| A1 | Candle CPU, 2 local workers | E1 | F32 reference | Pass (2026-09-29) |
| A2 | Candle Metal, 2 local workers | E1 | Defines the F32 reference | Pass (2026-09-29) |
| A3 | llama.cpp Metal F32, 2 workers | E1 | F32 reference | Pass (2026-09-29) |
| A4 | llama.cpp CPU F32 | E1 | F32 reference | Pass (2026-09-29, with a Candle CPU peer) |
| A5 | llama.cpp Metal Q4_K_M, 2 workers | E1 | Defines the Q4 reference | Pass (2026-09-29) |
| A6 | Candle CUDA, 2 workers on GPU 0 and GPU 1 | E2 | F32 reference | Pass (2026-09-29, [run](gpu-matrix-2026-09-29.md)) |
| A7 | llama.cpp CUDA F32, 2 workers on separate GPUs | E2 | F32 reference | Pass (2026-09-29, [run](gpu-matrix-2026-09-29.md)) |
| A8 | llama.cpp CUDA Q4_K_M | E2 | Q4 reference | Differs from the Metal Q4 reference; quantized output is file- and backend-specific ([run](gpu-matrix-2026-09-29.md), finding 2) |
| A9 | llama.cpp Vulkan on NVIDIA | E2 | F32 reference | Pass (2026-09-29, [run](gpu-matrix-2026-09-29.md)) |
| A10 | llama.cpp ROCm/HIP; confirms the `ROCm0` device name | E3 | F32 reference | Blocked: no AMD machine |
| A11 | llama.cpp Vulkan on AMD | E3 | F32 reference | Blocked: no AMD machine |
| A12 | Windows build (MSVC, `_putenv_s`), CUDA or Vulkan | E4 | F32 reference | Blocked: no Windows machine |
| A13 | llama.cpp Vulkan on Intel | E5 | F32 reference | Blocked |
| A14 | llama.cpp Vulkan on Qualcomm Adreno | E6 | F32 reference | Blocked |

## B. Mixed engines and devices in one route

| ID | Scenario | Env | Check | Status |
|---|---|---|---|---|
| B1 | llama.cpp Metal → Candle Metal | E1 | F32 reference | Pass (2026-09-29) |
| B2 | Candle CPU → llama.cpp CPU | E1 | F32 reference | Pass (2026-09-29) |
| B3 | Candle CPU → Candle Metal | E1 | F32 reference | Pass (2026-09-29) |
| B4 | Candle CUDA → llama.cpp CUDA | E2 | F32 reference | Pass (2026-09-29, [run](gpu-matrix-2026-09-29.md)) |
| B5 | llama.cpp CUDA → Candle CPU | E2 | F32 reference | Pass (2026-09-29, [run](gpu-matrix-2026-09-29.md)) |
| B7 | llama.cpp Vulkan → Candle CUDA (two builds) | E2 | F32 reference | Pass (2026-09-29, [run](gpu-matrix-2026-09-29.md)) |
| B8 | Q4_K_M + F32 on CUDA | E2 | Refused | Pass (2026-09-29) |
| B9 | Two different Q4_K_M GGUF files | E2 | Refused: route mixes different GGUF files | Pass after fix (2026-09-29); failed before `weights_sha256` was added |
| B6 | Q4_K_M worker + F32 worker | E1 | Refused: "route mixes weight precisions" | Pass (2026-09-29) |

## C. Across machines and geographies

Workers only listen on loopback; cross-machine hops use SSH tunnels (see [secure peer test](../secure-peer-test.md)) or the admitted mesh.

| ID | Scenario | Env | Check | Status |
|---|---|---|---|---|
| C1 | Shard 0 on Mac Metal, shard 1 on US CUDA, over an SSH tunnel | E1 + E2 | F32 reference; record RTT, time to first token, tokens/s | Pass: 1.6 s first token, 3.1 tok/s ([run](gpu-matrix-2026-09-29.md)) |
| C2 | Shard 0 on US CUDA, shard 1 on Mac Metal (reverse direction) | E1 + E2 | F32 reference; timings | Pass: 3.4 s, 1.2 tok/s ([run](gpu-matrix-2026-09-29.md)) |
| C3 | Mixed engines across the WAN: Mac llama.cpp Metal → US Candle CUDA | E1 + E2 | F32 reference | Pass ([run](gpu-matrix-2026-09-29.md)) |
| C4 | Q4_K_M across the WAN, both llama.cpp | E1 + E2 | Q4 reference; timings | Runs and is accepted with one GGUF file; differs from Metal-only Q4 (finding 2) |
| C5 | Same-region pair (two hosts in one metro) | Two E2-class hosts | F32 reference; compare tokens/s with C1 | Blocked: needs a second host |
| C6 | Intercontinental pair (US ↔ Europe or Asia datacenter) | Two datacenter hosts | F32 reference; tokens/s | Blocked: needs a second host |
| C7 | Two home networks behind NAT, admitted mesh with relay | Two home machines | F32 reference; relay vs direct path | Blocked: the long-standing two-home-network milestone |
| C8 | Mac ↔ US box through a relay on the Linode server | E1 + E2 + Linode | F32 reference; compare with C1 | Blocked: the Linode runs only the portal (ports 22, 80, 443); deploying a relay needs approval |

## D. Containers and GPU slicing

| ID | Scenario | Env | Check | Status |
|---|---|---|---|---|
| D1 | `compose.gpu.yml`, one whole GPU per worker container | E7 | F32 reference; each worker sees one GPU | Blocked: Vast instances cannot run Docker |
| D2 | `compose.gpu.yml` with MIG slices | E7 (H100/GH200/RTX PRO Blackwell) | Budget below slice size | Blocked |
| D3 | CPU container isolation tests (`scripts/test-containers.py`) | E1 with Docker | Existing harness passes | Planned (Docker Desktop was not running) |

## E. Refusals and failure handling

| ID | Scenario | Env | Check | Status |
|---|---|---|---|---|
| E-1 | GGUF not in `gguf.json`, path outside model dir, missing `--gguf` | E1 | Worker refuses to start | Pass (2026-09-29) |
| E-2 | Tampered GGUF hash, `gguf.json` from another manifest, precision mismatch | E1 | Worker refuses to start | Pass (2026-09-29) |
| E-3 | Device the build lacks (CUDA on a Metal build; Vulkan with Candle) | E1 | Clear error | Pass (2026-09-29) |
| E-4 | CUDA toolkit newer than the driver (12.8 toolkit, 12.4 driver) | E2 | Documented failure mode or clean run | Pass: refuses to start with a rebuild hint ([run](gpu-matrix-2026-09-29.md)) |
| E-5 | Worker killed mid-generation | E1 or E2 | Request fails; session released; no stuck lease | Pass ([run](gpu-matrix-2026-09-29.md)) |
| E-6 | Second client while a worker is busy | E1 or E2 | 409 or busy error; first session unaffected | Pass ([run](gpu-matrix-2026-09-29.md)) |
| E-7 | Memory budget too small (`--memory-budget-mib` on a managed worker) | E2 | Load refused with required vs budget MiB | Planned |
| E-8 | Tunnel drops mid-generation across the WAN | E1 + E2 | Request fails cleanly; no partial output | Planned |

## F. Performance baselines

Recorded per run, not pass/fail: model load time, time to first token, decode tokens/s, per-hop RTT, GPU memory per worker.

| ID | Route | Env | Status |
|---|---|---|---|
| F1 | Candle CUDA vs llama.cpp CUDA, F32, same box | E2 | Recorded: 55 vs 103 tok/s ([run](gpu-matrix-2026-09-29.md)) |
| F2 | llama.cpp CUDA F32 vs Q4_K_M | E2 | Recorded: 103 vs 220 tok/s ([run](gpu-matrix-2026-09-29.md)) |
| F3 | Local route vs cross-geography route (C1) | E1 + E2 | Recorded: 55–106 vs 1–3 tok/s ([run](gpu-matrix-2026-09-29.md)) |
