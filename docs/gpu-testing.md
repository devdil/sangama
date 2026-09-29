# GPU testing: slices, containers and unified memory

This note explains how to test several GPU workers cheaply, on NVIDIA and on Apple Silicon. Findings were checked against vendor documentation on 2026-09-29; sources are listed at the end. The CUDA path has not yet been run on NVIDIA hardware.

## How a worker measures its memory budget

| Device | Budget source |
|---|---|
| `cpu` | Host memory: `/proc/meminfo` (and the cgroup limit) on Linux, free/inactive/speculative pages from `vm_stat` on macOS |
| `cuda` | `cuMemGetInfo` on the first visible CUDA device. This reports the GPU, the MIG slice, or the MPS v3 memory partition the process was given [1][2] |
| `metal` | The smaller of host memory and the Metal device's `recommendedMaxWorkingSetSize − currentAllocatedSize` [3][4] |

`nvidia-smi --query-gpu=memory.free` is not used: under MIG it reports the whole GPU, or `N/A`, rather than the slice [5][6], and it does not accept MIG UUIDs as `-i` values [7].

## NVIDIA: one container per GPU or MIG slice

| Method | Isolation | Notes |
|---|---|---|
| One whole GPU per container, `--gpus device=N` | Full | Simplest. Inside the container the GPU is ordinal 0 [8] |
| MIG slice per container, `--gpus '"device=0:1"'` or `NVIDIA_VISIBLE_DEVICES=MIG-<uuid>` | Memory and compute | A100, A30, H100, H200, GH200, B200 and RTX PRO Blackwell cards. Not L40S, RTX 4090 or RTX 5090 [9] |
| MPS v3 memory partitions | Memory limit | CUDA 13.4+, cgroup v2, a non-MIG GPU [2] |
| Several containers on one GPU, no setup | None | Every worker sees the same free memory and they can overcommit [10]. Avoid |

MIG profiles include 7 × `1g.10gb` on an H100 80GB and 4 × `1g.24gb` on an RTX PRO 6000 Blackwell [11]. Enable and slice a GPU (instances do not survive a reboot) [8]:

```sh
sudo nvidia-smi -i 0 -mig 1
nvidia-smi mig -lgip                        # list profiles
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb -C
nvidia-smi -L                               # prints MIG-<uuid> for each slice
```

### Run the compose test

Needs Docker with the NVIDIA Container Toolkit, the prepared model (`python3 scripts/fetch-qwen.py`) and a token:

```sh
mkdir -p .secrets && python3 -c 'import secrets; print(secrets.token_hex(32))' > .secrets/gpu-test.token
chmod 600 .secrets/gpu-test.token
export SANGAMA_UID=$(id -u) CUDA_COMPUTE_CAP=90   # 90 = H100, 89 = L4/RTX 4090, 86 = A10/RTX 3090
export SANGAMA_GPU0=0 SANGAMA_GPU1=1              # or 0:0 and 0:1 for two MIG slices of GPU 0
docker compose -f docker/compose.gpu.yml up -d --build worker0 worker1
docker compose -f docker/compose.gpu.yml run --rm client
docker compose -f docker/compose.gpu.yml down
```

Check each worker's reported budget with `curl -H "Authorization: Bearer $(cat .secrets/gpu-test.token)" http://127.0.0.1:7901/v1/qwen/info`. On a MIG slice, `memory.budget_bytes` should be below the slice size, not the whole GPU.

Workers only accept loopback connections, so all services share the host network. This checks GPU isolation. Network separation is covered by the CPU container tests in [docker-testing](docker-testing.md).

### Where to rent GPUs (prices seen 2026-09-29)

| Need | Option | Approximate price |
|---|---|---|
| 2–4 whole GPUs | RunPod or Vast.ai multi-GPU pod (RTX A5000, RTX 4090) | $0.14–0.35 per GPU-hour [12][13] |
| MIG slices | Lambda GH200 (MIG with Docker is documented) | $2.29/hour [14][15] |
| Pre-sliced GPU VMs | AWS G6f (L4 fractions), Azure NVads A10 v5 | varies [16][17] |

## Apple Silicon: no GPU in containers

- **Docker Desktop** supports GPUs only on Windows with WSL2 [18]. Docker says "there is no GPU passthrough for Metal in containers" [19]. Docker Model Runner gets Metal speed by running its engines on the host, and only its built-in engines [20].
- **Podman with libkrun** gives Linux containers Vulkan compute, translated to Metal through MoltenVK [21][22], at about 75–80% of native speed for llama.cpp [23]. Candle 0.11 has no Vulkan backend (only an open proposal [24]), so this does not help Sangama.
- **Apple's `container` tool** has no GPU support. The request was closed as won't-fix [25].
- **macOS VMs** get Metal through paravirtualized graphics [26], but the guest GPU reports an older feature set without bf16 or simdgroup matrices, which slows ML kernels [27]. The macOS licence allows two extra macOS instances per Mac [28].
- **Unified memory cannot be partitioned.** Apple has no equivalent of MIG. Metal limits each process to `recommendedMaxWorkingSetSize`, about 74% of RAM on a 24 GB M5 Pro (measured). `sudo sysctl iogpu.wired_limit_mb=<MB>` raises the system limit until reboot; `0` restores the default [29].

For AMD, Intel and Qualcomm GPUs, build the [llama.cpp engine](llamacpp.md) with Vulkan or ROCm.

To test several Metal workers on one Mac, run native worker processes on different loopback ports (see [secure peer test](secure-peer-test.md)). They share the GPU, each within its own budget. One route may mix CPU, Metal and CUDA workers.

## Sources

1. cudarc `CudaContext::mem_get_info`, cudarc 0.19.10 `src/driver/safe/core.rs`
2. https://docs.nvidia.com/deploy/mps/mpsv3-memory-partitioning.html
3. https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
4. https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize
5. https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia-mig.html
6. https://forums.developer.nvidia.com/t/support-for-mig-devices-in-nvidia-smi-queries/189662
7. https://docs.nvidia.com/deploy/nvidia-smi/index.html
8. https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html
9. https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html
10. https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html
11. https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html
12. https://www.runpod.io/pricing
13. https://vast.ai/pricing/gpu/RTX-4090
14. https://lambda.ai/pricing
15. https://docs.lambda.ai/education/using-mig/
16. https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-ec2-g6f-instances-fractional-gpus
17. https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series
18. https://docs.docker.com/desktop/features/gpu/
19. https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/
20. https://docs.docker.com/ai/model-runner/inference-engines/
21. https://podman-desktop.io/docs/podman/gpu
22. https://developers.redhat.com/articles/2025/06/05/how-we-improved-ai-inference-macos-podman-containers
23. https://developers.redhat.com/articles/2025/09/18/reach-native-speed-macos-llamacpp-container-inference
24. https://github.com/huggingface/candle/issues/3985
25. https://github.com/apple/containerization/issues/46
26. https://developer.apple.com/documentation/paravirtualizedgraphics
27. https://cua.ai/blog/gpu-passthrough-macos-vms
28. https://www.apple.com/legal/sla/docs/macOSTahoe.pdf
29. https://github.com/ggml-org/llama.cpp/discussions/2182
