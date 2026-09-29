# Large models: peers needed and expected speed

This note estimates what it would take for Sangama to serve large open models such as Qwen3.5-397B and Kimi K2.6 instead of the Qwen2.5-0.5B test model. It is research, not a measurement of Sangama. Figures come from published model cards, papers and benchmarks as of September 2026; rows marked **(est.)** are calculations from those figures. Read [fundamentals](fundamentals.md) first if tokens, layers or KV caches are unfamiliar.

**Summary:** Qwen3.5-397B is a realistic next target: about 3–5 well-placed peers per copy, roughly 5–12 tokens/s for one conversation. Kimi K2.6 needs about 7–15 peers per copy and gives roughly 2–5 tokens/s. The 2–3 trillion-parameter flagships are too heavy for now. Network latency between peers, not per-machine speed, decides generation speed.

## Candidate models

| Model | Total / active parameters | Weights | Licence | Notes |
|---|---|---|---|---|
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | 397B / 17B | ~180 GB (3-bit) to ~245 GB (4-bit) | Apache 2.0 | 60 layers, hidden size 4096. Hybrid Gated DeltaNet + 15 full-attention layers; KV cache about 30 KB/token (est.). Best first target. |
| [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) / K2.7-Code | 1T / 32B | ~595 GB native INT4; ~340 GB 2-bit | Kimi licence | 61 layers, hidden size 7168, MLA attention; KV cache about 69 KB/token (est.). Second target. |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2.4T / 95B | 657 GB (2-bit) to 1.3 TB (4-bit) | Revenue-gated | Too heavy for now. |
| [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 2.8T / 104B | ~1.5 TB native MXFP4; ~861 GB 2-bit | Proprietary | Too heavy for now. |

Quantized sizes are from Unsloth's GGUF releases ([Qwen3.5](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF), [Kimi K2.6](https://unsloth.ai/docs/models/kimi-k2.6)). For Kimi K2.x, 8-bit files are no smaller than the native INT4 release, so there is nothing to gain from them.

A mixture-of-experts (MoE) model computes only its active parameters for each token, but every expert must still be loaded on some peer. MoE makes each token cheaper; it does not reduce the number of peers needed.

## Peers needed per copy (est.)

Assumes usable memory of about 20 GB per 24 GB GPU, 45 GB per 64 GB Mac and 95 GB per 128 GB Mac.

| Model and size | 24 GB GPUs | 64 GB Macs | 128 GB Macs |
|---|---|---|---|
| Qwen3.5-397B, 3-bit (~180 GB) | ~10 | 4–5 | 2 |
| Qwen3.5-397B, 4-bit (~245 GB) | ~13 | ~6 | 3 |
| Kimi K2.6, 2-bit (340 GB) | ~17 | ~8 | 4 |
| Kimi K2.6, INT4 (595 GB) | ~30 | ~14 | 7 |
| Kimi K3 / Qwen3.8-2.4T | 35–75 | 15–35 | 7–16 |

Multiply by 2–3 so every layer range has spare copies when volunteers disconnect. The public Petals network shows why: its [Llama-3.1-405B listing](https://health.petals.dev/) reports "not enough servers" because some layers have no one serving them. A dependable Qwen3.5-397B network is therefore about 6–15 large Macs or 25–40 GPUs.

## Expected speed

For one conversation, each token passes through every stage in order and returns to the client:

```
time per token ≈ compute time + (stages + 1) × one-way latency between peers
```

- **Compute is small.** Reading the active weights costs about 10–70 ms per token on RTX 4090s or M-series Macs (est.).
- **Latency dominates.** Round-trip times: home connections add 7–34 ms each ([FCC](https://www.fcc.gov/reports-research/reports/measuring-broadband-america/measuring-fixed-broadband-thirteenth-report)); same metro about 8 ms, same continent 18–66 ms, intercontinental 70–220 ms between data centres ([WonderNetwork](https://wondernetwork.com/pings/New%20York)).
- **Bandwidth matters for prompts, not decode.** Each decode step moves one BF16 activation per hop: 8 KB for Qwen3.5-397B, 14 KB for Kimi. An 8,000-token prompt through Kimi moves about 115 MB per hop, roughly 9 s at 100 Mbit/s.

| Setup (est.) | Tokens/s, one conversation |
|---|---|
| Qwen3.5-397B, 2–3 Macs, same city | 8–12 |
| Qwen3.5-397B, 3–5 peers, same continent | 4–7 |
| Qwen3.5-397B, 10–13 GPUs, same continent | 2–3 |
| Kimi K2.6, 4–8 peers, same continent | 2–5 |
| Any route spanning continents | 1–2 |

Published measurements for comparison:

| System | Result |
|---|---|
| [Petals](https://arxiv.org/pdf/2312.08361) | BLOOM-176B on 14 volunteer servers across Europe and North America: 0.83 tokens/s. Llama-2-70B at 100 ms RTT: 1.57 tokens/s (2.29 on a LAN). |
| [exo](https://blog.exolabs.net/day-2/) | DeepSeek V3 671B (4-bit) on 8 Mac minis over a LAN: 5.37 tokens/s. |
| [MLX, one M3 Ultra 512 GB](https://github.com/ml-explore/mlx/discussions/3209) | Kimi K2.5 (4-bit): 11.1 tokens/s at 1K context, 3.8 at 128K. |
| [ktransformers](https://ktransformers.net/en/benchmarks) | Kimi K2.6 on 4× RTX 5090 + 2× EPYC: 29 tokens/s; Qwen3.5-397B (FP8): 32–34 tokens/s. |

**Design consequences:**

- **Prefer fewer, larger peers that are close together.** Every extra stage adds a network hop per token.
- **Keep splitting by layers.** Tensor or expert parallelism needs synchronisation inside every layer, more than 100 round trips per token for Kimi, which is impractical over home connections. Even [llama.cpp RPC slows down as nodes are added](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/) on a LAN.
- **Throughput comes from concurrency.** A route with N stages needs at least N conversations in flight to keep every stage busy. Petals measured about 7× more total output with 10 concurrent clients.

## Work needed in Sangama

1. **Quantized weights.** Load 3–4-bit weights (GGUF or MLX) instead of F32, and send BF16 activations between stages. This is the largest blocker. The [llama.cpp spike](../spikes/llamacpp/README.md) ran Qwen2.5-0.5B as two layer-range stages, in F32 and Q4_K_M, bit-exact against the unsplit model and token-identical to Candle.
2. **Model architectures.** Qwen3.5 needs MoE and Gated DeltaNet layers; Kimi needs MLA and native INT4. Support in candle 0.11 has not been checked yet.
3. **Region-aware placement.** The allocator currently minimises client-to-peer probe latency. It should measure latency between peers and keep a route within one city or region.
4. **Concurrent conversations per worker.** Workers currently serve one conversation at a time.
5. **Failure recovery.** Today a disconnect fails the request. [Petals](https://arxiv.org/pdf/2312.08361) keeps the inputs sent to each stage on the client and replays them to a replacement server.
6. **Speculative decoding.** A draft predicts several tokens and the route verifies them in one pass. Qwen's multi-token-prediction heads can serve as the draft. [Reported speedups](https://arxiv.org/html/2511.11733) are 2.3–2.6×, though measured with simulated WAN latency.
7. **Partial downloads.** Each peer should download only its own 20–95 GB slice.

**Suggested milestone:** run Qwen3.5-397B at 3-bit on 2–3 Macs in one city, then repeat across two home networks and record time to first token and decode tokens/s.
