# Execution architecture

## Real Qwen backend

Tested development machine: Apple M5 Pro with 24 GB unified memory.
The model is pinned Qwen2.5-0.5B-Instruct: 24 Transformer layers, hidden width 896,
14 attention heads, two KV heads, and tied input/output embeddings. Candle 0.11.0 executes
F32 operations on Metal or CPU. Original BF16 tensors are converted at load time; this is not quantization.

1. `fetch-qwen.py` verifies original weights/config/tokenizer and partitions tensor data into actual shard files.
2. The client validates hashes and tokenizes the chat prompt. Only optional `qwen-test` runs the unsplit baseline.
3. `generate` never loads a complete local model. In verification mode, the baseline is dropped before automatic workers start.
4. Worker 0 embeds tokens and executes layers [0,12); worker 1 executes [12,24), final normalization, and output projection.
5. Workers own KV caches for their layers. Every step carries a session, position, manifest identity, shape, and explicit route.
6. Hidden activations pass directly between workers. The final logits return through the chain for greedy sampling and comparison.
7. Generation stops at EOS/token limit and resets sessions. Optional verification also compares logits and tokens against upstream Candle.

Weights remain resident. One conversation is admitted per worker, with a 60-second idle lease.
Prefill uses a causal mask; decoding appends to cached keys/values. Warmup is followed by cache reset.
The binary frame has a bounded JSON metadata header and little-endian F32 tensor data, a 4 MiB total cap,
and finite-value/shape validation. Context is capped at 4,096 tokens; CLI prompts remain capped at 512 and generation at 128. The text-chat API uses prefill chunks of at most 512 tokens for longer conversations.
The current real-model route is explicit and does not use the fixture's placement coordinator.

The endpoint embedding matrix is duplicated because weights are tied; shard sizes are about 630 MB each
on disk. F32 resident weights and temporary loading buffers require more RAM than compressed/original disk bytes.
This prototype has context limits but does not yet implement RAM-budget admission for real Qwen.

## What a tiny participant can own

A device can own a contiguous set of complete layers and their KV state; it need not hold the whole model.
Our two workers demonstrate this with physical files and separate processes. Arbitrarily tiny fractions are
not automatically useful: every boundary adds transport and scheduling latency, while endpoints may have large embeddings.
Phones need native GPU backends, thermal/battery controls, and enough sustained memory before joining a useful route.
No phone client or Kimi/MoE expert backend has been implemented here.

## Speed and measurement

The verification checker reports first-token latency and decode tokens/second after prompt warmup.
Standalone generation reports these without an extra warmup and requires no full local checkpoint.
Model download, hash checks, model loading, process startup, and tokenizer initialization are excluded.
Counts include EOS; decode rate excludes the first token. Single short runs are correctness smoke tests,
not steady-state capacity claims. Both local workers share the same GPU and memory bandwidth on this Mac.
Two physical devices are still required to establish LAN scaling, and localhost cannot predict Internet performance.

For one autoregressive stream, layer stages run in sequence. More workers add capacity, but do not inherently
reduce per-token latency. Approximate token latency is the sum of stage compute, transfers, scheduling, and sampling.
At width 896, one F32 decode activation is 3,584 bytes; a 512-token prefill activation is about 1.84 MB.
The validation harness also returns 151,936 F32 logits (about 608 KB) each step. Sampling on the final worker
is a clear next optimization, while retaining a diagnostic mode for complete-logit verification.

Further speed work: quantized kernels, endpoint sampling, persistent transport, measured placement,
chunked prefill, batching independent requests, and later block speculative decoding with quality checks.
Do not multiply published speedups together or infer a Kimi throughput number from this small dense Qwen test.

## Numerical fixture and coordinator

The separate fixture remains useful for testing registration, three-second heartbeats, 15-second leases,
contiguous range planning, overload handling, and explicit worker failures. Its matrices are deterministic
and its measurements are passes/second, not LLM tokens/second. `demo` starts services in one process;
process integration tests exercise standalone workers. Coordinator RTT is only a placement heuristic,
not pairwise topology measurement. Artificial delay is not WAN emulation.

## Infrastructure still needed

For a trusted LAN deployment: native worker builds, pinned model storage, authenticated discovery,
real-model resource admission, pairwise latency/bandwidth measurement, session scheduling, and observability.
For public volunteers: encrypted independent identities, NAT traversal/relays, signed manifests,
per-peer limits, audited updates, and explicit policies for unreliable or malicious participants.
Real Qwen endpoints are loopback-only, with explicit downstream allowlists. The documented two-person workflow
uses authenticated SSH tunnels over Tailscale. The numerical fixture remains a separate trusted-network development tool.
Neither is public volunteer infrastructure. See [secure peer testing](secure-peer-test.md).
Transport encryption alone does not hide activations from the worker executing them.

Mid-session replay, KV migration, replication, continuous batching, and automatic failover are not implemented.
A replica holding only weights cannot continue a conversation without its KV state or replaying the prompt.
Never skip missing layers or experts while claiming the same model result.

## Acceptance gates

- Completed: real checkpoint, physical shards, Metal/CPU execution, token/logit comparison, cache reset tests.
- Next: actual two-device run, repeatable long/sustained workloads, peak-memory and thermal measurements.
- Then: quantization quality checks, endpoint sampling, measured scheduling, recovery, and mobile workers.
- Public Internet participation follows encrypted transport, trust controls, and operational testing.

## References

- [Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
- [Candle Qwen implementation](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen2.rs)
- [Petals](https://github.com/bigscience-workshop/petals): distributed layer serving.
- [EXO](https://github.com/exo-explore/exo): local device clustering.
- [libp2p connectivity](https://docs.libp2p.io/connectivity/): NAT traversal and relays.
