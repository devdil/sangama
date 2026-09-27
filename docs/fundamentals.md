# Fundamentals: how a language model becomes a peer network

This guide explains the concepts used by the current code. Start with [contributor setup](../CONTRIBUTING.md) when you want to run it, and [architecture](architecture.md) when you want to trace it.

## What an LLM does

A language model predicts the next token given earlier tokens. A tokenizer converts text to integer IDs; a token can represent a word, part of a word, punctuation or another text fragment. A chat template adds markers identifying user and assistant messages. Tokens are not the same as characters or words.

Training adjusts numerical parameters called **weights** using examples. Inference uses those already-trained weights to compute outputs. Sangama performs inference; it does not train or update the model as people join.

The weights are organized into tensors, which are multidimensional arrays. A checkpoint stores those arrays. Safetensors is the file format used here; the model configuration describes shapes and architecture, and the tokenizer describes text encoding. The manifest identifies the expected files, hashes and layer ranges.

## One forward pass

For the pinned Qwen model, execution follows this sequence:

1. **Embedding:** look up a vector for each input token. This model uses a hidden width of 896 numbers per token.
2. **Transformer layers:** each of 24 layers combines attention and a feed-forward network, with normalization and residual connections. Attention uses earlier context; the feed-forward network transforms the representation. Causal masking prevents a position from attending to future positions.
3. **Final normalization and projection:** convert the final hidden representation into one score per vocabulary token. These scores are called logits.
4. **Selection:** choose the next token. Sangama currently uses greedy selection: choose the largest logit. Other systems can sample from a probability distribution, but that is not our current generation policy.
5. Append the chosen token and repeat until an end-of-sequence token or output limit is reached.

The command may call this operation “sampling” even when selection is greedy. A generated answer is not a database lookup and is not guaranteed to be correct.

## Prefill, decode and the KV cache

**Prefill** processes the prompt and builds the initial attention state. **Decode** processes subsequent tokens one step at a time. Since the next token depends on the previous result, one conversation has an inherently sequential loop.

Each attention layer caches its past keys and values: the **KV cache**. This saves recomputing attention projections for all earlier tokens at each step. The cache grows with context length and belongs to a particular conversation and position. It is separate from the weights, which can stay loaded across conversations.

A worker therefore needs memory for weights, KV state, intermediate tensors and loading/compute workspace. BF16 checkpoint bytes are not a runtime RAM estimate: this implementation converts weights to F32. Quantization would reduce numerical precision/storage using suitable kernels; it is not implemented here.

## How two devices run one model

Sangama uses a contiguous layer split:

```text
Token IDs -> worker 0: embedding + layers [0,12)
          -> hidden activations over the network
          -> worker 1: layers [12,24) + final projection
          -> selected token returns to the client
          -> repeat with the next position
```

`[0,12)` means layers 0 through 11. Workers keep weights and KV caches for their own layers. They send **activations**, the temporary results of computation, rather than sending model weights on every token.

The download script creates separate physical shard files. Worker 0 opens its shard; worker 1 opens its own. The client using existing workers needs metadata and a tokenizer, not the full weights. Qwen ties its input embedding and output projection weights, so the endpoint matrix is duplicated in these two physical shards. Total shard disk space therefore exceeds the original checkpoint size.

A node can hold only part of a model, but “any tiny part” is not automatically useful. The current allocator chooses among prepared shards; it cannot invent smaller layer splits or divide individual tensor operations. Endpoints can be large because of embeddings, and every extra stage adds communication overhead. Phone clients and arbitrary Qwen/Kimi models are not supported.

This is **layer/pipeline partitioning**, although a single request traverses its stages sequentially. It is different from tensor parallelism, where devices cooperate inside an operation, and data parallelism, where separate full-model replicas serve different requests. Mixture-of-experts models introduce routed expert computation; our current dense Qwen backend does not implement that.

## Why more devices do not automatically mean faster output

For one decode step, an approximate model is:

`token time = sum(stage compute) + activation/result transfers + queueing + protocol overhead`

More devices may let a model fit in memory, while making each token slower. Adding stages does not make their dependent computations simultaneous. Increasing throughput with multiple conversations requires scheduling/batching that the current single-session workers do not provide.

At width 896, one F32 activation vector is `896 × 4 = 3,584` bytes. A 512-token prefill tensor is `512 × 896 × 4 = 1,835,008` bytes before framing. Decode is sensitive to latency; prefill can be sensitive to bandwidth. Ordinary generation selects the token at the final worker, avoiding returning all 151,936 vocabulary logits (607,744 F32 bytes) at every step. Verification deliberately returns full logits to compare numerical results.

Measure time to first token separately from decode tokens/second. Download/startup/loading, cold caches, prompt length, context length and backend affect different parts of the result. Docker on one Mac cannot establish performance between continents.

## Discovery, identity and reachability are different problems

| Concept | What it answers in Sangama |
|---|---|
| Peer ID | Which cryptographic identity am I talking to? |
| Membership | Is that identity admitted to this network, in this role, now? |
| DHT | Which signed shard offers have peers published? |
| Bootstrap | How does a new node initially find network peers? |
| Relay | How can two peers communicate when direct inbound connections are unavailable? |
| Placement | Which eligible nodes should load the prepared shards? |
| Reservation | Who may use/change a worker right now? |
| Readiness | Did the requested model actually load and report the expected identity/range? |

The DHT is a distributed key/value discovery mechanism, not a database containing all model weights. Each node persists its local records in SQLite. PostgreSQL serves a different purpose: portal membership, invitations and administrative directory state.

Home routers commonly use NAT, so a local listening port need not be reachable from the Internet. Relay circuits provide a path through a reachable relay. AutoNAT/DCUtR support is enabled for reachability/hole-punching attempts, but the current acceptance simulation verifies forced relay paths, not every router's NAT behavior.

## Trust and failures

Noise encrypts peer traffic in transit. An inference worker still sees the data it must process; encryption does not hide those activations from that worker. A signed advertisement proves who signed a claim, not that the claimed hardware exists or that its computation is honest. Start with trusted invited groups.

If a worker disappears, its KV cache is also unavailable. Another copy of the weights alone cannot continue at the same token position. Current recovery reloads/reallocates and starts a new request; it does not migrate KV caches or transparently resume a failed generation.

Next: [trace the actual architecture](architecture.md), then [run the local development path](../CONTRIBUTING.md#local-development).
