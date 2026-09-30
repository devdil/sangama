# Round 7 plan: concurrent sessions on the 397B relay fleet

## Question

How much total throughput can the round-6 fleet deliver when many clients generate at once?

Round 6 decodes one stream at 5.0–5.5 tok/s, with each token spending ~190 ms on the network and ~21 ms on the GPUs. Each stage is busy about 1 ms per token, so a single stream leaves every GPU idle more than 99 % of the time. A pipeline this deep should carry many streams at once. The target is **100 tok/s aggregate** (about 20 streams at ~5 tok/s each), while a single stream stays near its round-6 speed.

This does not make one stream faster. Per-stream speed needs fewer hops, a closer placement, or better speculative drafts (see the round-6 notes).

## What blocks it today

| Where | Limit |
|---|---|
| `crates/llama-stage/src/shim.c` | `n_seq_max = 1`; every token is written to sequence 0 |
| `src/qwen/network.rs` | One `Option<Session>` per worker; `try_lock` returns "worker busy" instead of queueing |
| `src/qwen/network.rs` (last stage) | Results go into a `watch` channel that holds one value, so a second session would overwrite the first before its client collects it |
| `src/qwen/network.rs` (outbox) | One queue per stage, sent one at a time and each waiting for the next hop's acknowledgement. At 20–40 ms per relayed hop, that caps a stage at ~25–50 frames/s across all sessions |

The mesh layer (`src/mesh_owner.rs`) already tracks claims per session, up to 64.

## Changes

1. **Slots in the llama.cpp stage.** Open the context with `n_seq_max = slots`, `n_ctx = 4096 × slots`, `kv_unified = false`, so each slot keeps its own 4096-token KV and recurrent state. Decode, greedy, save/load state and clear take a slot id; clear only removes that slot's sequence.
2. **Sessions per worker.** Workers take `--slots N` (default 1, which keeps today's behaviour). Each reserved session gets a free slot; reserving when all slots are in use returns 409 as today. Expiry, reset and discard free only that session's slot.
3. **Queue, don't refuse.** Forwards wait for the stage's engine lock instead of failing with "worker busy". Compute still runs one frame at a time per GPU, which is ~1 ms per decode frame here.
4. **Results per session.** The last stage keeps the latest result for each session, so concurrent clients collect their own.
5. **Outbox per session.** Frames keep their order within a session, and different sessions send in parallel.
6. **Memory budget.** The KV term in the memory check scales with slots.

Batching several sessions into one `llama_decode` call is left for later. At ~1 ms of compute per frame it is not the bottleneck until well past 100 tok/s.

## Local verification (done 30 Sep, Apple M5 Pro, Metal, Qwen2.5-0.5B F32)

- **Stage test** (`crates/llama-stage/tests/split.rs`): three sessions in slots 1–3, stepped in turn over one two-stage split, each gave exactly its solo tokens. Running and clearing slot 0 midway did not disturb them. Existing split and bad-input tests pass.
- **Worker processes:** two `qwen-worker --slots 4` processes. Four prompts, run first one at a time and then all four at once (twice): every concurrent run's tokens were identical to its solo run.
- **Slot limit:** four reservations succeed, a fifth gets `409 all of this worker's slots are in use`, and a reset frees a slot for it.
- Full `cargo test` suite, `cargo clippy` and `cargo fmt --check` pass.

**Verified on the 397B model (w00, stage 0, RTX 4090, 30 Sep):** `first_stage_slots_are_isolated` runs layers 0–2 (`qwen35moe`, recurrent) in four slots, interleaved, and every output is bit-identical to running one session at a time.

**Memory per slot.** Unpatched, llama.cpp gave each slot cache for all 60 layers: 186 MB of recurrent state and 120 MB of attention KV at 4096 tokens, about 306 MB, so 32 slots would not fit in 16 GB. `patches/stage-memory-layers.patch` limits the cache to the stage's own layers. Stage 0's four slots went from 745 MB of recurrent state to 50 MB (~12 MB per slot), and to 0 MB of KV, since layers 0–2 have no attention layer. Stages with one attention layer add ~8 MB per slot at 4096 tokens. 32 slots now need under 1 GB per stage.

## Fleet run

**Fleet:** the round-6 fleet (relay + w00–w19). w17–w19 are down as of 30 Sep 08:25 IST and must be replaced in the relay's datacenter, as in round 6.

**Deploy:** one CUDA build with `--slots 32` to every worker; mesh processes are unchanged.

**Measurements**, at concurrency 1, 2, 4, 8, 16 and 32, each client generating 128 tokens from the standard prompt set:

| Metric | Why |
|---|---|
| Aggregate tok/s | The headline number |
| Per-stream tok/s, p50 and p95 | How much each user loses |
| Time to first token, p50 and p95 | Prefill from one session blocks others' decode on a stage |
| Tokens identical to the solo run | Correctness under concurrency |
| Per-stage GPU utilisation, relay Mbit/s and CPU | Where it saturates |
| Stage queue wait (`forward_ms` vs wall time in traces) | Whether the engine lock becomes the limit |

**Expected:** near-linear to ~16 streams (≈ 80–90 tok/s), then limited by relay CPU or bandwidth. Round 6 used 2.3 Mbit/s on the relay per stream, so 32 streams need ~75 Mbit/s, well within its link.

**Success:** at least 100 tok/s aggregate at some concurrency, with every stream's tokens identical to the solo run.

**Stop conditions:** relay CPU above 90 %, or per-stream p50 below 2 tok/s. Record the point where it saturates rather than pushing past it.

## Cost and time

About 2 hours of fleet time at ~$13/hour: roughly 30 minutes to replace w17–w19 and deploy, 1 hour to measure, and the rest as margin. The fleet should be torn down, or at least stopped, while the code is built and verified locally.

## After this round

- **Per-stream speed:** Qwen3.5's multi-token-prediction draft layer in the published slices, plus snapshots that save only the recurrent state.
- **Batching within a stage:** one `llama_decode` across all waiting sessions, once concurrency saturates the per-frame path.
- **Placement:** a latency-aware scheduler, replacing the hand placement of rounds 5–6.
