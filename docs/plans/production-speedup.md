# Plan: production-grade speed for the relayed pipeline

Written 30 September 2026, after rounds 7–9. It starts from measurements, names each bottleneck, records what research found about it, and splits the work into streams that can be built and measured on their own.

## Where the time goes

Measured on the round 7–9 fleet (20 stages spread over three US states, one relay). One plain pass, which yields one token, takes 800 ms:

| Part | Time | Share | Evidence |
|---|---|---|---|
| Distance | ~465 ms | 58 % | Sum of each worker's round trip to the relay. On round 6's co-located fleet the whole pass was ~190 ms. |
| Per-hop overhead | ~310 ms | 39 % | Pass time minus distance minus compute: about 15 ms for each of 21 hops. |
| GPU compute | 24 ms | 3 % | Stage traces. |

Two further limits:

| Limit | Cost | Evidence |
|---|---|---|
| Snapshot before each drafted pass | ~730 ms per pass, about 36 ms per stage | With 2 drafts and 41 of 42 accepted, a pass took 1,528 ms against 800 ms plain, although model compute was 43 ms. |
| Throughput ceiling | ~90 passes per second in total | CPUs 10–20 % busy and GPUs under 7 % at the ceiling. 51 sessions gave 91 passes/s; 64 gave 56. |

## Status after round 10

Round 10 tested these streams on three 96 GB cards in one datacenter. Results are in [the round 10 report](../test-results/qwen35-397b-3stage-2026-09-30.md).

| Stream | Status |
|---|---|
| A. Cheap rollback | Done and measured: 0.2–0.4 ms per pass inside workers, rollback included. |
| B. Direct connections | Works after a fix (nodes dial advertised public addresses). Gain across distance not yet measured. |
| C. Placement | Tool built and unit-tested (`scripts/route-order.py`). Not yet used on a fleet with distance. |
| D. Throughput ceiling | Instrumentation done and circuit cap lifted. It located a different ceiling on the close fleet: the last stage's GPU, one frame at a time. The twenty-stage ceiling is still open. |
| E. Per-hop overhead | Cheap clean-up only. Measured 1.3 ms per hop on the close fleet. |
| F. Fewer, larger stages | Done by layout: 3 stages ran one request at 48 tok/s, 107 with drafts. |
| G. Batching | Built (round 11): plain throughput 119 to 403 tok/s at 48 requests, 449 at 64. Drafts with batching were wrong on the fleet at first; the fix was validated in round 12, where drafts under load ran slower than plain (174 against 233 tok/s at 16 requests). |

## Work streams

Each stream has one owner-sized goal, a way to measure it, and no dependency on the others unless stated.

### A. Cheap rollback of drafts (done; measured on the fleet in round 10)

- **Finding:** llama.cpp keeps the last `n_rs_seq` recurrent states of a sequence on the device and rewinds with one `llama_memory_seq_rm`. No state is copied and nothing is decoded again. Its own server does MTP rollback this way. Attention KV is trimmed by the same call.
- **Built:** `qwen-worker --rollback N` (commit `20c66e6`). The saved-state path remains for engines that cannot rewind.
- **Local result:** Qwen3.5-0.8B across two workers, 4 drafts: a drafted pass fell from 126 ms to 61 ms, output identical to plain decoding.
- **Cost:** one more recurrent state per slot per rewindable position, about 12 MB each on a 397B stage. With 4 drafts that is ~60 MB per session, so a 16 GB stage holds about 40 sessions rather than 96.
- **Fleet gate:** a drafted pass within 10 % of a plain pass. If it holds, 2.7–5.3 tokens per pass becomes 2.5–5× for one request.
- **Open:** llama.cpp issue #23322 reports acceptance collapsing after a checkpoint desync on a hybrid model. Watch the acceptance rate, which was 35–98 % across round 8's runs.

### B. Direct connections, relay as fallback

- **Finding:** every hop crosses the relay twice. `force_relay` turns off hole punching (DCUtR), AutoNAT and direct dials, and hides listen addresses. A measurement over 4.4 million attempts found DCUtR succeeds 70 % of the time, equally for TCP and QUIC, almost always on the first try (Trautwein et al., arXiv 2510.27500).
- **Do:** let nodes with reachable ports dial each other; try DCUtR for the rest; keep the relay for pairs that fail. Check that request-response uses the direct connection once both exist. Prefer routes whose neighbours connect directly.
- **Expected:** about 1.5–2× on the distance term, and it removes the single relay as a bandwidth funnel (about 170 Mbit/s each way at 90 tokens/s).
- **Gate:** median hop time between two directly connected stages close to half their relayed time.
- **Risk:** the 30 % of pairs that stay relayed; QUIC is not compiled in.

### C. Placement and route order

- **Finding:** the chain costs the sum of neighbour-to-neighbour delays, so the order of machines matters. Petals routes each client by shortest path over measured latency plus compute and rebalances servers toward the bottleneck; Parallax groups by region and prefers fewer stages. Round 6 to round 7 showed the size of the effect here: 190 ms against 800 ms per pass with the same software.
- **Do:** measure pairwise delay with a real reply, not a TCP connect (a host proxy faked that in round 7); order the chain with nearest-neighbour plus 2-opt; record each machine's listing end date and move its stage before it ends.
- **Expected:** 1.3–2× on the distance term for a spread fleet; nothing for one already co-located.
- **Gate:** predicted pass time from the delay table within 15 % of the measured one.

### D. Find and remove the throughput ceiling

- **Finding:** a code audit found no extra round trip, sleep, per-request dial or lookup in the frame path, and lazy stream negotiation is already on. The ceiling is not explained by code reading. Suspects, all unconfirmed:
  - relay bandwidth or TCP congestion on one connection per node;
  - one saturated thread (the mesh event loop or the engine lock) that shows as low total CPU on a many-core host;
  - circuits ending: `max_circuit_bytes` is 1 GiB, which one circuit reaches after roughly 100,000 frames. A circuit that ends mid-run would also explain why 64 sessions ran slower than 51.
- **Do first, before changing anything:** three timestamps per frame (bridge receive, remote mesh receive, compute start); engine lock wait and hold time; per-thread CPU on a mesh process; relay interface counters and TCP retransmits during a 50-session run.
- **Then:** raise or remove the circuit byte limit; fix whichever suspect the numbers show.
- **Gate:** total throughput keeps rising to at least 96 sessions.

### E. Fewer wake-ups and copies per hop

- **Finding:** a hop passes through three processes with about ten task wake-ups, two loopback HTTP calls, five header parses and six copies of the frame. This is the likeliest home of the ~15 ms per hop, but unmeasured.
- **Do, in order of cost:** parse the header once per hop; run the mesh inside the worker process; one long-lived stream per neighbouring pair with its own framing; let the last stage push the result instead of being polled.
- **Expected:** up to the 310 ms of per-hop overhead on a spread fleet, less on a co-located one where it was a few ms per hop.
- **Gate:** stream D's timestamps show local handling under 2 ms per hop.

### F. Fewer, larger stages

- **Finding:** network time is roughly linear in hops. Machines with 24–48 GB can hold two or three of today's stages.
- **Do:** let the planner give a machine as many layers as its memory allows, within one process.
- **Expected:** 20 stages to 10 halves network time for that route.
- **Risk:** a lost machine takes more of the model with it.

### G. Batching sessions inside a stage

- **Finding:** as vendored, llama.cpp runs one sequence per micro-batch on 19 of our 20 stages, because a stage that returns hidden states takes the `split_seq` path. True batching needs a llama.cpp patch, and changes numerics (see below).
- **Do later:** first drain whatever frames queued while the GPU was busy into one call, with no timer. Patch the split only if stream D shows the engine lock is the limit.
- **Expected:** small until D and E are done, since compute is 3 % of a pass.

## Order

1. **A on the fleet** and **D's measurements**, in one short fleet session. A is built; D decides what to fix next.
2. **B and C**, which attack the largest share of a pass.
3. **E**, guided by D's numbers.
4. **F**, then **G**.

## What the numbers could become

Estimates, not measurements. They assume round 6's placement (190 ms per plain pass).

| Setup | One request | Basis |
|---|---|---|
| Round 6, measured | 5.0–5.5 tok/s | – |
| + drafts with cheap rollback (A) | 14–28 tok/s | 2.7–5.3 tokens per pass at about the cost of a plain pass |
| + direct connections (B) | 20–45 tok/s | distance term roughly halved |

Total throughput across users is about 90 passes per second today. With drafts that is roughly 250–450 tokens per second if the ceiling counts passes, and more once D removes it.

## Correctness rules for all streams

- Output is deterministic for a given batching, but changes with it. Compare like with like: a request against the same request run alone, or drafted against drafted.
- A stream is done when its gate is met on the fleet, not locally.
- Record acceptance rate, tokens per pass and per-stage time with every drafted run.

## Sources

- llama.cpp `n_rs_seq`: `include/llama.h`, `src/llama-memory-recurrent.cpp` (`seq_rm`), `tools/server/server-context.cpp`.
- Hole punching measurement: Trautwein et al., arXiv 2510.27500.
- Petals: Borzunov et al., arXiv 2312.08361. Parallax: arXiv 2509.26182.
- Distributed speculative decoding: arXiv 2511.11733. Speculative Pipeline Decoding: arXiv 2605.30852.
- llama.cpp on batching and determinism: server README; issues #7052 and #23322; PR #22673.
