# Sangama admitted mesh — simulated acceptance results

Tested locally on 27 September 2026. All 22 simulation checks passed.

## What ran

Actual Qwen2.5-0.5B-Instruct inference across two CPU containers on separate internal Docker networks, using an admitted Noise-encrypted Circuit Relay v2 path. Each worker had only its assigned physical shard. Workers began without weights loaded; resource-aware allocation reserved, loaded and checked both shards. The actual OpenCode 1.18.32 CLI used the local API, with tools and cloud fallback disabled.

PostgreSQL stored membership and invitations. Signed DHT offers persisted in each worker's local SQLite store. No host ports were exposed by the simulation.

## Validation

- All 22 end-to-end checks passed: identity proof, single-use invitation, outsider rejection, cold placement, relay-only connectivity, placement/session exclusivity, signed discovery, exact baseline token match, OpenCode, quotas, oversized request rejection, mid-generation disconnect, restart/reallocation, revocation, and authority expiry.
- All 29 root Rust unit/integration tests passed with Metal enabled. The shared authentication crate and portal tests also passed. Clippy passed; an optimized native Metal binary was built.
- Unit tests additionally exercised forged, expired and wrong-network advertisements, unsafe transport addresses, and invalid placement choices.

## Single-run CPU measurements

| Forced relay condition | Decode tokens/second |
|---|---:|
| No added impairment | 7.31 |
| Each worker egress: 25 ms delay, 5 ms jitter, 20 Mbit/s | 2.73 |

The prompt produced 20 tokens including EOS. Both runs exactly matched the independently recorded baseline token IDs. These are illustrative measurements on one Docker host, not predictions for two remote Macs. Decode rate excludes the first token and model loading. Other local build activity and cache warmup can affect timings.

The nested generation JSON retains the generic standalone command's null verification fields and legacy SSH transport note. The harness performs the baseline comparison and verifies relay topology separately; its top-level checks are the acceptance results.

## Remaining acceptance work and limits

A real two-Mac test on separate home networks and a publicly reachable HTTPS authority/relay deployment are still required. The simulation forces relay traffic; it does not prove successful hole punching through particular routers. No Linode server was provisioned in this change.

Placement selects existing prepared physical shards using measured available memory and probe latency; it does not repartition or download arbitrary models. A failed generation is aborted; recovery starts a new session after reallocation, without KV-cache migration. Membership expiry deliberately makes authority outages interrupt service. Participating workers can inspect their inference data. This remains a trusted-group prototype, not production certification or an anonymous public network.

## Reproduce

Follow `docs/admitted-mesh.md` in the repository. The harness is `scripts/test-mesh-containers.py`; its detailed result is `docs/test-results/mesh-simulation.json`.
