# Contributing to Sangama

Sangama runs a pinned language model across Rust workers, with an invited, encrypted peer network. Start here even if you have not worked on LLMs or peer-to-peer systems before.

## Reading path

1. [Fundamentals](docs/fundamentals.md): tokens, weights, layers, inference, KV caches and distributed execution.
2. [Architecture](docs/architecture.md): components, request flow, trust boundaries and source map.
3. [Local development](#local-development): build and run without a cloud account.
4. [Admitted mesh setup](docs/admitted-mesh.md): invitations, relay, managed workers and OpenCode.
5. [Recorded acceptance results](docs/test-results/mesh-acceptance.md): what was tested and what remains unproven.

There are three related but separate systems: real Qwen inference, the admitted mesh that transports it, and a deterministic numerical fixture used for networking experiments. A fixture pass is not evidence of LLM correctness. The legacy private-overlay DHT/UI also remains separate from the admitted mesh.

## Local development

Run commands from the repository root. Install stable Rust/Cargo, Python 3 and curl. Native compilation requires your platform's compiler/linker tools (Xcode Command Line Tools on macOS). Metal builds require an Apple environment with the Metal toolchain available. CUDA builds require Linux with an NVIDIA driver and the CUDA toolkit (`nvcc` on `PATH`); otherwise Linux uses CPU. Docker is optional until you run container tests. PostgreSQL is needed for the portal, not for basic local inference or Rust tests.

`scripts/cargo` uses the repository's isolated toolchain if present, otherwise your installed Cargo. The root, `portal/`, and `crates/network-auth/` have separate Cargo manifests; they are not one Cargo workspace, so check all affected packages explicitly.

Start without downloading weights:

```sh
./scripts/cargo build --locked
./scripts/cargo test --locked
./scripts/cargo run --locked -- doctor
./scripts/cargo run --release --locked -- demo
```

`demo` runs a deterministic matrix fixture and reports passes per second. It is useful for learning routes and failure handling; it does not generate text.

Run a real model on CPU:

```sh
python3 scripts/fetch-qwen.py
./scripts/cargo build --release --locked
./target/release/sangama generate --device cpu \
  --prompt 'Explain peer-to-peer computing in one short sentence.' \
  --max-tokens 40 --output runs/first-generation.json
```

On an Apple Silicon Mac, build with `--features metal` and use `--device metal` instead; on an NVIDIA machine, use `--features cuda` and `--device cuda`. Use release builds for inference measurements. The downloader verifies the pinned checkpoint and creates physical shard files; allow about 2.25 GB of disk space for the checkpoint plus shards, and additional runtime RAM. Without `--peers`, generation launches local workers and cleans them up afterward. This is two processes on one machine, not a WAN benchmark.

For independent numerical validation, run `qwen-test` with the same device and prompt. It requires the complete checkpoint and compares distributed tokens/logits against upstream Candle. See [generation](docs/generation.md) for role-specific file requirements.

## Choose a contribution

| Change | Start reading | Useful validation |
|---|---|---|
| Model math, attention or KV state | `src/qwen/model.rs` | `tests/qwen_layers.rs`, then real `qwen-test` |
| Generation, EOS or cleanup | `src/qwen/runner.rs`, `src/qwen/network.rs` | `tests/generation.rs`, real generation regression |
| Frame format and bounds | `src/qwen/wire.rs` | Wire/network tests, all-node compatibility review |
| Invitations and revocation | `crates/network-auth/src/lib.rs`, `portal/src/membership.rs` | Auth/portal tests, mesh simulation |
| Contribution credits | `src/credits.rs`, `crates/network-auth/src/credits.rs`, `portal/src/credits.rs` | Unit tests, `scripts/test-credits-local.py` |
| Relay and peer transport | `src/mesh.rs`, `src/mesh_transport.rs` | Transport tests, forced-relay simulation |
| Signed discovery | `src/mesh_store.rs` | Signature/expiry/persistence tests, simulation |
| Allocation and model loading | `src/mesh_allocate.rs`, `src/managed_worker.rs`, `src/resources.rs` | Placement tests, cold-start simulation |
| OpenCode/chat | `src/chat_api.rs`, `scripts/opencode.py` | Chat tests, actual OpenCode simulation |
| Hosted directory UI | `portal/src/main.rs`, `portal/schema.sql` | Portal tests and local preview |
| Numerical fixture | `src/kernel.rs`, `src/server.rs`, `src/planner.rs` | Correctness, network and process tests |

Good first changes include improving an error message with actionable context, adding a regression for an uncovered failure, or fixing a documented setup problem. Model/backend support and public-network policy changes need a design explanation and measured validation.

## Checks before submitting a change

```sh
./scripts/cargo fmt --all -- --check
./scripts/cargo clippy --locked --all-targets -- -D warnings
./scripts/cargo test --locked
./scripts/cargo test --locked --manifest-path crates/network-auth/Cargo.toml
./scripts/cargo test --locked --manifest-path portal/Cargo.toml
```

For Metal or CUDA changes, add `--features metal` or `--features cuda` to the root clippy/test commands. The CUDA build cannot be checked on a Mac or on GitHub's hosted runners, so record the GPU, driver and CUDA versions you tested on. For changes in the separate packages, also run fmt/clippy with their `--manifest-path`. Real checkpoints are not downloaded by ordinary Rust tests. Follow [the Docker simulation instructions](docs/admitted-mesh.md#reproduce-the-simulation) for networking, membership, placement or recovery changes; prepare the model first. The current full simulation uses a Linux ARM64 OpenCode binary, so do not assume it runs unchanged on every Docker architecture.

Record what passed, the backend/hardware, and any untested paths. Do not turn a local single-run speed result into a WAN performance claim. A useful change description states the triggering problem, new behavior, validation and remaining limitations. Keep unrelated refactors separate. Update docs when changing configuration or the wire protocol; mixed old/new workers are not supported.

## Invariants to preserve

- Never skip a missing layer or silently route to a different model. Require the pinned manifest and complete ordered coverage.
- Authenticate membership and validate advertisements; signatures establish authorship, not truthful compute capacity or correct inference.
- Keep worker/chat HTTP endpoints on loopback. Remote traffic uses the admitted transport or the explicitly documented private tunnel workflow.
- Check frame bounds, shapes, finite values, route destinations, session ownership and resource budgets before use.
- Reserve before mutating KV state. Release owned reservations on errors; abort a failed session rather than pretending it continued.
- Keep local bearer tokens, peer private keys and the authority signing key out of logs, reports and Git. Local tokens must not be sent across the mesh.
- An expired authority snapshot must fail closed. Do not add an availability fallback that silently bypasses membership checks.

Generated weights, keys, `.mesh/`, `.tools/`, `runs/` and `work/` are ignored. Publish only reviewed, non-secret reports under `docs/test-results/`. Follow the repository owner's sharing instructions; the current development workflow keeps changes local and does not push.

## Troubleshooting

- **No complete route:** ensure every prepared shard is present on some eligible worker, budgets fit, and all workers match the manifest/backend. Use `mesh-allocate` for unloaded managed nodes and `mesh-plan` for loaded ones.
- **Unauthorized or immediately disconnected:** check the pinned authority key, network ID, membership expiry/revocation, portal reachability and system clock. Directory registration alone is not mesh admission.
- **Connection works but forwarding fails:** confirm the same loopback alias port maps to the same peer on every node; verify local token files and downstream allowlists.
- **409/busy:** another inference or placement lease owns the worker. Finish/reset that session or wait for its lease to expire; do not steal its KV cache.
- **Memory rejection:** reduce the contribution requirement by using a supported smaller prepared shard/model arrangement, or provide more available memory. Disk size is not the RAM requirement.
- **Simulation failure:** inspect `runs/mesh-simulation/report.json` and retained logs. The harness cleans its containers and temporary credentials; a failed check must remain visible in the report.

For deployment operations, use [the portal guide](deploy/portal/README.md) and [mesh guide](docs/admitted-mesh.md). Real two-home-network testing remains an acceptance step, not an assumed property of passing Docker tests.
