# Sangama

**Sangama (संगम)** — a confluence of devices, running one model together.

A Rust foundation for running model pieces across participating computers.

**Status: real Qwen text generation works across separate Rust worker processes, using Candle on Metal or CPU.**
The original deterministic numerical fixture remains available for networking tests.

## UI and distributed discovery

```sh
./scripts/cargo run --release --locked --features metal -- ui --device metal
```

Open the private localhost URL printed in the terminal. The UI supports bootstrap joining,
local shard advertisements, signed provider discovery, standalone Qwen generation, optional verification, and JSON report downloads.
Discovery uses **libp2p Kademlia over Noise/TCP**, with **SQLite-backed records** and persistent identities.
See [DHT architecture and setup](docs/dht.md), [secure peer testing](docs/secure-peer-test.md),
and [Docker isolation tests](docs/docker-testing.md) for real Qwen generation and DHT discovery in separate containers.
Discovered devices are not automatically authorized for inference; application invitations and automatic
shard assignment remain future work. The initial network uses private Tailscale connectivity.

## Generate text without a full local model

```sh
cd /Users/diljit/Documents/Projects/p2p-inference
python3 scripts/fetch-qwen.py
./scripts/cargo run --release --locked --features metal -- generate --device metal --prompt 'Explain peer-to-peer computing in one short sentence.' --max-tokens 40 --output runs/generation.json
```

`generate` starts two shard workers and produces text without opening the original `model.safetensors`
or loading an unsplit model on the client. Each worker loads its assigned shard and retains its own KV cache.
The UI defaults to **Generate text · workers only**. This is greedy generation, with the completed result
returned at the end; streaming display and final-worker token selection are not implemented yet.

With `--peers`, the client needs only `manifest.json`, `config.json`, and `tokenizer.json` in `--model-dir`:

```sh
./target/release/sangama generate --device metal --model-dir path/to/client-metadata --peers 127.0.0.1:7901,127.0.0.1:7902 --token-file .secrets/peer.token --prompt 'Explain peer-to-peer computing in one short sentence.'
```

Prepare the worker files and SSH tunnels using [the secure peer guide](docs/secure-peer-test.md).
Without `--peers`, the same directory must also contain all shard files for automatic local workers.
`--device` selects the local worker backend, or the expected backend reported by existing workers;
standalone clients do not initialize a GPU. The downloader prepares a complete checkpoint plus shards,
but you can distribute only the files required by each role.

## Optional correctness verification

```sh
./target/release/sangama qwen-test --device metal --prompt 'Explain peer-to-peer computing in one short sentence.' --max-tokens 40 --output runs/verification.json
```

`qwen-test` still requires the complete model locally. It runs the upstream Candle baseline, releases it,
then compares every generated token and logit from the shard workers. The UI offers the same explicit **Verify** task.
Generation reports have `operation: "generate"` and null baseline/comparison fields, never a fabricated verification pass.
Both modes clean up sessions and stop any workers they launched. Reports include `finish_reason` (`eos`, `max_tokens`,
or a verification mismatch). See [standalone generation details](docs/generation.md).

This is an actual Transformer checkpoint, not simulated text or calls to a hosted API.
Inference uses F32 on both CPU and Metal; weights are not quantized. Disk usage is about 2.25 GB
for the original checkpoint and two shards. The tied embedding matrix appears at both endpoints,
so two shards together are larger than the original file. These disk sizes are not runtime RAM estimates.
Model loading and file verification are excluded from token timings. Standalone generation performs no extra warmup;
verification warms up and resets the prompt first. Their latency measurements are therefore not directly comparable.

For CPU, omit `--features metal` and use `--device cpu`. Python and curl are used only to prepare model files.
The current test supports one active conversation per worker, greedy decoding, at most 512 prompt tokens,
128 generated tokens, and a 4,096-token worker context cap. The chat API accepts longer prompts using 512-token chunks. It does not yet support arbitrary Qwen/Kimi checkpoints.

### Two physical devices, including Internet peers

Follow [the secure peer test guide](docs/secure-peer-test.md). Qwen workers now **refuse non-loopback binds**
and require an explicit `--allow-next` destination to forward activations. Use pinned-key SSH tunnels over
Tailscale, private token files, and dedicated access restricted to the participating devices.
The guide includes enrollment, exact commands, revocation, and the remaining prototype limitations.
Do not use the old direct-LAN Qwen commands: they intentionally fail now.

The real-model route remains explicit; the fixture coordinator is not integrated with Qwen.
Both workers and the validation baseline currently need the same backend type (CPU or Metal).

## Numerical networking fixture

The fixture runs deterministic CPU residual matrices. It reports passes/second, never tokens/second.

### Run the fixture

On this Mac, a Rust toolchain is installed in `.tools/`. The wrapper uses it without changing your shell settings.
On other machines, install stable Rust using [rustup](https://rustup.rs/); the same wrapper uses system Cargo if no local toolchain exists.

```sh
cd /Users/diljit/Documents/Projects/p2p-inference
./scripts/cargo run --release -- doctor
./scripts/cargo run --release -- demo
```

The demo starts a coordinator and three workers on ephemeral loopback ports, registers their shards,
executes a direct worker-to-worker pipeline, and checks its output against an unsplit local baseline.
Everything exits after measurement. No model download or cloud account is required.

```sh
# Each worker adds 10 ms per pass. Compare with the zero-delay report.
./scripts/cargo run --release -- demo --workers 3 --delay-ms 10 --rounds 30 --output runs/delay-10ms.json

# More numerical computation per pass, still a fixture rather than an LLM.
./scripts/cargo run --release -- demo --layers 12 --width 512 --workers 3 --rounds 30
```

The demo uses separate async services in **one process**. It does not simulate the memory bandwidth or compute capacity of multiple machines.
The commands below run real separate processes and also work on separate trusted LAN hosts.

## Run separate processes

Build once:

```sh
./scripts/cargo build --release --locked
```

Use the **same** `P2P_TOKEN` value in every terminal. Generate a private development value once with
`openssl rand -hex 24`, then export that value in each terminal. Tokens are not printed in logs.

Terminal 1:

```sh
export P2P_TOKEN='paste-the-same-generated-token-in-every-terminal'
./target/release/sangama coordinator
```

Terminal 2:

```sh
export P2P_TOKEN='paste-the-same-generated-token-in-every-terminal'
./target/release/sangama worker --id mac-a --start 0 --end 6
```

Terminal 3:

```sh
export P2P_TOKEN='paste-the-same-generated-token-in-every-terminal'
./target/release/sangama worker --id mac-b --listen 127.0.0.1:7802 --start 6 --end 12
```

Terminal 4:

```sh
export P2P_TOKEN='paste-the-same-generated-token-in-every-terminal'
./target/release/sangama peers
./target/release/sangama plan
./target/release/sangama bench --rounds 30 --output runs/two-processes.json
```

Layer ranges are `[start, end)`: the end is exclusive. All workers must use the same `--layers` and `--width`.
Press Ctrl-C to stop each long-running process. A departed worker expires after its 15-second lease;
a request encountering a failed worker before expiry returns an explicit error. Automatic mid-request replay is not implemented.

## Another computer on a trusted LAN

Build the project natively on that computer. If the coordinator is at `192.168.1.10` and the other worker is at `192.168.1.11`, use:

```sh
# Coordinator host:
./target/release/sangama coordinator --listen 0.0.0.0:7800
./target/release/sangama worker --id first --listen 0.0.0.0:7801 --advertise 192.168.1.10:7801 --coordinator 192.168.1.10:7800 --start 0 --end 6

# Second host:
./target/release/sangama worker --id second --listen 0.0.0.0:7801 --advertise 192.168.1.11:7801 --coordinator 192.168.1.10:7800 --start 6 --end 12

# Client:
./target/release/sangama bench --coordinator 192.168.1.10:7800
```

Substitute actual private addresses and set the shared token. Peers need bidirectional access to their worker ports.
The transport is HTTP/JSON with a shared bearer token, **not encrypted transport or public P2P networking**.
Use only trusted private LANs or a private encrypted overlay; do not expose these ports publicly.
Numeric loopback, RFC1918 IPv4, and IPv6 unique-local addresses are accepted. Public and link-local addresses are rejected.
There is no mDNS, NAT traversal, relay service, or independent peer identity yet.

## Fixture capabilities

- Resident CPU shards containing only their assigned matrices.
- Authenticated registration and live probes, three-second heartbeats, and 15-second leases.
- Exact contiguous layer coverage with a cost-based route planner and alternate shard candidates.
- Direct activation forwarding between workers; the coordinator initiates the route.
- Input, route, model/version, memory-budget, address, and message-size validation.
- Bounded compute admission and explicit overload errors.
- Local versus distributed correctness checks, p50/p95 latency, worker compute timing, and JSON reports.
- Unit, socket-level integration, and separate-process tests.

The planner uses worker calibration plus **coordinator-to-worker** probes as a placement heuristic.
It does not yet have pairwise network measurements. `--delay-ms` adds a wait once per worker;
it is not an emulation of WAN RTT, loss, contention, or bandwidth.
Remaining reported overhead includes JSON/HTTP, queueing, transport, timer overshoot, and framework work.

## Project layout

```text
src/main.rs         CLI: qwen-test, qwen-worker, doctor, demo, coordinator, worker, peers, plan, bench
src/qwen/          Real Qwen shards, KV cache, binary transport, independent baseline checks
src/protocol.rs     Fixture messages and validation
src/kernel.rs       Deterministic numerical fixture with resident matrices
src/server.rs       Coordinator, workers, leases, authentication, direct forwarding
src/planner.rs      Contiguous coverage and cost-based placement
src/benchmark.rs    Baseline checking and timing reports
tests/              Correctness, socket integration, and process integration
docs/architecture.md  Current architecture and remaining milestones
```

## Development checks

```sh
./scripts/cargo fmt --check
./scripts/cargo clippy --locked --all-targets -- -D warnings
./scripts/cargo test --locked
```

## Next milestone

Measure two physical devices, add quantized weights and sample tokens at the final worker (the checker currently returns full logits).
Then integrate measured placement, resource admission, and recovery. Speculative decoding, mobile apps,
Kimi/MoE support, encrypted peer identity, and public volunteer participation remain future work.
See [the architecture plan](docs/architecture.md). Real-checkpoint tests are explicit commands, not part of CI;
the automated Qwen test uses small random weights and verifies prefill, decode, and cache reset against upstream Candle.
For Metal checks, add `--features metal` to the clippy and test commands above.

## OpenCode with Sangama

A local, text-only OpenCode integration now uses the real shard workers with streaming chat responses.
Run `python3 scripts/opencode.py` after building Sangama and installing the pinned client with
`./scripts/install-opencode.sh`. See [setup, tests, and current limits](docs/opencode.md).
The 0.5B model is a connectivity/demo model; automatic tools and file edits are disabled.

## Hosted HTML portal

The separate `portal/` Rust application provides an invitation-based device directory backed by private PostgreSQL.
See [deployment and local preview](deploy/portal/README.md). The hosted portal does not expose the local inference
control panel or replace the DHT. Linode configuration is prepared; provisioning needs a valid Linode credential.
