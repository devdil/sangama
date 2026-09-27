# Numerical networking fixture

This is the legacy deterministic networking test, separate from real Qwen inference and the admitted mesh. Return to [the project overview](../README.md) or [contributor setup](../CONTRIBUTING.md).


The fixture runs deterministic CPU residual matrices. It reports passes/second, never tokens/second.

### Run the fixture

Install stable Rust using [rustup](https://rustup.rs/). The `scripts/cargo` wrapper uses system Cargo
unless an isolated toolchain exists in `.tools/`. Run the following commands from your repository checkout.

```sh
cd /path/to/sangama
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
This legacy numerical fixture does not use the admitted mesh transport or its identity/membership controls.

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

