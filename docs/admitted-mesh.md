# Invited Internet mesh (experimental)

Sangama now has an admitted libp2p transport for the pinned Qwen2.5-0.5B model.
Workers keep their HTTP API on loopback. A local bridge carries bounded binary
requests over Noise + Yamux, either directly or through Circuit Relay v2. The
existing OpenCode-compatible chat API remains local. No SSH tunnel is required.

This is a trusted-group prototype, not an anonymous production network. Workers
can inspect the data they process. Signatures authenticate claims, not inference
correctness. The included simulation does not establish all real NAT behaviors.

## Membership and authority

PostgreSQL owns invitation hashes and member roles/expiry/revocation. Ed25519 peer
keys stay on their nodes. The portal issues a 60-second nonce for a single-use
invitation; the node signs a domain-separated message binding the network, nonce,
and invitation. Atomic SQL consumes the invitation and nonce together. Roles are
`worker`, `client`, and `relay`, with 24-hour membership. An operator can issue a
new invitation to renew a member. Directory registration is a separate operation.

The portal signs snapshots valid for 15 seconds. The snapshot endpoint publishes admitted peer IDs, roles, and expiry times; it contains no invitation or private key. Nodes pin its public key, check
the network ID and signature, refresh every three seconds, and disconnect peers
when membership expires or is revoked. Loss of the authority fails closed, so it
is an availability dependency. With synchronized clocks, revocation normally
propagates in a few seconds; the maximum stale-snapshot window is approximately
16 seconds, plus clock skew (five seconds is accepted when checking issuance).

Generate a dedicated authority key after building the portal image:

```sh
python3 deploy/portal/prepare-secrets.py
docker run --rm --network none --user 0:0 \
  -e MEMBERSHIP_KEY_FILE=/keys/membership_key \
  --mount type=bind,src="$(pwd)/deploy/portal/secrets",dst=/keys \
  sangama-portal:local authority-init
```

The host secret directory must remain 0700. Compose mounts only the named private
key into the portal. Distribute `membership_key.pub` to peers through a trusted
channel; never distribute the signing key. HTTPS is mandatory outside isolated
simulations. `test_http` and `SIMULATION_HTTP=1` disable that requirement for tests
and must not be enabled on a public deployment.

Operator commands inside the portal service:

```sh
docker compose exec portal sangama-portal network-invite worker
docker compose exec portal sangama-portal network-invite client
docker compose exec portal sangama-portal network-invite relay
docker compose exec portal sangama-portal revoke PEER_ID
```

Invite output is a secret: deliver it privately and store it in a 0600 file. It
must not appear in command arguments, Git, browser URLs, or test reports.

## A node configuration

Build on a Mac with `./scripts/cargo build --release --features metal`. Linux CPU
workers omit `--features metal`. Use absolute paths in configuration files.

```sh
target/release/sangama mesh-identity --state-dir .mesh/alice
```

This creates a private persistent identity and prints its public peer ID. The
operator supplies the pinned relay address and a consistent bridge map. Example
worker 0 configuration (replace placeholders):

```json
{
  "state_dir": "/absolute/path/.mesh/alice",
  "authority_file": "/absolute/path/authority.pub",
  "network": "sangama-private-v1",
  "portal": "https://YOUR_PORTAL_HOST",
  "listen": "/ip4/0.0.0.0/tcp/9000",
  "relay": "/ip4/RELAY_IPV4/tcp/9000/p2p/RELAY_PEER_ID",
  "worker": "127.0.0.1:7900",
  "managed": {
    "model_dir": "/absolute/path/prepared-model",
    "device": "metal",
    "memory_budget_mib": 8192
  },
  "token_file": "/absolute/path/.secrets/alice-worker-token",
  "bridges": [
    {"listen": "127.0.0.1:7901", "peer": "WORKER_0_PEER_ID"},
    {"listen": "127.0.0.1:7902", "peer": "WORKER_1_PEER_ID"}
  ]
}
```

Each machine uses its own random local worker token (at least 16 characters;
recommend 32 random bytes encoded as hex), stored in a 0600 file. Tokens are not
sent between peers. The same alias port must refer to the same peer on every
machine in a route. Alias addresses are always loopback and never advertised as
remote HTTP endpoints. Current direct dialing supports explicit public IPv4/TCP;
DNS and arbitrary private/link-local targets are rejected at the TCP transport
boundary, including addresses learned through Kademlia. The exact configured
relay socket is an operator-authorized exception for private-network tests.

```sh
target/release/sangama mesh-join --config alice.json \
  --invitation-file /absolute/path/.secrets/invitation
python3 scripts/mesh-node.py --config alice.json
```

Each worker uses its own key/token and the same bridge map. Managed workers start
unloaded. Stage one or more allowed physical shard files plus `manifest.json` and
`config.json` on each node; the allocator chooses among files actually available
there and loads only the selected shard. It will not download arbitrary files or
repartition an unknown checkpoint. The existing fetch/shard tools prepare files.
For manually loaded workers, omit `managed` and use `--shard INDEX --device metal`.

For a relay, use `"relay_server": true`, omit `relay` and `worker`, use an empty
bridge list, and set `external` to its reachable `/ip4/IP/tcp/9000` address. Only
the relay needs an inbound public TCP port. Keep PostgreSQL and worker ports
private. Do not expose any localhost API. A client-only configuration omits
`worker` and retains the bridge map.

## Placement, reservations, and OpenCode

`mesh-allocate` probes managed nodes before loading. Each reports its measured
available memory, configured contribution budget, prepared shard files and current
assignment. A bounded assignment solver finds complete shard coverage with distinct
physical peers, respects memory budgets, and minimizes summed probe time. It
reserves every chosen node, asks each to load its assigned hash-verified file, checks
readiness, then releases the placement leases. A placement lease blocks remote
inference while weights are changing and expires after 120 seconds if abandoned.
Failed allocations release acquired leases; verified weights can remain resident.

`mesh-plan` selects a route among already loaded, ready shards. Both commands are
limited to the prepared manifest; neither creates arbitrary layer boundaries or
assumes total network RAM is a single memory pool. Probe time is gateway-to-peer
round-trip time, not a measured pairwise bandwidth/compute model.

Workers measure available host memory (including Linux cgroup limits), count Mac
unified memory once, and check a conservative F32 weight + KV-cache + workspace
estimate before loading. Generation reserves every chosen worker before advancing
any KV cache. Reservations expire after 60 seconds without use. Partial admission
is released on failure. Sessions stay on their route; no mid-session KV migration
is attempted. On disconnect, retry as a new request after the worker recovers.

```sh
target/release/sangama --token-file /path/client-worker-token mesh-allocate \
  --model-dir /path/metadata --candidates 127.0.0.1:7902,127.0.0.1:7901
target/release/sangama --token-file /path/client-worker-token mesh-plan \
  --model-dir /path/metadata --candidates 127.0.0.1:7902,127.0.0.1:7901
python3 scripts/mesh-node.py --config client.json --chat --device metal \
  --model-dir /path/metadata --api-token-file /path/separate-chat-token
```

Alternatively, start the client mesh without `--chat`, then launch the existing pinned OpenCode integration:

```sh
python3 scripts/opencode.py --device metal \
  --peers 127.0.0.1:7901,127.0.0.1:7902 \
  --worker-token-file /path/client-worker-token
```

Do not start two gateways on port 8090. Point the existing OpenCode provider at `http://127.0.0.1:8090/v1`, using the
separate chat token and the model ID returned by `/v1/models`. The gateway starts
only after a complete ready route is available; the launcher attempts managed
allocation when the prepared model is not yet loaded. OpenCode remains text-only for
this small model; tools and autonomous file editing remain disabled.

Normal generation greedily samples at the tail worker and sends back one token.
`qwen-test` retains full logits for independent numerical verification. Mixed old
and new workers are not supported; update every node together.

## DHT and limits

The admitted mesh has a separate `/sangama/admitted-kad/1` namespace. Signed offers
expire after 60 seconds and bind the network, peer ID, model hash and shard.
Kademlia records persist in each node's `mesh-discovery.sqlite`; PostgreSQL is not
the DHT. The authenticated loopback `/v1/mesh/offers` endpoint exposes discovered
offers. Offers are checked again against current membership and live worker
metadata before use. The legacy private-overlay DHT remains available separately.

Current bounded test-network limits:

- 64 established connections, four per peer, 16 pending inbound/outbound.
- 64 relay reservations, one per peer; 32 circuits, four per peer.
- Each relay circuit: 120 seconds and 128 MiB, plus libp2p reservation/circuit rate limits.
- 4 MiB inference frames, 16 outbound RPCs, eight concurrent local worker forwards.
- 1,200 RPC requests and 128 MiB request/response payload per peer per minute.
- 120 inbound advertisement writes per source peer per minute; 8 KiB signed records.
- 1,024 persisted offers/members; periodic discovery currently probes at most 32 workers.

Do not interpret those ceilings as demonstrated scaling capacity. Initial testing
is a handful of invited machines. A circuit limit may interrupt a long request;
a new request can reconnect, but there is no transparent session continuation.
Relay listeners retry after closure. AutoNAT and DCUtR are enabled unless
`force_relay` is explicitly set for deterministic relay testing.

## Reproduce the simulation

```sh
./scripts/cargo build --features metal
./scripts/cargo test --features metal
./scripts/cargo test --manifest-path crates/network-auth/Cargo.toml
./scripts/cargo test --manifest-path portal/Cargo.toml
docker build -f docker/Dockerfile -t sangama:mesh-test .
docker build -f deploy/portal/Dockerfile -t sangama-portal:mesh-test .
docker build -f docker/Dockerfile.netem -t sangama-netem:test .
python3 scripts/fetch-test-opencode.py
python3 scripts/test-mesh-containers.py
```

The harness gives the workers and client three separate internal Docker networks.
They cannot dial one another directly. The relay is attached to all three; the
portal also has access for membership, and PostgreSQL has a separate private
network. No test ports are published. Worker containers are non-root, read-only,
and drop all capabilities. A short-lived NET_ADMIN helper adds netem only to the
worker network namespace. Real weights are mounted by shard; the client has only
metadata. The harness removes its containers, volumes, networks, identities and
invitations; it retains redacted operational logs and `runs/mesh-simulation/report.json`.

This simulates unavailable inbound connectivity and a delayed, bandwidth-limited
relay path. It does not emulate every CGNAT mapping behavior or prove DCUtR works
between two particular home routers. Real two-Mac WAN acceptance remains a
separate deployment test. No Linode server is provisioned by this harness.
