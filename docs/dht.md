# Kademlia discovery and the local UI

This repository now runs a real libp2p Kademlia DHT. SQLite is the persistent local record store,
not a central database shared by the network. Each node has its own database, routing state,
and persistent Ed25519 identity. Nodes replicate metadata through Kademlia over Noise-authenticated,
encrypted TCP connections with Yamux multiplexing. The protocol is `/mesh/kad/1.0.0`.

## Open the UI

```sh
./scripts/cargo run --release --locked --features metal -- ui --device metal
```

Open the private localhost URL printed by the command. Its browser capability is distinct from the peer token.
The UI can join via a bootstrap multiaddress, verify and advertise a local Qwen shard, and search the DHT
for model providers. It also runs real local/split inference verification and downloads a report.
Peers found in the DHT are **not automatically approved or attached to an inference route**.
To use already-approved inference workers, add `--peers 127.0.0.1:7901,127.0.0.1:7902 --token-file .secrets/peer.token`.

The default DHT listener is an ephemeral localhost port, useful for local testing.
For remote peers, bind the DHT to your exact Tailscale IP, e.g.:

```sh
./target/release/sangama ui --device metal --dht-listen /ip4/100.64.1.10/tcp/9000
```

Replace the example address with your device's actual address. Share the resulting node multiaddress
with the intended peer and allow the chosen DHT TCP port only between approved devices in your overlay policy.
The browser UI remains localhost-only. Existing inference traffic still uses the SSH tunnel workflow.
Do not share the browser control URL or `.mesh/` files.

## CLI: three nodes

No model download is needed to exercise discovery. Each process requires a distinct state directory.
A bootstrap node supplies initial contact addresses; it does not own a central registry.
Use more than one bootstrap node for deployment so discovery does not depend on a single contact.

```sh
# Terminal 1: bootstrap. Copy the address from the first JSON line.
./target/release/sangama dht-node --state-dir .mesh/bootstrap --listen /ip4/127.0.0.1/tcp/9000

# Terminal 2: provider. Substitute BOOTSTRAP_ADDRESS and the manifest SHA256.
./target/release/sangama dht-node --state-dir .mesh/provider --listen /ip4/127.0.0.1/tcp/9001 --bootstrap BOOTSTRAP_ADDRESS --model-hash MANIFEST_SHA256 --start 12 --end 24

# Terminal 3: seeker knows only the bootstrap, not the provider.
./target/release/sangama dht-node --state-dir .mesh/seeker --listen /ip4/127.0.0.1/tcp/0 --bootstrap BOOTSTRAP_ADDRESS --find-model MANIFEST_SHA256
```

A node address looks like `/ip4/127.0.0.1/tcp/9000/p2p/12D3KooW...`.
The manifest hash for the downloaded checkpoint is shown in the UI and real-model reports.
The CLI advertisement is a self-declared capability; only the UI checks that the selected local shard file
matches its manifest before advertising. Neither proves a remote worker is honest, online, or able to serve inference.

## Data and validation

- `.mesh/NODE/identity.key`: private persistent identity, mode 600 on Unix, never published.
- `.mesh/NODE/discovery.sqlite`: signed value records and provider pointers; WAL mode.
- `.mesh/NODE/node.lock`: prevents two processes opening the same identity/store concurrently.
- Routing connections are reconstructed from bootstrap/Identify; routing tables are not persisted.
- Value key: `/mesh/peer/PEER_ID`; payload includes model-manifest SHA256, layer range, discovery addresses,
  issue time, and expiry. Ed25519 signatures bind the record key and payload to that PeerId.
- Provider key: `/mesh/model/MANIFEST_SHA256`; standard Kademlia provider pointers are lookup hints.
  Results are displayed only after a valid, unexpired signed peer advertisement with the requested hash is retrieved.
- Five-minute advertisement leases renew while the advertiser runs. Expiry is enforced on load and read;
  expired database entries are pruned on startup/writes. Replaying an expired signed record is rejected.
- Limits: 8 KiB signed values, 512 values, 256 provider keys, 20 providers per key, 64 established connections,
  bounded pending connections/command queues and 32 KiB Kademlia packets. SQLite writes are synchronous and
  intended for this small test network; large deployments need a storage/throughput review.

Stopping a node does not instantly erase all replicas: leases expire. Signatures prove who published metadata,
not the truth of a capability claim. Discovery data is visible to participating DHT nodes.
Never put prompts, inference activations, KV caches, credentials, or private keys in the DHT.

## Security and remaining work

Bootstrap addresses include expected PeerIds, verified by the Noise handshake. Node listeners and user-supplied
advertisements/bootstrap addresses are constrained to numeric loopback/private/Tailscale IP/TCP addresses.
This is a controlled overlay network: protocol names and encryption do not grant membership authorization.
Network admission currently relies on the Tailscale access policy. Do not expose it as a public volunteer DHT.

The browser API checks Host, Origin, cross-site fetch metadata, and a per-launch capability; responses are
not cached and a strict Content Security Policy blocks external resources and framing. Peer credentials are
kept server-side. Keep the private browser URL private; anyone with it on the host can control that local UI.

Not implemented: application-issued invitations, membership approval/revocation, reputation/Sybil defense,
DHT-native NAT hole punching/relays, automatic shard downloads/assignment, automatic inference-route activation,
resource-sharing budgets, or mobile clients. Tailscale handles reachability for the initial remote test.
Existing localhost inference protections are preserved. See [the secure peer guide](secure-peer-test.md).

## Verification

`cargo test` includes three encrypted Kademlia swarms: a seeker bootstrapped through a third node discovers
another node's signed shard record. Tests also check SQLite reopen, stable identity across restart,
exclusive state locking, and rejection of modified, expired, oversized, and identity-substituted records.
These local tests do not establish worldwide reachability, performance, or hostile-network resilience.

Primary APIs: [libp2p Kademlia](https://docs.rs/libp2p/0.56.0/libp2p/kad/index.html),
[RecordStore](https://docs.rs/libp2p/0.56.0/libp2p/kad/store/trait.RecordStore.html).

## Naming compatibility

The project and executable are named Sangama. Existing `/mesh/...` protocol and record namespaces,
`.mesh/` state directories, and `P2P_TOKEN` environment variables are retained for compatibility.
Renaming the app does not replace existing node identities or invalidate discovery records.
