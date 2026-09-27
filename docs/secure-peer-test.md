# Secure two-person Qwen test, including Internet peers

This is a controlled test between trusted people, not a production/public volunteer service.
The source is in `/Users/diljit/Documents/Projects/p2p-inference`. Do not publish credentials.
Only the real Qwen commands described here are hardened for this workflow; the older numerical
fixture coordinator/worker commands are separate development tools and must not be exposed.

## Connection design

```text
Your Mac: client + shard 0 (127.0.0.1:7901)
    -> local SSH tunnel (127.0.0.1:7902)
    -> encrypted, authenticated SSH over private Tailscale
    -> peer: shard 1 (127.0.0.1:7902)
```

The tunnel carries both directions of each request. No reverse tunnel is needed for two shards.
Workers refuse non-loopback listen and destination addresses, including LAN/Tailscale addresses.
Worker 0 additionally requires `--allow-next 127.0.0.1:7902`; authenticated requests cannot choose
arbitrary downstream ports. HTTP exists only on each machine's loopback interface. SSH authenticates
both ends; the bearer token also gates local access. Local administrators or a compromised endpoint
can still inspect model inputs/activations. Use non-sensitive test prompts and a trusted peer.

## 1. Enroll the two devices privately

Install [Tailscale](https://tailscale.com/download) on both computers and sign in with MFA-protected accounts.
The peer can [share only their device](https://tailscale.com/kb/1084/sharing) with you rather than inviting you
into their whole network. Limit the access policy to your identity/device reaching the peer's TCP port 22.
Check for broader existing allow rules: an added narrow grant does not cancel a broad grant.
Do not enable Funnel, router port forwarding, subnet routes, or exit-node access for this test.
Use the peer's `100.x.x.x` Tailscale IPv4 address with the tunnel helper.

Tailscale may connect directly or use a relay; both work, but latency varies.
Record `tailscale ping PEER_IP` when measuring. No global Tailscale or firewall settings are changed by our scripts.

## 2. Peer prepares an SSH account

Use ordinary OpenSSH over the Tailscale connection (this guide does not configure the separate Tailscale SSH feature).
The peer enables SSH only for a dedicated non-admin test account, with key authentication.
The host firewall should limit inbound SSH to the private overlay where supported; do not forward router port 22.
macOS Remote Login must be limited to the intended account and must not receive Full Disk Access for this test.
The peer starts their worker themselves; you do not need a remote shell or their login password.

Generate a dedicated client key on your Mac:

```sh
mkdir -p .secrets
chmod 700 .secrets
ssh-keygen -t ed25519 -a 100 -f .secrets/peer_ed25519
ssh-add .secrets/peer_ed25519
```

Choose a passphrase. Send **only** `.secrets/peer_ed25519.pub` to the peer.
On the peer, put that public key on one line in the dedicated account's `authorized_keys`, prefixed with:

```text
restrict,port-forwarding,permitopen="127.0.0.1:7902",command="/usr/bin/false" ssh-ed25519 PUBLIC_KEY_HERE
```

The peer's administrator should restrict that account in sshd configuration (validate with `sshd -t`
before applying, keeping existing admin access open):

```text
Match User p2ptest
    AuthenticationMethods publickey
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AllowTcpForwarding local
    AllowStreamLocalForwarding no
    PermitTunnel no
    PermitOpen 127.0.0.1:7902
    AllowAgentForwarding no
    X11Forwarding no
    PermitTTY no
    MaxSessions 0
```

Substitute the actual dedicated account name. `MaxSessions 0` allows forwarding while preventing shell/subsystem sessions.
`AllowTcpForwarding local` prevents reverse forwards; the authorized-key `port-forwarding` option alone does not.
The test scripts do not modify sshd or enable system services automatically.

Pin the SSH host key before connecting. Have the peer obtain its public host key and SHA256 fingerprint locally
(e.g. `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`) and confirm the fingerprint through a trusted separate channel.
Create `.secrets/known_hosts` containing the verified public host key:

```text
100.PEER.IP.ADDRESS ssh-ed25519 VERIFIED_HOST_PUBLIC_KEY
```

The placeholders must be replaced. For a nonstandard SSH port use `[100.PEER.IP.ADDRESS]:PORT`.
Do not trust an unverified `ssh-keyscan` result or turn off host-key checking. A changed key must be investigated.

## 3. Build and prepare model files

Both machines build from the same source. On Apple Silicon:

```sh
./scripts/cargo build --release --locked --features metal
python3 scripts/fetch-qwen.py
```

The peer only needs `config.json`, `manifest.json`, `LICENSE`, and `shard-1-of-2.safetensors` to run shard 1.
They can run the downloader to derive matching files independently, then remove the unneeded original/shard-0 files.
Your validation client needs the complete model and tokenizer for the baseline. Downloaded files are hash checked.
On a CPU-only machine omit `--features metal`; for this initial test select `--device cpu` on both workers and client.
Mixed CPU/Metal validation is currently rejected. Native Windows has not been verified; macOS/Linux are the documented targets.

Create the shared application token once on your Mac:

```sh
python3 scripts/new-peer-token.py
```

Transfer `.secrets/peer.token` via an existing authenticated encrypted file transfer or password-manager share.
Do not send it in chat, commit it, put it in CLI arguments, or use the restricted tunnel key for file transfer.
The peer stores it in their project's `.secrets/peer.token`, with directory mode 700 and file mode 600.
`--token-file` rejects symlinks and files accessible to other users on Unix. Unset old `P2P_TOKEN`/`P2P_TOKEN_FILE`
environment values if they conflict. Each separate test pair should use a fresh token.

## 4. Start the two workers and tunnel

Peer terminal (keep running):

```sh
./target/release/sangama qwen-worker --device metal --shard 1 --listen 127.0.0.1:7902 --token-file .secrets/peer.token
```

Your Mac, terminal 1:

```sh
python3 scripts/peer-tunnel.py --host PEER_TAILSCALE_IP --user p2ptest
```

The helper pins the verified host key, requires key authentication, disables agent/X11 forwarding,
binds the tunnel only to localhost, and fails if it cannot bind the forwarding port.
A tunnel starting successfully does not prove the remote worker is ready; the model test checks that separately.
Keep the terminal open. No secret values appear in the command.

Your Mac, terminal 2:

```sh
./target/release/sangama qwen-worker --device metal --shard 0 --listen 127.0.0.1:7901 --allow-next 127.0.0.1:7902 --token-file .secrets/peer.token
```

Your Mac, terminal 3:

```sh
./target/release/sangama qwen-test --device metal --peers 127.0.0.1:7901,127.0.0.1:7902 --token-file .secrets/peer.token --prompt 'Explain peer-to-peer computing in one short sentence.' --max-tokens 40 --output runs/secure-two-device.json
```

Expected: `passed: true`, identical token IDs, maximum logit error <= 0.001, two distinct worker identities.
Check the peer terminal confirms their worker received the load, and record both computers and network route
alongside the report; process IDs alone do not prove separate physical hosts.
A small numerical difference across hardware is possible; do not increase tolerance just to hide a failure.

## 5. Stop and revoke

Ctrl-C both workers and the SSH tunnel. Remove the test key from the peer's authorized_keys,
revoke the Tailscale device share/grant when no longer needed, and delete the test token from both machines.
If a token is exposed, stop the workers, replace it on both ends, and restart. Existing processes do not hot-reload tokens.
Reports contain prompts and generated text: keep or delete them deliberately. Restart/replay after a disconnect;
automatic KV-cache migration or failover is not implemented.

## Performance expectations

The small-model local benchmark does not predict intercontinental speed. Each token must cross the route;
RTT, relay use, bandwidth, CPU/GPU time, and returning complete diagnostic logits all add latency.
For example, 150 ms of network round-trip time alone limits a sequential route to under 6.7 tokens/sec,
before compute and transfer costs. This is an illustrative bound, not a measurement.

## Primary references

- [OpenSSH client options](https://man.openbsd.org/ssh_config): host verification and forwarding options.
- [OpenSSH key restrictions](https://man.openbsd.org/sshd): `restrict`, `permitopen`, forced commands.
- [OpenSSH server options](https://man.openbsd.org/sshd_config): account limits and forwarding direction.
- [Tailscale sharing](https://tailscale.com/kb/1084/sharing): private device enrollment and access control.
- [Tailscale connection types](https://tailscale.com/docs/reference/connection-types): direct and relayed connections.
