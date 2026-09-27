# Network UI workflow

The hosted portal and local inference UI have different permissions. The portal manages invitations and membership; the local UI measures workers and controls inference. Keep local APIs private.

## Hosted classic HTML portal

- **Sign up / Sign in:** invite-only accounts with username and password. Signup asks for no device details. Accounts do not connect or authorize a worker. See [portal accounts](portal-accounts.md).
- **Connect a worker:** gives the local identity, invitation-redemption and managed-node commands. The operator supplies the authority public key and configuration. Private peer keys never enter the website.
- **Operator:** view admitted/expired/revoked memberships, issue a role-scoped network invitation or a signup invitation, and revoke a peer with explicit confirmation.

Run `python3 deploy/portal/prepare-secrets.py` before rebuilding an existing Compose deployment. It now also creates a dedicated `secrets/admin_token` without replacing existing secrets. Compose mounts this only in the portal and sets `ADMIN_TOKEN_FILE`. Custom deployments can omit the variable to disable browser administration; the existing operator CLI still works. The configured credential must contain 32 random bytes encoded as 64 hex characters.

Use that separate credential in the Operator form for each action. It is not the database password, a worker token, an invitation or an identity key. Operator authentication does not use the account session cookie and the credential is not reflected in results. Forms enforce the configured exact Origin; responses prohibit caching and framing. Admin POSTs have a process-wide limit of 20 per minute. This basic shared-operator access is for trusted deployments; it is not multi-user administration with individual audit identities or MFA. The global limit can temporarily deny an operator during abuse.

An invitation is shown in its creation response only and expires after 24 hours or use. Do not refresh/resubmit the creation response if you do not want another invitation. Send invitations privately. A new valid invitation can readmit a previously revoked identity; revocation is not a permanent identity ban.

The directory does not claim a device is online. Authenticated membership listings show role and expiry, not inferred readiness. Public membership snapshots already expose admitted peer IDs, roles and expiry.

## Local worker controls

Start the admitted client mesh and managed workers using [the mesh guide](admitted-mesh.md). Then start the local UI with the same candidate bridge addresses and that client's local token:

```sh
./scripts/cargo build --release --locked --features metal
./target/release/sangama ui --device metal \
  --model-dir /path/to/client-metadata \
  --peers 127.0.0.1:7902,127.0.0.1:7901 \
  --token-file /path/to/client-local-token
```

For CPU workers, omit the Metal build feature and use `--device cpu`. Open the private capability URL printed locally. Do not publish or share it. The browser never receives the worker token. The server probes only its configured loopback candidates; it accepts no browser-supplied remote destinations.

In **Invited worker network**:

1. **Refresh workers** reports measured available memory/contribution budget, prepared or loaded shard, busy/unavailable state and probe duration. The duration combines capacity and info checks, not a pure network RTT. Automatic refresh is every ten seconds after other UI polls finish.
2. Current mesh bridges also report membership validity/expiry and observed direct or relay connections. Multiple paths can coexist; this is connection telemetry, not proof of the exact path of every inference packet. Older bridges/private tunnels report unknown membership/path. These are point-in-time observations.
3. **Allocate & load shards** runs managed allocation with reservations and readiness checks. Only prepared files on managed workers are eligible. The UI reports the operation as running until it succeeds or reports an error; it does not estimate per-file progress.
4. **Check ready route** validates complete model coverage among already loaded candidates. Allocation is not necessary for manually loaded workers.
5. Select **Connected peers** and generate. The server probes and orders the ready route again before generation; the runner reserves it before computing. A status snapshot is never treated as a permanent reservation.

Placement and inference share a job lock, preventing simultaneous UI operations. Other clients may still contend for the same workers; worker-side reservations remain authoritative. If generation is interrupted, recover/reallocate the workers, check the route and submit a new request. The UI does not resume old KV state.

The legacy private-overlay discovery panel is explicitly labeled. Its bootstrap input is not a membership invitation and its signed offers do not authorize execution in the admitted network.

## Validation

`python3 scripts/test-portal-admin.py` exercises the local Compose preview's authenticated operator forms and removes its temporary membership/invitation rows. Do not aim it at a production database. `scripts/test-mesh-containers.py` additionally exercises protected UI capacity/relay telemetry, allocation, route ordering and real Qwen generation on isolated Docker networks. It rejects unauthorized and cross-origin placement requests and compares generated tokens to the reference baseline.
