# Sangama hosted portal

This is a separate Rust server-rendered portal, not the localhost inference control panel. PostgreSQL stores
single-use invitations, user accounts, hashed login sessions, legacy directory registrations, and cryptographic network memberships. The DHT continues
to store signed records in each node's SQLite database. Directory registration does not authorize inference;
the separate membership API verifies peer-key ownership and signs short-lived authorization snapshots.
See [the admitted mesh guide](../../docs/admitted-mesh.md) for enrollment, relay setup and revocation. No model weights or inference worker ports are deployed. The current Linode firewall configuration does
not yet expose a relay port; deploy an admitted relay deliberately before opening its TCP port.

## Current verification

Local Docker tests pass: PostgreSQL persistence, invitation consumption, duplicate handling, form-origin checks,
password hashing, session rotation/expiry, concurrent invitation redemption and operator controls. The directory and registration pages
have been inspected in the browser. Cloud provisioning and public TLS still need a Linode credential and hostname.

## Local preview

From the repository root:

```sh
python3 deploy/portal/prepare-secrets.py
docker build -f deploy/portal/Dockerfile -t sangama-portal:local .
docker run --rm --network none --user 0:0 \
  -e MEMBERSHIP_KEY_FILE=/keys/membership_key \
  --mount type=bind,src="$(pwd)/deploy/portal/secrets",dst=/keys \
  sangama-portal:local authority-init
export PUBLIC_ORIGIN=http://127.0.0.1:18080 PUBLIC_HOST=127.0.0.1
docker compose -f deploy/portal/compose.yaml -f deploy/portal/compose.test.yaml -p sangama-portal-test up -d postgres portal
python3 scripts/test-portal.py
```

Visit http://127.0.0.1:18080. The integration test creates temporary accounts and invitations, then removes them. It is not a test against production. The local override publishes only the portal on loopback;
PostgreSQL has no published port. Remove the test containers and their disposable data with the same Compose
files/project and `down -v` when finished. Never use the test override on the public server.

## Provision a new Linode

`deploy/linode/main.tf` describes a separate Ubuntu 24.04 Linode in Chennai: 2 GB, one shared vCPU, backups,
and a default-deny firewall. SSH is limited to the operator's explicit IPv4 /32; only TCP 80/443 is public.
Linode's public API quoted $12/month plus $2.50/month for backups on 2026-09-27, before tax. Check the plan
and current regional prices before applying. This size is for the portal and database, not model inference.

```sh
python3 deploy/linode/prepare-keys.py
terraform -chdir=deploy/linode init
# Set TF_VAR_linode_token securely in this process; do not paste it into tracked files or command arguments.
# Set TF_VAR_ssh_cidr to the operator's current public IPv4 with /32.
umask 077
terraform -chdir=deploy/linode plan -out=portal.tfplan
terraform -chdir=deploy/linode apply portal.tfplan
terraform -chdir=deploy/linode output -raw server_ip
```

Terraform state/plan files contain secrets and are ignored by Git. Keep the dedicated keys and root password
in `.secrets/linode` private. The cloud-init payload installs the pre-generated server host key so deployment
can pin it without disabling SSH verification. Root/password SSH is disabled; `deploy` uses its dedicated key.
`prevent_destroy` protects the instance from accidental replacement. These resources are separate from
MailMyCard and its Terraform state. No MailMyCard instances or secrets are changed.

## Deploy

Point a chosen DNS A record at the new IP, then:

```sh
python3 deploy/portal/deploy.py --host SERVER_IPV4 --domain YOUR_HOSTNAME
```

Only an explicit source-file allowlist is shipped over authenticated SSH. The server builds its native Docker
image. Database secrets are created on the server in a 0700 parent directory and mounted read-only only into
services that need them. The portal's DB role is not a superuser. Portal and PostgreSQL have no host ports;
Caddy terminates public HTTPS. The script verifies HTTPS without bypassing certificate validation.
Persistent database and Caddy volumes survive app releases. Deployment does not push Git or registry images.
The single-server stack is not highly available. Provider backups are enabled; a tested database restore and
an off-server logical backup policy are still needed before storing important production data.

## Create an invitation

On the dedicated server, from `/srv/sangama/current/deploy/portal`, an operator can run:

```sh
sudo docker compose --env-file .env exec -T portal /usr/local/bin/sangama-portal invite
```

This prints a secret invitation that expires in 24 hours. Give it only to the intended peer. The database stores
its SHA-256 hash, and consumption/registration happens atomically. The invitation grants only a portal account,
not DHT or inference membership. Do not publish invitations in Git, logs, screenshots, or public URLs.

## Operations

The portal fails closed on DB loss and Docker restarts it. Public forms have same-origin enforcement, an 8 KiB
body limit, bounded concurrent handlers, parameterized SQL, escaped output, no JavaScript, and restrictive CSP.
Peer identifiers are private in the portal; device names/specifications are public only with explicit consent.
Signup requires an invite code, username and password. Account login never grants worker or admin permissions. There are no automated emails, password-recovery flows, device ownership proofs, or public inference APIs. See [portal accounts](../../docs/portal-accounts.md).

### Browser operator controls

The classic HTML `/admin` form uses a separate `ADMIN_TOKEN_FILE` credential.
Run `prepare-secrets.py` on upgrades to generate the new named `admin_token`
secret before `docker compose up`. The form can issue scoped network/signup
invitations, inspect membership expiry and revoke a peer. No operator browser
session is stored. See [the UI workflow](../../docs/network-ui.md) for credential
handling, limitations and local tests. Worker onboarding remains a local signed
challenge; do not upload private peer keys to the portal.

### Credits

The portal records signed credit receipts from workers and clients. Set
`CREDIT_ALLOWANCE` (whole credits) in the Compose environment to refuse new sessions
to members whose balance falls below minus that amount; leave it unset to record
without enforcing. Link a person's peers to their account with
`sangama-portal link-peer <peer-id> <username>` or the `/admin` form. See
[credits](../../docs/credits.md).

### Member invitations

Signed-in members can issue worker and client network invitations from `/account`,
up to `MEMBER_INVITES` per 30 days (default 3; `0` disables). Stop a misbehaving
inviter and revoke its peers with `sangama-portal stop-inviter <username>`. See
[member invitations](../../docs/admitted-mesh.md#member-invitations).

### Local provider credential

Store the Linode API token as `LINODE_TOKEN` in the repository-root `.env` (mode 0600).
The file and environment-specific variants are ignored by Git and excluded from Docker/SSH
bundles. Load the value into `TF_VAR_linode_token` only in the Terraform child process;
never pass it as a command-line argument or print it. The portal server does not need this token.
