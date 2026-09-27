# Sangama hosted portal

This is a separate Rust server-rendered portal, not the localhost inference control panel. PostgreSQL stores
single-use invitations and unverified device registrations. The DHT continues to store signed records in each
node's SQLite database. The hosted directory does not advertise peers into the DHT, verify ownership,
authorize inference, or assign model layers. No model weights or inference worker ports are deployed.

## Current verification

Local Docker tests pass: PostgreSQL persistence, invitation consumption, duplicate handling, form-origin checks,
publication consent, HTML escaping, and database-role/container isolation. The directory and registration pages
have been inspected in the browser. Cloud provisioning and public TLS still need a Linode credential and hostname.

## Local preview

From the repository root:

```sh
python3 deploy/portal/prepare-secrets.py
docker build -f deploy/portal/Dockerfile -t sangama-portal:local .
export PUBLIC_ORIGIN=http://127.0.0.1:18080 PUBLIC_HOST=127.0.0.1
docker compose -f deploy/portal/compose.yaml -f deploy/portal/compose.test.yaml -p sangama-portal-test up -d postgres portal
python3 scripts/test-portal.py
```

Visit http://127.0.0.1:18080. The integration test expects a fresh database and leaves two clearly synthetic test
registrations. It is not a test against production. The local override publishes only the portal on loopback;
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
its SHA-256 hash, and consumption/registration happens atomically. The invitation grants only a directory entry,
not DHT or inference membership. Do not publish invitations in Git, logs, screenshots, or public URLs.

## Operations

The portal fails closed on DB loss and Docker restarts it. Public forms have same-origin enforcement, an 8 KiB
body limit, bounded concurrent handlers, parameterized SQL, escaped output, no JavaScript, and restrictive CSP.
Peer identifiers are private in the portal; device names/specifications are public only with explicit consent.
There are no user accounts, automated emails, device ownership proofs, or public inference APIs in this release.
