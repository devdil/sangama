# Security policy

Sangama admits peers with signed invitations, encrypts mesh traffic and runs a hosted membership portal. Flaws in those paths can expose members' machines or let unadmitted peers join, so please report them privately.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/devdil/sangama/security/advisories/new). Do not open a public issue, discussion or pull request for a suspected vulnerability.

Include what you can of:

- the affected component (worker, relay, portal, network-auth, chat API, installer) and commit;
- steps or a proof of concept that reproduces the problem;
- the impact you expect, such as admission bypass, key or token disclosure, remote code execution, or denial of service.

You should get an acknowledgement within 7 days. Fixes are developed in a private advisory and credited to the reporter unless you prefer otherwise.

## Supported versions

Sangama is pre-1.0. Only the latest commit on `main` receives security fixes, and mixed old/new workers are not supported.

## In scope

- Membership, invitation and revocation checks (`crates/network-auth`, `portal/`)
- Mesh transport, relay and signed discovery (`src/mesh*.rs`)
- Frame parsing and bounds checks (`src/qwen/wire.rs`)
- Local bearer tokens, peer keys and the authority signing key leaking into logs, reports or the network
- Packaging and install scripts

Model output quality, and performance claims from the numerical fixture, are not security issues. Please open a regular issue for those.
