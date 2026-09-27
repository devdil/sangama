# Docker isolation tests

Docker Desktop/Engine must be running. On Apple Silicon this builds native Linux ARM64 and uses CPU inference;
Metal is not available to Linux containers. These tests establish process, filesystem, and network separation,
not multi-computer GPU scaling or Internet latency.

## Build once

```sh
docker build -f docker/Dockerfile -t sangama:container-test .
```

The allowlist in `.dockerignore` excludes model weights, credentials, node identities, Git history, and local
build artifacts from the build context. Model files are mounted only at test runtime. The builder uses a
cached Rust release build; the runtime includes the binary, Python test helpers, and OpenSSH.
The first build needs Internet access for official base images, Debian packages, and Cargo dependencies.

## Real Qwen inference

Prepare the default two-shard checkpoint with `python3 scripts/fetch-qwen.py`, then:

```sh
python3 scripts/test-containers.py
```

Alternatively add `--build` to build the image first. Allow roughly 8 GB of Docker VM memory for these tests.
Worker containers are capped at 3 GiB each and the client at 768 MiB, with two CPU cores per container.

| Container | Model files mounted read-only | Responsibility |
|---|---|---|
| worker0 | Config, manifest, shard 0 | Layers 0–11, forward through SSH to worker1 |
| worker1 | Config, manifest, shard 1 | Layers 12–23, return logits |
| client | Config, manifest, tokenizer | Tokenize, generate, reset, save report; no weights |

Each has its own Docker network namespace on an internal bridge. Workers still bind inference only to
127.0.0.1. The client opens pinned-key SSH tunnels to both workers; worker0 opens its own tunnel to worker1.
The SSH daemons listen on TCP 2222 inside the private Docker network. No host ports are published.
Tailscale is not needed for this one-host internal network test.

Every service runs as UID 10001 with a read-only root filesystem, all Linux capabilities dropped,
`no-new-privileges`, and a bounded temporary filesystem/process count. Dedicated host keys, per-caller SSH
identities, and a test token are created in disposable volumes. Worker SSH authorization limits forwarding
to its inference port and denies shells, reverse forwarding, agent forwarding, and Unix-socket forwarding.
The initializer alone runs as root with networking disabled to set secret volume ownership; it exits before services start.

The test generates a short explanation, then answers an arithmetic prompt to check session reuse. It checks:

- Client has no model/shard weight files.
- Worker mounts contain exactly their assigned shard.
- Three distinct container network namespaces and no published host ports.
- Incorrect application tokens are rejected through SSH.
- Generation reports no fabricated baseline result.
- Generated token IDs match the recorded independent baseline, when that report is available.

This token comparison is not a fresh full-logit numerical verification. Output and inspection evidence are
written to `runs/docker-test/container-generation.json`; service logs are saved alongside it.
The script removes its containers, private network, and secret/report volumes in `finally`, including on ordinary failures.
It retains the image/build cache and copied host-side reports. Interrupted host processes or a failed Docker daemon
can require manual cleanup of resources labelled `sangama.test=true`; avoid deleting unrelated resources.

## DHT across separate containers

```sh
python3 scripts/test-dht-containers.py
```

This starts a seed, provider, and seeker in distinct network namespaces with no model mounts. The seeker knows
only the seed's address and must discover the provider's signed advertisement. Kademlia uses Noise-encrypted
TCP on private bridge addresses. SQLite/identity state is temporary for this test. The all-`a` hash is synthetic
metadata, not a real model checkpoint. No inference membership is granted by discovery.
Results go to `runs/docker-dht-test/dht-containers.json`; containers/network are then removed.

Both scripts use unique resource names and never push images or Git commits to a remote registry/repository.

## Recorded run — 2026-09-27

Passed on an M5 Pro host, Docker Desktop Linux ARM64, approximately 8 GB VM memory.
Three inference containers produced 20 tokens (including EOS), matching every token in the recorded
independent Metal baseline. The follow-up arithmetic prompt returned `4`. Incorrect bearer authentication
returned HTTP 401. Container inspection confirmed separate network namespaces, exact read-only model
mounts, non-root execution, dropped capabilities, and no published ports.

| Measurement | Observed |
|---|---:|
| First token, after workers loaded | 2,155 ms |
| Decode throughput | 3.06 tokens/s |
| Generation total | 8,357 ms |

These are one-run CPU measurements on a shared VM, excluding image build and worker loading; they do not
predict remote peers, Metal performance, or production capacity. The three-container DHT test separately
found the provider through the seed using signed discovery metadata. All test containers, networks, and
credential volumes were removed afterward; the local image and build cache remain for reuse.

Raw evidence: [generation](test-results/docker-generation.json) and [DHT](test-results/docker-dht.json).
