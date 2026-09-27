# OpenCode + Sangama

The pinned OpenCode client talks to a local Rust chat gateway, which runs generation through the existing
Qwen shard workers. This is a working text-only coding assistant integration, not an autonomous coding agent.
It can discuss pasted code and suggest corrections. Tools, shell commands, repository browsing, and automatic
file edits are disabled; the gateway rejects tool requests. The current 0.5B model is not reliable enough for
normal coding work. A correct response to an explicit addition fix and an incorrect response to a less explicit
version of the same task are both retained in the test evidence.

## Run on this Mac

From the Sangama repository:

```sh
./scripts/cargo build --release --features metal --locked
./scripts/install-opencode.sh
python3 scripts/opencode.py
```

OpenCode 1.18.32 is already installed locally for the recorded test. The installer pins npm dependencies with
a lockfile, skips dependency lifecycle scripts, and then runs the reviewed upstream binary installer.
It installs under `.tools/opencode`, without changing global packages or shell configuration.
The Rust build needs the prepared Qwen checkpoint to run; use `python3 scripts/fetch-qwen.py` if it is missing.

The launcher starts two independent Metal worker processes and the gateway, opens OpenCode in a disposable
`work/opencode-workspace` repository, and stops its own child processes when OpenCode exits. Choose the
**sangama** agent and **Sangama peers / Qwen 0.5B** model. Paste a short code example into the prompt.
Only Sangama is enabled as an inference provider; there is no cloud-model fallback. Sharing and auto-updates
are disabled. First-time OpenCode startup may still download client/provider metadata or packages; this is
not an offline-network guarantee.

For a noninteractive example:

```sh
python3 scripts/opencode.py -- run 'Explain what this does: def double(x): return x * 2'
```

Use `--device cpu` for a CPU-only build. The launcher keeps OpenCode's XDG config/data/cache/state in
`.mesh/opencode` and passes an API token through its child environment, never a command-line argument.
This local state includes conversation history; it is ignored by Git. The launcher doesn't overwrite your
normal OpenCode settings. Credential files are private and retained under `.secrets/opencode` for reuse.

## Existing or remote workers

Start the [admitted mesh bridges](admitted-mesh.md) first, then use their local endpoints in layer order.
Authenticated SSH tunnels remain an alternative:

```sh
python3 scripts/opencode.py --device metal \
  --peers 127.0.0.1:7901,127.0.0.1:7902 \
  --worker-token-file .secrets/peer-test.token
```

This starts only the gateway and OpenCode; it does not stop workers supplied by you. Worker devices and
model manifests must match. See [secure peer setup](secure-peer-test.md). DHT discovery does not automatically
select or authorize this inference route.

## Protocol and boundaries

- `GET /v1/models` and `POST /v1/chat/completions`, bound only to `127.0.0.1:8090` by the launcher.
- Distinct API and worker bearer tokens. OpenCode receives only the API token.
- Exact Host validation, browser Origin rejection, 256 KiB request-body limit, constant-time credential comparison.
- One active request per gateway; concurrent requests receive HTTP 429. Do not run multiple gateways against
  the same workers expecting a shared scheduler; workers still support one active session.
- Text messages with system/user/assistant roles; no images, tool messages, tool calls, structured output,
  custom stop sequences, or nonzero temperature. Deterministic greedy generation only.
- 4,096-token total context including output, up to 64 messages, 128 output tokens maximum. No silent truncation.
  CLI `generate`/`qwen-test` retain their 512-input-token limit. Chat prefill uses chunks of at most 512 tokens.
- SSE emits decoded text while inference runs, followed by finish reason, usage, and `[DONE]`. Failures after
  streaming headers are sent become an error event; earlier validation failures return HTTP errors.
- Sessions reset after generation/failure. A disconnected streaming client stops generation at a subsequent
  chunk/token and then resets the workers. A disconnected non-streaming request finishes bounded generation
  and resets. History is resubmitted by OpenCode on the next turn; this is not persistent conversational KV reuse.
- The gateway never loads model weights when using explicit peers. Each worker loads only its shard.

OpenCode permission settings are application controls, not an OS sandbox. This integration has no enabled
execution tools. Before adding tools, use a separate sandbox and a model with tested tool-calling support.
Neither this integration nor encrypted transport makes untrusted inference peers confidential or trustworthy.

## Verification

```sh
python3 scripts/test-opencode.py
# In another session, start services and keep them running:
python3 scripts/opencode.py --serve-only
# Then:
python3 scripts/test-chat-api.py
# Stop serve-only with Ctrl-C.
```

Recorded tests on 2026-09-27 used the actual pinned model and two Metal worker processes on the M5 Pro.
API tests cover authentication, Host/Origin restrictions, unsupported requests, JSON/SSE parity, a 694-token
prompt spanning multiple prefill chunks, and conversation recall. The OpenCode test parses the returned
Python suggestion and checks `return a + b` without executing model-generated code. This demonstrates the
integration and a narrow suggestion task, not an autonomous read/edit/test loop or general coding quality.

Evidence: [API test](test-results/opencode-api.json), [OpenCode task](test-results/opencode-integration.json),
and [earlier unsuccessful suggestion](test-results/opencode-model-limitation.json).

Upstream references: [custom providers](https://opencode.ai/docs/providers),
[agent configuration](https://opencode.ai/docs/agents), [configuration](https://opencode.ai/docs/config).
