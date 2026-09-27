# Recorded validation

Local runs on Apple M5 Pro (24 GB). All nodes/workers ran on one physical computer.

- `dht-process-test.json`: three independent processes; seeker discovers a signed provider through bootstrap only. The all-`a` hash is a synthetic discovery key, not a checkpoint.
- `ui-dht-test.json`: browser joined another node, advertised a locally verified Qwen shard, found the remote process's signed advertisement, and ran actual Qwen inference with matching tokens/logits.
- `qwen-metal-final.json`: direct two-worker real-model verification before DHT integration.
- `qwen-ssh-local.json`: actual SSH-encrypted loopback inference, exact logit/token agreement.
- `secure-peer-checks.json`: rejected wrong host key/token, unapproved egress, malformed and oversized frames.

21 Rust tests and two Python tests passed locally. Formatting and Metal clippy checks passed.
These are engineering checks, not production certification or worldwide performance benchmarks.
Private identities, tokens, control-panel URLs, and model weights are excluded from the repository.
