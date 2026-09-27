# Standalone generation

`generate` runs only the distributed decoding loop. It never opens the original complete checkpoint,
constructs a local Qwen model, or computes baseline logits. It shares manifest checking, tokenization,
worker identity/range validation, bounded binary transport, greedy selection, and session cleanup with `qwen-test`.

## File requirements

| Role | Required files in model directory |
|---|---|
| Client using existing peers | manifest.json, config.json, tokenizer.json |
| Worker | manifest.json, config.json, its assigned shard file |
| Client starting local workers | metadata/tokenizer plus every shard file |
| Verification client | metadata/tokenizer plus original model.safetensors; shards if starting workers |

Use `--model-dir` to select the appropriate directory. For `generate --peers`, `--device` is the expected
worker backend, not a request to load a model on the client. A CPU-only client build can request Metal workers.
Workers are loopback-only and token-authenticated. Remote traffic can use admitted encrypted mesh
bridges or the older private SSH tunnel setup. Managed mesh workers can load prepared shards on request;
see [mesh allocation](admitted-mesh.md#placement-reservations-and-opencode).
Discovering a DHT advertisement does not activate a route automatically.

## Execution

1. Validate manifest/config/tokenizer hashes and the prompt/context limits.
2. Start local shard processes or connect to the supplied authenticated workers.
3. Check every worker's manifest hash, layer range, shard hash, precision, and backend.
4. Reserve every worker, then send the prompt for prefill. No separate warmup prompt runs in generation mode.
5. The final worker selects the highest logit and returns a token. Send that token for the next step.
6. Stop at Qwen EOS or the requested token limit, then reset the session on workers.
7. Return text, token IDs, finish reason, timing, and worker information.

Ordinary generation returns only the selected token from the tail worker; `qwen-test` receives complete
vocabulary logits for verification. The standalone command does not stream text to the UI; the chat API
supports streaming. It does not automatically download weights or resume a failed session.
The standalone generation command uses: pinned Qwen2.5-0.5B-Instruct, F32, 512 prompt tokens,
128 generated tokens maximum, and one active session per worker.

## Honest reports

Generation returns null for `local`, `local_text`, `local_token_ids`, `tokens_match`, `passed`,
`maximum_logit_absolute_error`, and `logit_tolerance`. Successful generation does not imply baseline validation.
Verification retains these measurements and exits unsuccessfully on a mismatch.
First-token timing includes prompt execution but excludes loading/startup. Verification includes an extra
warmup before measuring; generation does not. Decode timing excludes the first token and counts EOS if emitted.

## Tests

`cargo test --test generation` uses fake protocol peers to isolate metadata-only client behavior,
EOS, token limits, reset/reuse, and absence of warmup. It is not a model-quality test.

The optional real-checkpoint regression is:

```sh
python3 scripts/test-generation.py --device metal
```

It creates a client directory containing only the three metadata/tokenizer files, generates twice through
real workers, verifies session reuse, tests autostart from shards without a complete checkpoint, then compares
its token IDs with a separate full-model verification run. It stops its temporary workers on exit.
The recorded Metal run generated 20 tokens with the same token IDs as the independent baseline.
All processes were on one physical Mac; a two-computer test remains the next milestone.

The OpenCode chat API separately supports conversation history with a 4,096-token total context budget and chunked prefill; see [OpenCode setup](opencode.md).
