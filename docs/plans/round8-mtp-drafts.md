# Round 8 plan: drafting with Qwen3.5's own MTP head

## Question

How much faster does one stream decode when the last stage drafts tokens with the model's multi-token-prediction (MTP) head, and the next pass verifies them?

Every decode step costs one trip through all 20 stages, about 190 ms, almost all of it network. Verifying several drafted tokens in that trip costs about the same as computing one. Round 6's prompt-lookup drafts were right only 5–26 % of the time, and restoring all 20 stages after a rejection cost more than they saved. The model's own MTP head should draft far better.

## How it works

- **Draft on the last stage.** It already has the final hidden state and the LM head. After each batch it feeds the kept positions to the MTP head, pairing each token with the hidden state before it, then drafts up to K tokens after the token it chose. Drafts go back to the client with the result, so drafting adds no network trip.
- **Verify in the next step.** The client sends `[token, drafts…]`. The existing speculative path verifies them in one pass and rolls every stage back to the accepted prefix.
- **Protocol additions**, all ignored unless requested:
  - `inputs`: every frame's input tokens, carried to the last stage so the head sees every position;
  - `mtp_drafts`: how many drafts the client wants;
  - `drafts`: the last stage's reply.
- **Settings:**
  - client: `SANGAMA_MTP=K`;
  - last worker: `--mtp-gguf FILE`;
  - optional confidence floor on the last worker: `SANGAMA_MTP_P_MIN`.
- **Code:**
  - `crates/llama-stage/src/shim_mtp.cpp`: the MTP context and draft loop, using llama.cpp's MTP context type;
  - `Stage::attach_mtp` / `mtp_step`;
  - drafting in `compute()` in `src/qwen/network.rs`.

## Local results (30 Sep, M5 Pro, Metal, Qwen3.5-0.8B BF16 split into two stages)

Output is token-for-token identical to plain greedy decoding in every run.

**Stage-level test** (`mtp_drafts_keep_greedy_output`, 4 drafts per step, 48 tokens):

| Prompt | Drafts accepted | Tokens per pass |
|---|---|---|
| Code | 32 / 60 (53 %) | 3.1 |
| Explanation | 30 / 76 (39 %) | 2.6 |
| Factual | 29 / 72 (40 %) | 2.6 |

**Two worker processes**, `SANGAMA_MTP=4`, 96 tokens:

| Prompt | Pipeline passes | Drafts accepted |
|---|---|---|
| Code | 96 → **28** | 68 / 106 |
| Explanation | 96 → **43** | 53 / 161 |

On one machine tokens per second did not improve, because there is no network time to save and snapshots cost compute. On the fleet a pass costs ~190 ms of network, so the gain should track the pass reduction.

## The 397B MTP head

The published unsloth GGUF omits the MTP layer. It was converted from the original release (`Qwen/Qwen3.5-397B-A17B`, safetensors files 91–94) with `convert_hf_to_gguf.py --mtp`:

| File | Size | SHA-256 |
|---|---|---|
| `mtp-q4_k_m.gguf` | 5.7 GB | `5c01abdc7fb733c0f33f16f62dee287c52ddd2c7397f5b12ec024455b5d2c11b` |
| `mtp-q8_0.gguf` | 9.2 GB | `cb574a8f18eb6dcfeab23d36a03e1ec12b9275fc2de6338338db3a9391624ac5` |

Both files include their own copy of the embedding and LM head. They sit on the model-splitting box in `/root/mtp/`.

**Memory:** the last stage uses ~13.8 GB today. With the head it needs ~19.5 GB (Q4_K_M) or ~23 GB (Q8_0), which is over the 16 GB budget. The round-6 last stage is an RTX 6000 Ada (48 GB), so for this round only the last stage runs with a larger `--memory-budget-mib`. Alternatives, if it must stay at 16 GB:
- give the last stage 2 layers instead of 3;
- run the head on its own machine next to the relay, at the cost of one extra 0–4 ms hop.

## Fleet run

1. Replace w17–w19 (down since 30 Sep) and deploy the round-7 build, which includes this code, to all stages.
2. Copy the MTP head to the last stage and restart it with `--mtp-gguf`.
3. Standard prompt set, 128 tokens each. Measure `SANGAMA_MTP` = 0, 2, 3, 4 and 6, with the Q4_K_M head and then the Q8_0 head. colibri reports that heavily quantised heads accept almost nothing, so this comparison matters.
4. Record for each run:
   - tokens per pass;
   - drafts accepted;
   - decode tok/s;
   - time of the verify pass against a plain pass;
   - snapshot and restore time per stage.
5. Every run must match the plain run's tokens exactly.

**Expected:** 2–3 tokens per pass. If a verify pass costs ~1.1× a plain pass, that gives roughly 10–14 tok/s, up from 5.4. This is an estimate, not yet measured.

**If rollbacks dominate:** make them cheaper. For example, save only the recurrent state (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`) and trim attention KV instead, or keep per-position recurrent states during verify so a rejection needs no replay.
