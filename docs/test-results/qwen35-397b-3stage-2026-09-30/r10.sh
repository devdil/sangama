#!/bin/sh
# r10.sh TAG DRAFTS...: draft sweep on the relay box; compares each run's tokens with the 0-draft run.
TAG=$1; shift
P1="Write a Python function that returns the n-th Fibonacci number iteratively, then explain it."
P2="Write a short explanation of how pipeline parallelism lets several small computers run one large language model together."
P3="What is the capital of Australia, and why was it chosen over Sydney and Melbourne? Answer in detail."
for k in "$@"; do for p in 1 2 3; do eval "prompt=\$P$p"
  SANGAMA_MTP=$k /root/bench.sh $TAG-m$k-p$p 128 "$prompt" > /dev/null 2>&1
  python3 - $TAG $k $p <<'PY'
import json, sys
tag, k, p = sys.argv[1:4]
try: d = json.load(open(f"/root/{tag}-m{k}-p{p}.json"))
except Exception: print(f"drafts={k} prompt={p} FAILED", open(f"/root/{tag}-m{k}-p{p}.log").read()[-200:].replace("\n", " ")); sys.exit()
try: ref = json.load(open(f"/root/{tag}-m0-p{p}.json"))["distributed_token_ids"]
except Exception: ref = None
s = d["speculation"]; x = d["distributed"]; n = d["generated_tokens"]; st = s["steps"] or n; b = d["breakdown"]
same = "n/a" if ref is None else d["distributed_token_ids"][:min(n, len(ref))] == ref[:min(n, len(ref))]
print(f"drafts={k} prompt={p} tokens={n} passes={st} tok/pass={n/st:.2f} accepted={s['accepted']}/{s['drafted']} tok/s={x['decode_tokens_per_second']:.1f} ms/pass={n/x['decode_tokens_per_second']/st*1000:.1f} ttft_ms={x['first_token_ms']:.0f} identical={same}")
PY
done; done
