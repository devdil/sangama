#!/bin/sh
# [STAGGER=seconds] [MTP=drafts] cbench.sh PREFIX CONCURRENCY [MAXTOK]: run CONCURRENCY generate clients at once on the relay box,
# cycling through four prompts, and report aggregate and per-stream speed. Each stream's tokens are
# compared with the solo run of the same prompt. "cbench.sh solo 4" first runs the four prompts one
# after another to make those references (/root/solo-c4-K.json).
cd "$(dirname "$0")"
P=$1; C=$2; N=${3:-128}; MTP_N=${MTP:-0}
./mark.sh "$P-c$C-start"
./ssh.sh relay "sh -s" <<EOF
cd /root/run; ulimit -n 65536
export SANGAMA_MTP=${MTP:-0}
BIN=/root/sangama-mesh4
PEERS=\$(seq -s, -f '127.0.0.1:%g' 7901 7920)
set -- "Write a short explanation of how pipeline parallelism lets several small computers run one large language model together." \
  "Write a Python function that returns the n-th Fibonacci number iteratively, then explain it." \
  "What is the capital of Australia, and why was it chosen over Sydney and Melbourne? Answer in detail." \
  "Describe how volunteer computing projects worked in the late 1990s and what they achieved."
start=\$(date +%s.%N)
for i in \$(seq 0 \$(($C - 1))); do
  k=\$((i % 4)); eval "prompt=\\\${\$((k + 1))}"
  [ "$P" = solo ] && [ \$i -gt 0 ] && wait
  ( \$BIN --token-file /root/mesh/client.token generate --model-dir model --peers \$PEERS --prompt "\$prompt" \
      --max-tokens $N --output /root/$P-c$C-\$i.json > /root/$P-c$C-\$i.log 2>&1; echo "\$k \$(date +%s.%N)" > /root/$P-c$C-\$i.done ) &
  sleep ${STAGGER:-0}
done
wait
end=\$(date +%s.%N)
python3 - "\$start" "\$end" <<'PY'
import json, sys, os, statistics as st
start, end = map(float, sys.argv[1:3])
P, C = "$P", $C
rows, fails, same = [], 0, []
passes = acc = drafted = 0
for i in range(C):
    try:
        d = json.load(open(f"/root/{P}-c{C}-{i}.json"))
    except Exception:
        fails += 1
        continue
    k = int(open(f"/root/{P}-c{C}-{i}.done").read().split()[0])
    x = d["distributed"]
    rows.append((d["generated_tokens"], x["decode_tokens_per_second"], x["first_token_ms"]))
    sp = d.get("speculation") or {}; passes += sp.get("steps", 0); acc += sp.get("accepted", 0); drafted += sp.get("drafted", 0)
    solo = f"/root/solo-c4-{k}.json"
    if P != "solo" and os.path.exists(solo):
        ref, got = json.load(open(solo))["distributed_token_ids"], d["distributed_token_ids"]; n = min(len(ref), len(got)); same.append(n > 0 and ref[:n] == got[:n])
wall = end - start
tokens = sum(r[0] for r in rows)
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else 0
tps = [r[1] for r in rows]; ttft = [r[2] / 1000 for r in rows]
out = dict(prefix=P, concurrency=C, ok=len(rows), failed=fails, wall_s=round(wall, 1), tokens=tokens,
           aggregate_tps=round(tokens / wall, 2), steady_tps=round(sum(tps), 1), stream_tps_p50=round(st.median(tps), 2) if tps else 0,
           stream_tps_p5=round(q(tps, 0.05), 2), ttft_s_p50=round(st.median(ttft), 2) if ttft else 0,
           ttft_s_p95=round(q(ttft, 0.95), 2), identical_to_solo=f"{sum(same)}/{len(same)}" if same else "n/a",
           mtp=$MTP_N, tokens_per_pass=round(tokens / passes, 2) if passes else 0, drafts_accepted=f"{acc}/{drafted}")
print(json.dumps(out))
open(f"/root/{P}-c{C}-summary.json", "w").write(json.dumps(out))
PY
EOF
./mark.sh "$P-c$C-end"
