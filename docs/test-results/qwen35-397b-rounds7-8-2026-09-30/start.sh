#!/bin/sh
# [SLOTS=N] [BUDGET=MiB] [MTP=file.gguf] start.sh NAME STAGE: start this worker's stage (llama.cpp, CUDA,
# 16 GiB budget by default) and its relay-only mesh node. MTP (final stage only) drafts tokens.
NAME=$1; STAGE=$2
cd /root/sangama
M="python3 scripts/sysmetrics.py event --dir /root/metrics"
pkill -f '^/root/sangama-cuda'; sleep 2
GGUF=$(python3 -c "import json;print(json.load(open('model/manifest.json'))['shards'][$STAGE]['file'])")
NEXT=""; [ "$STAGE" -lt 19 ] && NEXT="--allow-next 127.0.0.1:$((7902 + STAGE))"
$M load-start --detail "$GGUF"
setsid -f /root/sangama-cuda --token-file /root/mesh/worker.token qwen-worker --model-dir /root/sangama/model --shard "$STAGE" \
  --engine llamacpp --device cuda --gguf "$GGUF" --memory-budget-mib ${BUDGET:-16384} --slots ${SLOTS:-1} ${MTP:+--mtp-gguf $MTP} --listen 127.0.0.1:7900 $NEXT > /root/worker.log 2>&1 < /dev/null
for i in $(seq 600); do
  curl -sf -H "Authorization: Bearer $(cat /root/mesh/worker.token)" http://127.0.0.1:7900/v1/qwen/info > /root/info.json && break
  sleep 1
done
$M load-end --detail "$GGUF"
setsid -f /root/sangama-cuda mesh --config /root/mesh/config.json > /root/mesh.log 2>&1 < /dev/null
[ -s /root/info.json ] && echo "READY $(head -c 300 /root/info.json)" || { echo FAILED; tail -5 /root/worker.log; }
