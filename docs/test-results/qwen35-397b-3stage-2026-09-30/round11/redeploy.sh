#!/bin/sh
# [BIN=/root/sangama-cudaN] [PMIN=x] [SLOTS=n] [ROLLBACK=n] [BATCH=0|1] redeploy.sh
# Three-stage fleet: optionally roll a new worker binary from the relay box, then restart the
# three workers with the given settings. The relay's open port serves the binary while its
# mesh is stopped; peers keep their direct connections meanwhile.
set -u
cd "$(dirname "$0")"
ADDR=$(python3 -c "import json;r=[h for h in json.load(open('hosts.json')) if h['name']=='relay'][0];print(f\"{r['ip']}:{r['relay_port']}\")")
if [ -n "${BIN:-}" ]; then
  SHA=$(./ssh.sh relay "sha256sum $BIN | cut -d' ' -f1" < /dev/null | tail -1); TOK=$(openssl rand -hex 16)
  ./ssh.sh relay "pkill -f '^/root/sangama-mesh [m]esh --config /root/mesh/relay'; sleep 1; mkdir -p /root/dist/$TOK && cp $BIN /root/dist/$TOK/sangama && cd /root/dist && setsid -f python3 -m http.server 9000 --bind 0.0.0.0 > /root/dist.log 2>&1 < /dev/null; sleep 1" < /dev/null
  for i in 0 1 2; do ( ./ssh.sh w0$i "curl -sf --retry 3 --max-time 600 -o /root/sangama-cuda.new http://$ADDR/$TOK/sangama && echo '$SHA  /root/sangama-cuda.new' | sha256sum -c --quiet && chmod +x /root/sangama-cuda.new && pkill -x sangama-cuda; sleep 1; mv /root/sangama-cuda.new /root/sangama-cuda && echo w0$i:binary-ok" < /dev/null ) & done; wait
  ./ssh.sh relay "pkill -f '^python3 -m http.[s]erver 9000'; rm -rf /root/dist; sleep 1; ulimit -n 65536; RUST_LOG=info setsid -f /root/sangama-mesh mesh --config /root/mesh/relay.json > /root/relay.log 2>&1 < /dev/null" < /dev/null
fi
for i in 0 1 2; do n=w0$i; M=""; [ $i = 2 ] && M="MTP=mtp-q4_k_m.gguf SANGAMA_MTP_P_MIN=${PMIN:-0}"
  ( r=$(./ssh.sh $n "SANGAMA_BATCH=${BATCH:-1} ${UNIFIED:+SANGAMA_KV_UNIFIED=$UNIFIED} ${BATCH_MAX:+SANGAMA_BATCH_MAX=$BATCH_MAX} SLOTS=${SLOTS:-8} ROLLBACK=${ROLLBACK:-8} BUDGET=96000 LAST=2 $M sh /root/sangama/start.sh $n $i 2>&1 | grep -oE 'READY|FAILED'; nvidia-smi --query-gpu=memory.used --format=csv,noheader" < /dev/null 2>&1 | tr '\n' ' '); echo "$n: $r" ) &
done; wait
./ssh.sh relay 'sleep 8; for k in 1 2; do /root/bench.sh warm 16 "Explain peer-to-peer computing in one short sentence." | head -1 | cut -c1-110; done' < /dev/null
