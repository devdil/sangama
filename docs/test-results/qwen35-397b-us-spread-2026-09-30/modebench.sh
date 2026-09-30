#!/bin/sh
# modebench.sh MODE: switch the mesh to MODE, then one request (3 prompts) and 16 and 48 at once.
cd "$(dirname "$0")"; M=$1
echo "== $M: $(./setmode.sh $M | tr '\n' ' ')"
./ssh.sh relay "/root/r10.sh q-$M 0; python3 /root/pass.py q-$M-m0-p1 q-$M-m0-p2" < /dev/null
for c in 16 48; do STAGGER=0.1 ./cbench.sh q-$M $c 128 2>&1 | tail -1 | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print({k:d[k] for k in ('concurrency','ok','failed','wall_s','aggregate_tps','steady_tps','stream_tps_p50','ttft_s_p50','identical_to_solo')})"; done
