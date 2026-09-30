#!/bin/sh
# meshdeploy.sh BIN: roll a mesh-only build (a binary on the relay box, e.g. /root/sangama-mesh2) onto
# the relay, the client and every worker's mesh process. Loaded model stages are untouched.
# The relay's one open port serves the binary while its mesh is stopped.
set -u
cd "$(dirname "$0")"
BIN=$1
ADDR=$(python3 -c "import json;r=[h for h in json.load(open('hosts.json')) if h['name']=='relay'][0];print(f\"{r['ip']}:{r['relay_port']}\")")
SHA=$(./ssh.sh relay "sha256sum $BIN | cut -d' ' -f1" < /dev/null 2>/dev/null | tail -1)
TOK=$(openssl rand -hex 16)
./ssh.sh relay "pkill -f '^/root/sangama[-a-z0-9]* mesh'; sleep 1; mkdir -p /root/dist/$TOK && cp $BIN /root/dist/$TOK/sangama && cd /root/dist && setsid -f python3 -m http.server 9000 --bind 0.0.0.0 > /root/dist.log 2>&1 < /dev/null" < /dev/null > /dev/null 2>&1
sleep 2
for i in $(seq -w 0 19); do
  ( r=$(./ssh.sh w$i "curl -sf --retry 3 --max-time 600 -o /root/sangama-mesh.tmp http://$ADDR/$TOK/sangama && echo '$SHA  /root/sangama-mesh.tmp' | sha256sum -c --quiet && chmod +x /root/sangama-mesh.tmp && mv /root/sangama-mesh.tmp /root/sangama-mesh && echo ok" < /dev/null 2>&1 | tail -1); echo "w$i:$r" ) &
done | sort | tr '\n' ' '
wait; echo
./ssh.sh relay "pkill -f '^python3 -m http.[s]erver 9000'; rm -rf /root/dist; sleep 1; RUST_LOG=info,sangama=debug setsid -f $BIN mesh --config /root/mesh/relay.json > /root/relay.log 2>&1 < /dev/null" < /dev/null > /dev/null 2>&1
sleep 2
# A mesh exits at start if the portal's membership snapshot times out, so check and retry.
for i in $(seq -w 0 19); do
  ( r=$(./ssh.sh w$i "pkill -f '^/root/sangama-(cuda|mesh) mesh'; sleep 1; for t in 1 2 3; do setsid -f /root/sangama-mesh mesh --config /root/mesh/config.json > /root/mesh.log 2>&1 < /dev/null; sleep 8; pgrep -f '^/root/sangama-mesh mesh' > /dev/null && break; done; grep -c ReservationReqAccepted /root/mesh.log" < /dev/null 2>&1 | tail -1); echo "w$i:$r" ) &
done | sort | tr '\n' ' '
wait; echo
./ssh.sh relay "setsid -f $BIN mesh --config /root/mesh/client.json > /root/client-mesh.log 2>&1 < /dev/null; sed -i 's#^/root/sangama[-a-z0-9]* #$BIN #' /root/bench.sh; sleep 20; T=\$(cat /root/mesh/client.token); for p in \$(seq 7901 7920); do echo -n \"\$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H \"Authorization: Bearer \$T\" http://127.0.0.1:\$p/v1/qwen/info) \"; done" < /dev/null 2>&1 | tail -1
sed -i '' "s|^BIN=.*|BIN=$BIN|" cbench.sh
