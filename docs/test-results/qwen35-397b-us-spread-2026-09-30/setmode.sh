#!/bin/sh
# setmode.sh MODE: restart every mesh process in one of tcp-relay, quic-relay, tcp-direct,
# quic-direct. Loaded model stages are untouched.
cd "$(dirname "$0")"; M=$1
./ssh.sh relay "pkill -f '^/root/sangama-mesh [m]esh'; sleep 1; ulimit -n 65536; RUST_LOG=info setsid -f /root/sangama-mesh mesh --config /root/mesh/relay.json > /root/relay.log 2>&1 < /dev/null; sleep 2" < /dev/null
for i in 0 1 2; do ( ./ssh.sh w0$i "pkill -f '^/root/sangama-mesh [m]esh'; sleep 1; cp /root/mesh/config-$M.json /root/mesh/config.json; for t in 1 2 3; do setsid -f /root/sangama-mesh mesh --config /root/mesh/config.json > /root/mesh.log 2>&1 < /dev/null; sleep 6; pgrep -f '^/root/sangama-mesh [m]esh' > /dev/null && break; done" < /dev/null ) & done; wait
./ssh.sh relay "ulimit -n 65536; setsid -f /root/sangama-mesh mesh --config /root/mesh/client-$M.json > /root/client-mesh.log 2>&1 < /dev/null; sleep 40; for k in 1 2 3; do /root/bench.sh warm 16 'Explain peer-to-peer computing in one short sentence.' > /dev/null 2>&1; done; sed -E 's/\x1b\[[0-9;]*m//g' /root/client-mesh.log | grep 'mesh load' | tail -1 | grep -oE 'direct=[0-9]+ relayed=[0-9]+'; sed -E 's/\x1b\[[0-9;]*m//g' /root/client-mesh.log | grep -c 'quic-v1'" < /dev/null
