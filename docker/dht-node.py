"""Run a discovery-only node bound to this container's private bridge address."""
import json
import os
import socket
import sys

role=sys.argv[1]
args=['sangama','dht-node','--state-dir','/tmp/node','--listen',
      f'/ip4/{socket.gethostbyname(socket.gethostname())}/tcp/9000']
if len(sys.argv)>2:
    args+=['--bootstrap',sys.argv[2]]
if role=='provider':
    args+=['--model-hash','a'*64,'--start','12','--end','24']
elif role=='seeker':
    args+=['--find-model','a'*64,'--wait-seconds','25']
elif role!='seed':
    raise ValueError('expected seed, provider, or seeker')
print(json.dumps({'network_namespace':os.readlink('/proc/self/ns/net'),'ip':socket.gethostbyname(socket.gethostname())}),flush=True)
os.execvp(args[0],args)
