"""Initialize disposable Docker volumes with separate host/client identities."""
import os
from pathlib import Path
import secrets
import subprocess

folders = {name: Path('/init') / name for name in ['worker0', 'worker1', 'client']}
for folder in folders.values():
    folder.mkdir(exist_ok=True)
    folder.chmod(0o700)

def key(folder, name):
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(folder/name)], check=True)

for name in ['worker0', 'worker1']:
    key(folders[name], 'host')
for name in ['worker0', 'client']:
    key(folders[name], 'identity')

client_pub = (folders['client']/'identity.pub').read_text().strip()
worker_pub = (folders['worker0']/'identity.pub').read_text().strip()
for name, port, public_keys in [('worker0', 7901, [client_pub]), ('worker1', 7902, [client_pub, worker_pub])]:
    lines = [f'restrict,port-forwarding,permitopen="127.0.0.1:{port}",command="/usr/bin/false" {pub}\n' for pub in public_keys]
    (folders[name]/'authorized_keys').write_text(''.join(lines))
known = ''.join(f'[{name}]:2222 '+(folders[name]/'host.pub').read_text() for name in ['worker0', 'worker1'])
for name in ['worker0', 'client']:
    (folders[name]/'known_hosts').write_text(known)
token = secrets.token_hex(32)
for folder in folders.values():
    (folder/'token').write_text(token+'\n')
    for file in folder.iterdir():
        file.chmod(0o600)
        os.chown(file,10001,10001)
    os.chown(folder,10001,10001)
os.chown('/init/reports',10001,10001)
print('Disposable identities and token prepared; no secret values printed.')
