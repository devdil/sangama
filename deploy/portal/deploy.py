#!/usr/bin/env python3
"""Ship only portal sources to the dedicated Linode and deploy over pinned-key SSH."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--host',required=True,help='Linode public IPv4 address from Terraform')
parser.add_argument('--domain',required=True,help='DNS name already pointing to that address')
args=parser.parse_args()
ipaddress.IPv4Address(args.host)
if not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?',args.domain) or '..' in args.domain:
    raise ValueError('Use a plain lowercase DNS hostname')
secret=ROOT/'.secrets/linode'
known=secret/'known_hosts'
public=(secret/'host_ed25519.pub').read_text().strip()
known.write_text(args.host+' '+public+'\n');known.chmod(0o600)
ssh=['ssh','-i',str(secret/'id_ed25519'),'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=10','deploy@'+args.host]
subprocess.run(ssh+['cloud-init status --wait'],check=True)
release=time.strftime('%Y%m%d%H%M%S',time.gmtime())
archive=ROOT/'work'/('portal-'+release+'.tgz');archive.parent.mkdir(exist_ok=True)
files=['.dockerignore','crates/network-auth/Cargo.toml','crates/network-auth/src/lib.rs','portal/src/admin.rs','portal/src/membership.rs','portal/Cargo.toml','portal/Cargo.lock','portal/schema.sql','portal/src/main.rs','portal/static/style.css','deploy/portal/Dockerfile','deploy/portal/compose.yaml','deploy/portal/Caddyfile','deploy/portal/init-db.sh','deploy/portal/prepare-secrets.py']
with tarfile.open(archive,'w:gz') as tar:
    for path in files:tar.add(ROOT/path,arcname=path,recursive=False)
remote=f'/srv/sangama/releases/{release}'
subprocess.run(ssh+[f'mkdir -p {remote}'],check=True)
with archive.open('rb') as data:subprocess.run(ssh+[f'tar -xzf - -C {remote}'],stdin=data,check=True)
script=f'''set -eu
umask 077
mkdir -p /srv/sangama/shared
cp {remote}/deploy/portal/prepare-secrets.py /srv/sangama/shared/prepare-secrets.py
python3 /srv/sangama/shared/prepare-secrets.py
ln -s /srv/sangama/shared/secrets {remote}/deploy/portal/secrets
printf '%s\\n' 'PUBLIC_HOST={args.domain}' 'PUBLIC_ORIGIN=https://{args.domain}' > {remote}/deploy/portal/.env
cd {remote}/deploy/portal
sudo docker compose --env-file .env build portal
sudo docker run --rm --network none --user 0:0 -e MEMBERSHIP_KEY_FILE=/keys/membership_key --mount type=bind,src=/srv/sangama/shared/secrets,dst=/keys sangama-portal:local authority-init
sudo docker compose --env-file .env up -d
ln -sfn {remote} /srv/sangama/current
'''
subprocess.run(ssh+['sh -s'],input=script,text=True,check=True)
url='https://'+args.domain
for _ in range(60):
    try:
        with urllib.request.urlopen(url+'/healthz',timeout=8) as r:
            if r.status==200 and r.read().strip()==b'ok':break
    except OSError:pass
    time.sleep(5)
else:raise RuntimeError('HTTPS health check failed; inspect Caddy and DNS. Do not bypass certificate checks.')
report={'deployed':True,'url':url,'host':args.host,'release':release,'https_verified':True}
(ROOT/'runs/portal-deployment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
