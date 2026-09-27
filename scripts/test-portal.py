#!/usr/bin/env python3
"""Test the Docker portal/Postgres stack started with compose.test.yaml."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:18080'
env=os.environ|{'PUBLIC_ORIGIN':BASE,'PUBLIC_HOST':'127.0.0.1'}
compose=['docker','compose','-f','deploy/portal/compose.yaml','-f','deploy/portal/compose.test.yaml','-p','sangama-portal-test']
def run(*args):return subprocess.run(compose+list(args),cwd=ROOT,env=env,capture_output=True,text=True,check=True).stdout
for attempt in range(100):
    try:
        urllib.request.urlopen(BASE+'/healthz',timeout=2).read();break
    except OSError:time.sleep(.3)
else:raise RuntimeError('Portal did not become healthy')
with urllib.request.urlopen(BASE) as response:
    assert "default-src 'none'" in response.headers['Content-Security-Policy']
    assert 'No devices registered yet' in response.read().decode()
def invite():return run('exec','-T','portal','/usr/local/bin/sangama-portal','invite').strip()
def register(fields,origin=BASE):
    req=urllib.request.Request(BASE+'/join',data=urllib.parse.urlencode(fields).encode(),headers={'Origin':origin})
    try:
        with urllib.request.urlopen(req) as response:return response.status,response.read().decode()
    except urllib.error.HTTPError as e:return e.code,e.read().decode()
code=invite()
fields={'name':'Cedar <script>alert(1)</script>','peer_id':'12D3KooW'+uuid.uuid4().hex,'platform':'macOS','memory_gib':'24','invitation':code,'consent':'yes'}
assert register(fields,origin='https://evil.example')[0]==403
assert register(fields|{'consent':'no'})[0]==400
assert register(fields|{'invitation':'0'*64})[0]==403
assert register(fields)[0]==201
assert register(fields)[0]==403
second=invite()
assert register(fields|{'invitation':second})[0]==409
# The duplicate-device transaction must not consume the second invitation.
assert register(fields|{'invitation':second,'name':'Birch Linux','platform':'Linux','peer_id':'12D3KooW'+uuid.uuid4().hex})[0]==201
html=urllib.request.urlopen(BASE).read().decode()
assert '&lt;script&gt;' in html and '<script>' not in html
assert '2 registered' in html and 'Registered · not admitted' in html
role=run('exec','-T','postgres','psql','-U','postgres','-d','sangama','-Atc',"SELECT rolsuper FROM pg_roles WHERE rolname='sangama'").strip()
assert role=='f'
ids=run('ps','-q').splitlines()
inspection=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True))
for container in inspection:
    service=container['Config']['Labels']['com.docker.compose.service']
    if service=='postgres':assert not container['HostConfig'].get('PortBindings')
    if service=='portal':
        assert container['Config']['User']=='10001:10001'
        assert container['HostConfig']['ReadonlyRootfs']
        assert 'ALL' in container['HostConfig']['CapDrop']
run('restart','portal')
for _ in range(100):
    try:
        if '2 registered' in urllib.request.urlopen(BASE,timeout=2).read().decode():break
    except OSError:pass
    time.sleep(.2)
else:raise AssertionError('Registrations did not persist across restart')
report={'passed':True,'database':'PostgreSQL 17','checks':['health and directory rendering','CSP header','cross-origin POST rejected','publication consent enforced','invalid invitation rejected','single-use invitation enforced','duplicate peer rejected without consuming invitation','HTML escapes stored user content','application DB role is not superuser','Postgres has no published ports','portal is non-root/read-only/capabilities dropped','registrations survive portal restart'],'scope':'Local Docker test. Public HTTPS and Linode deployment are not verified.'}
(ROOT/'runs/portal-test.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
