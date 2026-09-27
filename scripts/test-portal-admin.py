#!/usr/bin/env python3
"""Test local operator forms using disposable membership rows; never print credentials."""
import hashlib,json,os,re,subprocess,urllib.request,urllib.parse,urllib.error,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:18080'
env=os.environ|{'PUBLIC_ORIGIN':BASE,'PUBLIC_HOST':'127.0.0.1'}
compose=['docker','compose','-f','deploy/portal/compose.yaml','-f','deploy/portal/compose.test.yaml','-p','sangama-portal-test']
def sql(query):
 return subprocess.run(compose+['exec','-T','postgres','psql','-U','postgres','-d','sangama','-Atc',query],cwd=ROOT,env=env,check=True,capture_output=True,text=True).stdout.strip()
credential=(ROOT/'deploy/portal/secrets/admin_token').read_text().strip()
def post(action,token=credential,origin=BASE,**extra):
 fields={'admin_token':token,'action':action,'role':'worker','peer':''}|extra
 req=urllib.request.Request(BASE+'/admin',data=urllib.parse.urlencode(fields).encode(),headers={'Origin':origin})
 try:
  with urllib.request.urlopen(req) as r:
   assert r.headers['Cache-Control']=='no-store'
   return r.status,r.read().decode()
 except urllib.error.HTTPError as e:return e.code,e.read().decode()
state=ROOT/'work'/('admin-test-'+uuid.uuid4().hex)
peer=subprocess.run([str(ROOT/'target/debug/sangama'),'mesh-identity','--state-dir',str(state)],check=True,capture_output=True,text=True).stdout.strip()
assert re.fullmatch('[A-Za-z0-9]+',peer)
digest=None
try:
 assert post('inspect',token='wrong')[0]==403
 assert post('invite',origin='https://evil.invalid')[0]==403
 assert post('invite',role='administrator')[0]==400
 code,body=post('invite');assert code==200
 invitation=re.search(r'<pre>([a-f0-9]{64})</pre>',body).group(1)
 digest=hashlib.sha256(invitation.encode()).hexdigest()
 assert sql(f"SELECT role FROM network_invitations WHERE token_hash='{digest}'")=='worker'
 sql(f"INSERT INTO network_members(peer_id,role) VALUES ('{peer}','worker')")
 code,body=post('inspect');assert code==200 and peer in body and 'Admitted' in body and credential not in body
 assert post('revoke',peer=peer)[0]==400
 assert post('revoke',peer=peer,confirm='yes')[0]==200
 assert sql(f"SELECT revoked FROM network_members WHERE peer_id='{peer}'")=='t'
 assert 'Revoked' in post('inspect')[1]
 for page,expected in [('/connect','mesh-join'),('/admin','Admin credential'),('/','network membership')]:
  assert expected in urllib.request.urlopen(BASE+page).read().decode()
 assert any(post('inspect',token='wrong')[0]==429 for _ in range(21))
 report={'passed':True,'checks':['operator credential required','cross-origin admin POST rejected','role validation','invitation persisted as hash','authenticated membership listing','credential not reflected','revocation confirmation required','revocation persisted and displayed','connection setup page','operator request rate limit'],'scope':'Local Docker portal and PostgreSQL; temporary membership/invitation rows removed.'}
 (ROOT/'runs/portal-admin-test.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
finally:
 sql(f"DELETE FROM network_members WHERE peer_id='{peer}'")
 if digest:sql(f"DELETE FROM network_invitations WHERE token_hash='{digest}'")
 import shutil
 shutil.rmtree(state)
