#!/usr/bin/env python3
"""Exercise invite-only accounts against the local Docker portal and PostgreSQL."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
def run(*args):
    return subprocess.run(compose+list(args),cwd=ROOT,env=env,capture_output=True,text=True,check=True).stdout.strip()
def sql(query):
    return run('exec','-T','postgres','psql','-U','postgres','-d','sangama','-Atc',query)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args): return None
opener=urllib.request.build_opener(NoRedirect)
def request(path, fields=None, cookie=None, origin=BASE):
    headers={'Origin':origin}
    if cookie: headers['Cookie']=cookie
    req=urllib.request.Request(BASE+path,data=None if fields is None else urllib.parse.urlencode(fields).encode(),headers=headers)
    try: r=opener.open(req,timeout=10)
    except urllib.error.HTTPError as e: r=e
    with r: return r.status, r.headers, r.read().decode()
def healthy():
    for _ in range(100):
        try:
            if request('/healthz')[0]==200: return
        except OSError: pass
        time.sleep(.3)
    raise AssertionError('Portal did not become healthy')
healthy()
# Fresh process resets the intentionally conservative preview-wide attempt limit.
run('restart','portal');healthy()
names=['test_'+uuid.uuid4().hex[:20] for _ in range(4)]
codes=[]
directory_count=sql('SELECT count(*) FROM registrations')
def invite():
    code=run('exec','-T','portal','/usr/local/bin/sangama-portal','invite');codes.append(code);return code
try:
    code=invite()
    fields={'username':names[0],'password':'a test passphrase '+uuid.uuid4().hex,'invitation':code}
    status,headers,body=request('/join')
    assert status==200 and "default-src 'none'" in headers['Content-Security-Policy']
    from html.parser import HTMLParser
    class Inputs(HTMLParser):
        def __init__(self): super().__init__();self.names=[]
        def handle_starttag(self,tag,attrs):
            if tag=='input': self.names.append(dict(attrs).get('name'))
    inputs=Inputs();inputs.feed(body)
    assert inputs.names==['invitation','username','password']
    assert request('/join',fields,origin='https://evil.example')[0]==403
    assert request('/join',fields|{'password':'short'})[0]==400
    assert request('/join',fields|{'invitation':'0'*64})[0]==403
    assert request('/join',fields)[0]==201
    assert request('/join',fields|{'username':names[1]})[0]==403
    second=invite()
    assert request('/join',fields|{'invitation':second,'username':names[0].upper()})[0]==409
    assert request('/join',fields|{'invitation':second,'username':names[1]})[0]==201
    expired=invite()
    digest=hashlib.sha256(expired.encode()).hexdigest()
    sql(f"UPDATE invitations SET expires_at=now()-interval '1 second' WHERE token_hash='{digest}'")
    assert request('/join',fields|{'invitation':expired})[0]==403
    race=invite()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda name: request('/join',fields|{'invitation':race,'username':name})[0],names[2:]))
    assert sorted(results)==[201,403],results
    stored=sql(f"SELECT password_hash FROM accounts WHERE username='{names[0]}'")
    assert stored.startswith('$argon2id$') and fields['password'] not in stored
    assert request('/account')[0]==303
    login={k:fields[k] for k in ['username','password']}
    assert request('/signin',login|{'password':'wrong'})[0]==401
    status,headers,_=request('/signin',login|{'username':names[0].upper()})
    assert status==303 and headers['Location']=='/account'
    cookie=headers['Set-Cookie'].split(';')[0]
    assert 'HttpOnly' in headers['Set-Cookie'] and 'SameSite=Strict' in headers['Set-Cookie']
    assert names[0] in request('/account',cookie=cookie)[2]
    assert 'Device directory' in request('/',cookie=cookie)[2] and 'Device directory' not in request('/')[2]
    for path in ['/connect','/about']:
        assert request(path)[0]==303 and request(path,cookie=cookie)[0]==200
    session=cookie.split('=',1)[1]
    assert sql(f"SELECT count(*) FROM account_sessions WHERE token_hash='{hashlib.sha256(session.encode()).hexdigest()}'")=='1'
    assert sql(f"SELECT count(*) FROM account_sessions WHERE token_hash='{session}'")=='0'
    run('restart','portal');healthy()
    assert request('/account',cookie=cookie)[0]==200
    _,headers,_=request('/signin',login)
    rotated=headers['Set-Cookie'].split(';')[0]
    assert rotated!=cookie and request('/account',cookie=cookie)[0]==303
    assert request('/signout',{},cookie=rotated,origin='https://evil.example')[0]==403
    assert request('/account',cookie=rotated)[0]==200
    assert request('/signout',{},cookie=rotated)[0]==303
    assert request('/account',cookie=rotated)[0]==303
    _,headers,_=request('/signin',login)
    expired_cookie=headers['Set-Cookie'].split(';')[0]
    sql(f"UPDATE account_sessions SET expires_at=now()-interval '1 second' WHERE account_id=(SELECT id FROM accounts WHERE username='{names[0]}')")
    assert request('/account',cookie=expired_cookie)[0]==303
    assert any(request('/signin',login|{'password':'wrong'})[0]==429 for _ in range(21))
    assert sql('SELECT count(*) FROM registrations')==directory_count
    assert all(name not in request('/')[2] for name in names)
    report={'passed':True,'checks':['signup has exactly invite code, username, password','cross-origin signup/signout blocked','password length enforced','invalid/expired/reused invites rejected','concurrent invite redemption has one winner','case-insensitive username uniqueness','duplicate signup preserves invitation','Argon2id password storage','authenticated account page','directory, connect and how-it-works require sign-in','HttpOnly SameSite cookie','only session hashes stored','session survives restart','sign-in rotates session','sign-out revokes session','expired session rejected','authentication attempt rate limit','legacy directory preserved and account names not published'],'scope':'Local Docker portal and PostgreSQL. No production deployment.'}
    (ROOT/'runs/portal-test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:
    for name in names: sql(f"DELETE FROM accounts WHERE username='{name}'")
    for code in codes: sql(f"DELETE FROM invitations WHERE token_hash='{hashlib.sha256(code.encode()).hexdigest()}'")
    run('restart','portal');healthy()
