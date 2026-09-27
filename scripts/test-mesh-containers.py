#!/usr/bin/env python3
"""Exercise real Qwen inference over an admitted relay across isolated Docker networks.
All resources use unique names and are removed; no cloud service or host port is used.
"""
import argparse, json, os, re, secrets, subprocess, time, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def run(*args,**kw):
    return subprocess.run(list(args),check=True,text=True,capture_output=True,**kw)
def docker(*args,**kw):return run('docker',*args,**kw)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='sangama:mesh-test')
    parser.add_argument('--portal-image',default='sangama-portal:mesh-test')
    parser.add_argument('--output-dir',default='runs/mesh-simulation')
    args=parser.parse_args()
    out=(ROOT/args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    private=out/'private';private.mkdir(mode=0o700,exist_ok=True);private.chmod(0o700)
    prefix='sangama-mesh-'+uuid.uuid4().hex[:8]
    networks=[];containers=[];volumes=[];checks={};report={'simulation':'separate Docker bridge networks; forced Circuit Relay v2; CPU Qwen2.5-0.5B','checks':checks}
    model=ROOT/'.models/qwen2.5-0.5b-instruct';manifest=json.loads((model/'manifest.json').read_text())
    image=args.image
    def create(name,net,extra,entry='python3',command=None):
        cname=prefix+'-'+name
        docker('create','--name',cname,'--network',net,'--network-alias',name,'--label','sangama.test=true',
               '--read-only','--cap-drop=ALL','--security-opt','no-new-privileges=true','--pids-limit','128',
               '--tmpfs','/tmp:rw,nosuid,noexec,size=32m,mode=1777','--entrypoint',entry,*extra,*(command or []))
        containers.append(cname);return cname
    def execpy(name,code):return docker('exec',name,'python3','-c',code)
    def wait(fn,seconds=90):
        deadline=time.monotonic()+seconds;last=None
        while time.monotonic()<deadline:
            try:
                value=fn()
                if value:return value
            except Exception as e:last=e
            time.sleep(1)
        raise RuntimeError(f'Timeout: {last}')
    try:
        for role in ['a','b','c','db']:
            n=prefix+'-'+role;docker('network','create','--internal',n);networks.append(n)
        nets=dict(zip(['a','b','c','db'],networks))
        # Authority key is never mounted into a worker or relay.
        authority=private/'authority'
        run(str(ROOT/'target/debug/sangama'),'mesh-identity','--state-dir',str(authority))
        (authority/'identity.key').chmod(0o444)
        dbpassword=secrets.token_hex(32);(private/'dbpass').write_text(dbpassword);(private/'dbpass').chmod(0o444)
        init=private/'init.sql';init.write_text("CREATE USER sangama PASSWORD '"+dbpassword+"';\nCREATE DATABASE sangama OWNER sangama;\n")
        init.chmod(0o444)
        db=create('postgres',nets['db'],['--mount',f'type=bind,src={private}/dbpass,dst=/run/dbpass,readonly','--mount',f'type=bind,src={init},dst=/docker-entrypoint-initdb.d/10-init.sql,readonly','--tmpfs','/var/lib/postgresql/data:rw,size=512m','-e','POSTGRES_PASSWORD_FILE=/run/dbpass','postgres:17-bookworm'],entry='/usr/local/bin/docker-entrypoint.sh',command=['postgres'])
        # Postgres requires its normal setup capabilities; recreate only this test database accordingly.
        docker('rm',db);containers.remove(db)
        docker('run','-d','--name',db,'--network',nets['db'],'--network-alias','postgres','--tmpfs','/var/lib/postgresql/data:rw,size=512m','--mount',f'type=bind,src={private}/dbpass,dst=/run/dbpass,readonly','--mount',f'type=bind,src={init},dst=/docker-entrypoint-initdb.d/10-init.sql,readonly','-e','POSTGRES_PASSWORD_FILE=/run/dbpass','postgres:17-bookworm');containers.append(db)
        wait(lambda:docker('exec',db,'pg_isready','-U','postgres').returncode==0)
        portal=create('portal',nets['db'],['--mount',f'type=bind,src={private}/dbpass,dst=/run/dbpass,readonly','--mount',f'type=bind,src={authority}/identity.key,dst=/run/authority.key,readonly','-e','DATABASE_PASSWORD_FILE=/run/dbpass','-e','DATABASE_HOST=postgres','-e','PUBLIC_ORIGIN=http://portal:8080','-e','SIMULATION_HTTP=1','-e','NETWORK_ID=simulation','-e','MEMBERSHIP_KEY_FILE=/run/authority.key',args.portal_image],entry='/usr/local/bin/sangama-portal')
        for n in ['a','b','c']:docker('network','connect','--alias','portal',nets[n],portal)
        docker('start',portal)
        roles=['relay','worker0','worker1','client','outsider']
        ids={};tokens={};state={};names={};configs={}
        for role in roles:
            state[role]=private/role
            ids[role]=run(str(ROOT/'target/debug/sangama'),'mesh-identity','--state-dir',str(state[role])).stdout.strip()
            tokens[role]=secrets.token_hex(32)
            (state[role]/'token').write_text(tokens[role]);(state[role]/'token').chmod(0o600)
            (state[role]/'api-token').write_text(secrets.token_hex(32));(state[role]/'api-token').chmod(0o600)
            (state[role]/'authority.pub').write_bytes((authority/'public.key').read_bytes())
            v=prefix+'-'+role+'-state';docker('volume','create',v);volumes.append(v)
        vmap=dict(zip(roles,volumes))
        # Keep containers idle during enrollment; start daemons with exec once configuration is installed.
        for role in roles:
            net=nets['a' if role=='worker0' else 'b' if role=='worker1' else 'c']
            extra=['--mount',f'type=bind,src={ROOT}/scripts/mesh-node.py,dst=/app/mesh-node.py,readonly','--mount',f'type=volume,src={vmap[role]},dst=/state','--memory','3g' if role.startswith('worker') else '768m','--cpus','2']
            files=['config.json','manifest.json']+([manifest['shards'][int(role[-1])]['file']] if role.startswith('worker') else ['tokenizer.json'] if role=='client' else [])
            if role=='client':
                extra+=['--mount',f'type=bind,src={ROOT}/.tools/opencode-linux/opencode,dst=/usr/local/bin/opencode,readonly','--mount',f'type=bind,src={ROOT}/integrations/opencode/opencode.json,dst=/app/opencode.json,readonly']
            if role in ['worker0','worker1','client']:
                for f in files:extra+=['--mount',f'type=bind,src={model/f},dst=/model/{f},readonly']
            names[role]=create(role,net,extra+[image],command=['-c','import time; time.sleep(7200)'])
            if role=='relay':
                for n in ['a','b']:docker('network','connect',nets[n],names[role])
            docker('start',names[role])
        relay_ips=json.loads(docker('inspect',names['relay']).stdout)[0]['NetworkSettings']['Networks']
        for role in roles:
            net=nets['a' if role=='worker0' else 'b' if role=='worker1' else 'c']
            relay_addr=f"/ip4/{relay_ips[net]['IPAddress']}/tcp/9000/p2p/{ids['relay']}"
            config={'state_dir':'/state','authority_file':'/state/authority.pub','network':'simulation','portal':'http://portal:8080','test_http':True,'listen':'/ip4/0.0.0.0/tcp/9000','external':[], 'relay_server':role=='relay','relay':None if role=='relay' else relay_addr,'worker':'127.0.0.1:7900' if role.startswith('worker') else None,'token_file':'/state/token','force_relay':True,'bridges':[]}
            if role.startswith('worker'):config['managed']={'model_dir':'/model','device':'cpu','memory_budget_mib':2200}
            if role=='relay':config['external']=[f"/ip4/{entry['IPAddress']}/tcp/9000" for entry in relay_ips.values()]
            if role in ['client','worker0','worker1']:
                config['bridges']=[{'listen':f'127.0.0.1:{7901+i}','peer':ids[f'worker{i}']} for i in range(2)]
            configs[role]=config;(state[role]/'config.json').write_text(json.dumps(config))
            if role!='outsider':
                invitation=wait(lambda:docker('exec',portal,'sangama-portal','network-invite','worker' if role.startswith('worker') else role).stdout.strip())
                (state[role]/'invite').write_text(invitation);(state[role]/'invite').chmod(0o600)
            # Copy only this role's keys into its volume, then restrict ownership.
            docker('run','--rm','--network','none','--user','0:0','--entrypoint','python3','--mount',f'type=bind,src={state[role]},dst=/input,readonly','--mount',f'type=volume,src={vmap[role]},dst=/output',image,'-c',"import pathlib,shutil,os; src=pathlib.Path('/input');dst=pathlib.Path('/output');[shutil.copyfile(p,dst/p.name) for p in src.iterdir() if p.is_file() and p.name!='node.lock'];os.chown(dst,10001,10001);os.chmod(dst,0o700);[(os.chown(p,10001,10001),os.chmod(p,0o600)) for p in dst.iterdir()]")
            if role!='outsider':
                docker('exec',names[role],'sangama','mesh-join','--config','/state/config.json','--invitation-file','/state/invite')
                replay=subprocess.run(['docker','exec',names[role],'sangama','mesh-join','--config','/state/config.json','--invitation-file','/state/invite'],capture_output=True,text=True)
                assert replay.returncode!=0
        checks['ownership_join_and_single_use_invitations']=True
        # Outsider holds a fresh identity and authority public key, but no invitation.
        outsider=subprocess.run(['docker','exec',names['outsider'],'sangama','mesh','--config','/state/config.json'],capture_output=True,text=True,timeout=15)
        assert outsider.returncode!=0 and 'membership' in outsider.stderr
        checks['unenrolled_node_rejected']=True
        for role in ['worker0','worker1']:
            index=int(role[-1]);cmd=['python3','/app/mesh-node.py','--binary','/usr/local/bin/sangama','--config','/state/config.json','--model-dir','/model','--device','cpu']
            execpy(names[role],f"import subprocess; subprocess.Popen({cmd!r},stdout=open('/state/mesh.log','w'),stderr=subprocess.STDOUT)")
            time.sleep(1)
        for role in ['relay','client']:
            execpy(names[role],"import subprocess;subprocess.Popen(['sangama','mesh','--config','/state/config.json'],stdout=open('/state/mesh.log','w'),stderr=subprocess.STDOUT)")
            time.sleep(2)
        def probe(port,path="/v1/qwen/info"):
            return execpy(names['client'],f"import urllib.request;print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:{port}{path}',headers={{'Authorization':'Bearer '+open('/state/token').read()}}),timeout=5).status)").stdout.strip()=='200'
        wait(lambda:probe(7901,'/v1/node/capacity') and probe(7902,'/v1/node/capacity'),120)
        allocation=json.loads(docker('exec',names['client'],'sangama','--token-file','/state/token','mesh-allocate','--model-dir','/model','--candidates','127.0.0.1:7902,127.0.0.1:7901',timeout=120).stdout)
        assert [a['shard'] for a in allocation]==[0,1]
        assert all(a['candidate']['capacity']['loaded_shard'] is None for a in allocation)
        report['cold_placement']=allocation;checks['automatic_memory_fitting_cold_shard_loading']=True
        wait(lambda:probe(7901) and probe(7902),120)
        checks['both_workers_reachable_through_relay']=True
        inspection=json.loads(docker('inspect',names['worker0'],names['worker1'],names['client'],names['relay']).stdout)
        isolation=[]
        for item in inspection:
            host=item['HostConfig'];assert item['Config']['User']=='10001:10001' and host['ReadonlyRootfs'] and not host.get('PortBindings') and 'ALL' in host['CapDrop']
            files=sorted(m['Destination'].split('/')[-1] for m in item['Mounts'] if m['Destination'].startswith('/model/'))
            isolation.append({'role':item['Name'].removeprefix('/'+prefix+'-'),'model_files':files,'image_id':item['Image'],'read_only_root':True,'host_ports_published':False,'capabilities_dropped':host['CapDrop']})
        assert 'model.safetensors' not in str(isolation)
        report['container_isolation']=isolation;checks['worker_and_client_container_isolation']=True
        ip1=json.loads(docker('inspect',names['worker1']).stdout)[0]['NetworkSettings']['Networks'][nets['b']]['IPAddress']
        direct=execpy(names['worker0'],f"import socket; s=socket.socket();s.settimeout(2); print(s.connect_ex(({ip1!r},9000)))").stdout.strip()
        assert direct!='0';checks['worker_networks_cannot_dial_each_other']=True
        # Reservations are exclusive and can be released by their owning session.
        def api(port,path,body):
            code="import json,urllib.request,urllib.error; r=urllib.request.Request('http://127.0.0.1:%d%s',data=%r,headers={'Authorization':'Bearer '+open('/state/token').read(),'Content-Type':'application/json'});\ntry: print(urllib.request.urlopen(r,timeout=15).status)\nexcept urllib.error.HTTPError as e: print(e.code)" % (port,path,json.dumps(body).encode())
            return int(execpy(names['client'],code).stdout.strip())
        import hashlib
        assignment={'lease':str(uuid.uuid4()),'model_hash':hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest(),'shard':1}
        assert api(7902,'/v1/node/load',assignment)==409
        assert api(7902,'/v1/node/reserve',assignment|{'model_hash':'0'*64})==409
        assert api(7902,'/v1/node/reserve',assignment)==200
        assert api(7902,'/v1/node/reserve',assignment|{'lease':str(uuid.uuid4())})==409
        assert api(7902,'/v1/qwen/reserve',{'session':str(uuid.uuid4())})==503
        assert api(7902,'/v1/node/release',assignment|{'lease':str(uuid.uuid4())})==409
        assert api(7902,'/v1/node/release',assignment)==200
        checks['placement_lease_ownership_and_inference_exclusion']=True
        owner=str(uuid.uuid4());other=str(uuid.uuid4())
        assert api(7902,'/v1/qwen/reserve',{'session':owner})==200
        assert api(7902,'/v1/qwen/reserve',{'session':other})==409
        assert api(7902,'/v1/qwen/reset',{'session':other})==409
        assert api(7902,'/v1/qwen/reset',{'session':owner})==200
        checks['exclusive_session_reservations']=True
        plan=json.loads(docker('exec',names['client'],'sangama','--token-file','/state/token','mesh-plan','--model-dir','/model','--candidates','127.0.0.1:7902,127.0.0.1:7901').stdout)
        assert plan['peers']==['127.0.0.1:7901','127.0.0.1:7902'];report['placement']=plan;checks['automatic_ready_route_selection']=True
        def read_offers():
            data=execpy(names['client'],"import urllib.request;print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7901/v1/mesh/offers',headers={'Authorization':'Bearer '+open('/state/token').read()}),timeout=5).read().decode())").stdout
            value=json.loads(data)
            return value if len(value)>=2 else None
        report['discovered_shards']=wait(read_offers,90);checks['signed_dht_shard_discovery']=True
        for role in ['worker0','worker1']:
            assert int(execpy(names[role],"import sqlite3;print(sqlite3.connect('/state/mesh-discovery.sqlite').execute('SELECT count(*) FROM offers').fetchone()[0])").stdout.strip())>=1
        checks['worker_dht_records_persist_in_sqlite']=True
        def generate(prompt="Explain peer-to-peer computing in one short sentence."):

            value=docker('exec',names['client'],'sangama','--token-file','/state/token','generate','--model-dir','/model','--device','cpu','--peers','127.0.0.1:7901,127.0.0.1:7902','--prompt',prompt,'--max-tokens','20',timeout=180)
            return json.loads(value.stdout)
        baseline=generate();report['generation']=baseline
        reference=json.loads((ROOT/'docs/test-results/qwen-metal-final.json').read_text())
        assert baseline['distributed_token_ids']==reference['local_token_ids'];checks['real_qwen_matches_independent_baseline']=True
        # netem runs in a short-lived helper sharing only the worker network namespace.
        # Inference containers retain cap-drop=ALL and a non-root UID.
        for role in ['worker0','worker1']:
            docker('run','--rm','--network','container:'+names[role],'--cap-drop=ALL','--cap-add=NET_ADMIN','sangama-netem:test','qdisc','replace','dev','eth0','root','netem','delay','25ms','5ms','rate','20mbit')
        impaired=generate();report['impaired_generation']=impaired
        assert impaired['distributed_token_ids']==baseline['distributed_token_ids'];checks['delay_jitter_bandwidth_preserve_tokens']=True
        report['impairment']={'worker_egress_delay_ms':25,'jitter_ms':5,'rate_mbit':20,'packet_loss_percent':0}
        for role in ['worker0','worker1']:
            docker('run','--rm','--network','container:'+names[role],'--cap-drop=ALL','--cap-add=NET_ADMIN','sangama-netem:test','qdisc','del','dev','eth0','root')
        # The OpenCode-facing interface uses these same aliases; exercise its real chat API.
        execpy(names['client'],"import subprocess;subprocess.Popen(['sangama','--token-file','/state/token','chat-api','--model-dir','/model','--device','cpu','--peers','127.0.0.1:7901,127.0.0.1:7902','--api-token-file','/state/api-token'],stdout=open('/state/chat.log','w'),stderr=subprocess.STDOUT)")
        chatcode="import json,urllib.request; r=urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions',data=json.dumps({'model':'sangama-qwen','messages':[{'role':'user','content':'What is 2 + 2? Answer with just the number.'}],'max_tokens':8}).encode(),headers={'Authorization':'Bearer '+open('/state/api-token').read(),'Content-Type':'application/json'}); print(urllib.request.urlopen(r,timeout=120).read().decode())"
        # Model name is read from gateway metadata below, avoiding a hard-coded alias.
        models=wait(lambda:execpy(names['client'],"import urllib.request;print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/v1/models',headers={'Authorization':'Bearer '+open('/state/api-token').read()}),timeout=2).read().decode())").stdout)
        model_id=json.loads(models)['data'][0]['id'];chatcode=chatcode.replace("'sangama-qwen'",repr(model_id))
        completion=json.loads(execpy(names['client'],chatcode).stdout);report['chat_completion']=completion
        assert '4' in completion['choices'][0]['message']['content'];checks['opencode_compatible_chat_api']=True
        # Run the actual pinned OpenCode CLI, with tools and all cloud providers disabled.
        oc_code="import os,pathlib,subprocess;env=os.environ.copy();base=pathlib.Path('/state/opencode');base.mkdir(exist_ok=True);\nfor key,folder in [('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('XDG_CACHE_HOME','cache'),('XDG_STATE_HOME','state')]:\n p=base/folder;p.mkdir(exist_ok=True);env[key]=str(p)\nenv['SANGAMA_API_KEY']=open('/state/api-token').read();env['OPENCODE_CONFIG_CONTENT']=open('/app/opencode.json').read();r=subprocess.run(['opencode','run','--format','json','--model','sangama/qwen2.5-0.5b-instruct','Explain briefly what this function does: def double(x): return x * 2'],cwd=base,env=env,capture_output=True,text=True,timeout=180);print(r.stdout);open('/state/opencode.log','w').write(r.stderr);raise SystemExit(r.returncode)"
        oc_output=execpy(names['client'],oc_code).stdout
        events=[json.loads(line) for line in oc_output.splitlines() if line.startswith('{')]
        text=''.join(e.get('part',{}).get('text','') for e in events if e.get('type')=='text')
        assert text.strip(), 'OpenCode returned no model text'
        report['opencode']={'version':'1.18.32','response':text,'provider':'sangama','tools_enabled':False,'cloud_fallback':False}
        checks['actual_opencode_cli_uses_relay_workers']=True
        # Interrupt an active generation, rather than only probing an offline node.
        active=subprocess.Popen(['docker','exec',names['client'],'sangama','--token-file','/state/token','generate','--model-dir','/model','--device','cpu','--peers','127.0.0.1:7901,127.0.0.1:7902','--prompt','Count from 1 to 100, writing every number in order.','--max-tokens','128'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def busy():
            value=execpy(names['worker0'],"import urllib.request;print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7900/v1/qwen/info',headers={'Authorization':'Bearer '+open('/state/token').read()}),timeout=2).read().decode())").stdout
            return json.loads(value)['busy']
        wait(busy,15);time.sleep(.5)
        docker('stop','--time','1',names['worker1'])
        active.communicate(timeout=90)
        assert active.returncode!=0;checks['mid_generation_disconnect_fails_closed']=True
        # Restart with persistent identity, reload the shard, and start a fresh session.
        docker('start',names['worker1'])
        execpy(names['worker1'],"import subprocess;subprocess.Popen(['python3','/app/mesh-node.py','--binary','/usr/local/bin/sangama','--config','/state/config.json','--model-dir','/model','--device','cpu'],stdout=open('/state/mesh.log','a'),stderr=subprocess.STDOUT)")
        wait(lambda:probe(7902,'/v1/node/capacity'),120)
        docker('exec',names['client'],'sangama','--token-file','/state/token','mesh-allocate','--model-dir','/model','--candidates','127.0.0.1:7901,127.0.0.1:7902',timeout=120)
        wait(lambda:probe(7902),120)
        recovered=generate();assert recovered['distributed_token_ids']==baseline['distributed_token_ids'];checks['restart_recovers_new_session']=True
        # Bound abuse by an already-admitted identity (not merely by its IP address).
        quota_code="import urllib.request,urllib.error; denied=0\nfor i in range(1250):\n r=urllib.request.Request('http://127.0.0.1:7902/v1/qwen/info',headers={'Authorization':'Bearer '+open('/state/token').read()})\n try: urllib.request.urlopen(r,timeout=5).read()\n except urllib.error.HTTPError as e:\n  if e.code==429: denied+=1;break\nprint(denied)"
        assert int(execpy(names['client'],quota_code).stdout.strip())==1;checks['admitted_peer_request_quota']=True
        oversized="import urllib.request,urllib.error;r=urllib.request.Request('http://127.0.0.1:7901/v1/qwen/forward',data=b'x'*(4*1024*1024+1),headers={'Authorization':'Bearer '+open('/state/token').read()});\ntry: print(urllib.request.urlopen(r,timeout=5).status)\nexcept urllib.error.HTTPError as e: print(e.code)"
        assert execpy(names['client'],oversized).stdout.strip()=='413';checks['oversized_frames_rejected']=True
        # Revoke the head worker: already-established connections must stop working.
        docker('exec',portal,'sangama-portal','revoke',ids['worker0'])
        def rejected():
            try:return not probe(7901)
            except subprocess.CalledProcessError:return True
        wait(rejected,20);checks['revocation_blocks_established_peer']=True
        # A stale authority cannot leave peers authorized indefinitely.
        docker('stop','--time','1',portal);time.sleep(17)
        log=execpy(names['relay'],"print(open('/state/mesh.log').read())").stdout
        assert 'membership expired or revoked' in log;checks['authority_outage_expires_membership']=True
        # Collect evidence that this was relayed, not a same-host direct TCP shortcut.
        for role in ['relay','worker0','client']:
            log=execpy(names[role],"print(open('/state/mesh.log').read())").stdout
            (out/(role+'-mesh.log')).write_text(log)
        assert 'relayed=true' in re.sub(r'\x1b\[[0-9;]*m','',(out/'client-mesh.log').read_text());checks['encrypted_relay_path_observed']=True
        report['passed']=True
    finally:
        report['checks']=checks;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for c in containers:
            logs=subprocess.run(['docker','logs',c],capture_output=True,text=True)
            (out/(c.removeprefix(prefix+'-')+'.log')).write_text(logs.stdout+logs.stderr)
            if c in list(locals().get('names',{}).values()):
                subprocess.run(['docker','cp',c+':/state/mesh.log',str(out/(c.removeprefix(prefix+'-')+'-mesh.log'))],capture_output=True)
            subprocess.run(['docker','rm','-f',c],capture_output=True)
        for v in volumes:subprocess.run(['docker','volume','rm',v],capture_output=True)
        for n in reversed(networks):subprocess.run(['docker','network','rm',n],capture_output=True)
        # Invitation and identity material never belongs in retained test reports.
        import shutil
        shutil.rmtree(private)
    print(json.dumps({'passed':True,'checks':checks,'report':str(out/'report.json')},indent=2))
if __name__=='__main__':main()
