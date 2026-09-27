#!/usr/bin/env python3
"""Verify Kademlia discovery across three Docker network namespaces; no model files needed."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid


def docker(*args,**kwargs):
    return subprocess.run(['docker',*args],check=True,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='sangama:container-test')
    parser.add_argument('--output-dir',default='runs/docker-dht-test')
    args=parser.parse_args()
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    prefix='sangama-dht-'+uuid.uuid4().hex[:10]
    network=prefix+'-net';containers=[]
    docker('network','create','--internal','--label','sangama.test=true',network,stdout=subprocess.DEVNULL)
    def launch(role,bootstrap=None):
        name=prefix+'-'+role
        argv=['run','-d','--name',name,'--hostname',role,'--network',network,'--label','sangama.test=true',
              '--read-only','--cap-drop=ALL','--security-opt','no-new-privileges=true','--pids-limit','64',
              '--memory','256m','--cpus','1','--tmpfs','/tmp:rw,noexec,nosuid,size=32m,mode=1777',
              '--entrypoint','python3',args.image,'/app/docker/dht-node.py',role]
        if bootstrap:argv+=[bootstrap]
        docker(*argv,stdout=subprocess.DEVNULL);containers.append(name)
        for _ in range(100):
            log=docker('logs',name,capture_output=True,text=True)
            if len(log.stdout.splitlines())>=2:
                metadata=json.loads(log.stdout.splitlines()[0])
                return name,{**json.loads(log.stdout.splitlines()[1]),**metadata}
            time.sleep(.1)
        raise RuntimeError(f'{role} failed to start')
    try:
        seed,s=launch('seed')
        provider,p=launch('provider',s['address'])
        seeker,q=launch('seeker',s['address'])
        waited=docker('wait',seeker,capture_output=True,text=True,timeout=35)
        if waited.stdout.strip()!='0':raise RuntimeError('DHT seeker failed; inspect saved logs')
        log=docker('logs',seeker,capture_output=True,text=True).stdout
        found=json.loads(log.split('\n',2)[2])
        assert any(r['peer_id']==p['peer_id'] and r['model_hash']=='a'*64 for r in found['discoveries'])
        inspect=json.loads(docker('inspect',*containers,capture_output=True,text=True).stdout)
        metadata={'seed':s,'provider':p,'seeker':q}
        assert len({entry['network_namespace'] for entry in metadata.values()})==3
        evidence=[{'role':item['Config']['Hostname'],'container_id':item['Id'],
                   'network_namespace':metadata[item['Config']['Hostname']]['network_namespace'],'ip':metadata[item['Config']['Hostname']]['ip'],
                   'uid':item['Config']['User'],'read_only_root':item['HostConfig']['ReadonlyRootfs']} for item in inspect]
        report={'passed':True,'topology':'three containers on one Docker Desktop VM; seeker knows only the seed',
                'seed':s['peer_id'],'provider':p['peer_id'],'seeker':q['peer_id'],
                'discoveries':found['discoveries'],'container_isolation':evidence,'image_id':inspect[0]['Image'],
                'note':'The all-a model hash is synthetic discovery metadata. No inference weights are loaded in this test.'}
        (output/'dht-containers.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'providers_found':len(found['discoveries']),'report':str(output/'dht-containers.json')},indent=2))
    finally:
        for name in containers:
            log=subprocess.run(['docker','logs',name],capture_output=True,text=True)
            (output/(name.removeprefix(prefix+'-')+'.log')).write_text(log.stdout+log.stderr)
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run(['docker','network','rm',network],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__':
    main()
