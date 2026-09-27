#!/usr/bin/env python3
"""Run real CPU inference in three isolated containers. Docker Desktop/Engine must be running."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid


def docker(*args, **kwargs):
    return subprocess.run(['docker',*args],check=True,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--image',default='sangama:container-test')
    parser.add_argument('--model-dir',default='.models/qwen2.5-0.5b-instruct')
    parser.add_argument('--output-dir',default='runs/docker-test')
    args=parser.parse_args()
    docker('info',stdout=subprocess.DEVNULL)
    if args.build:
        docker('build','-f','docker/Dockerfile','-t',args.image,'.')
    model=Path(args.model_dir).resolve()
    manifest=json.loads((model/'manifest.json').read_text())
    if len(manifest['shards'])!=2:
        raise ValueError('Prepare the default two-shard checkpoint first.')
    for name in ['config.json','tokenizer.json','manifest.json',*[s['file'] for s in manifest['shards']]]:
        if not (model/name).is_file():
            raise ValueError(f'Missing required file: {name}')
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    prefix='sangama-test-'+uuid.uuid4().hex[:10]
    network=prefix+'-net'
    volumes={role:prefix+'-'+role+'-data' for role in ['worker0','worker1','client','reports']}
    containers=[]
    created_volumes=[]
    created_network=False
    def launch(role,files,memory):
        name=prefix+'-'+role
        command=['run','-d','--name',name,'--hostname',role,'--network',network,'--network-alias',role,
                 '--label','sangama.test=true','--read-only','--cap-drop=ALL','--security-opt','no-new-privileges=true',
                 '--pids-limit','128','--memory',memory,'--cpus','2',
                 '--tmpfs','/tmp:rw,nosuid,noexec,size=64m,mode=1777',
                 '--mount',f'type=volume,src={volumes[role]},dst=/credentials,readonly']
        for file in files:
            command+=['--mount',f'type=bind,src={model/file},dst=/model/{file},readonly']
        if role=='client':
            command+=['--mount',f'type=volume,src={volumes["reports"]},dst=/reports']
        command+=[args.image,role]
        docker(*command,stdout=subprocess.DEVNULL)
        containers.append(name)
        return name
    try:
        docker('network','create','--internal','--label','sangama.test=true',network,stdout=subprocess.DEVNULL)
        created_network=True
        for volume in volumes.values():
            docker('volume','create','--label','sangama.test=true',volume,stdout=subprocess.DEVNULL)
            created_volumes.append(volume)
        init=['run','--rm','--network','none','--user','0:0','--entrypoint','python3']
        for role,volume in volumes.items():
            init+=['--mount',f'type=volume,src={volume},dst=/init/{role}']
        docker(*init,args.image,'/app/docker/prepare.py')
        shared=['config.json','manifest.json']
        launch('worker1',shared+[manifest['shards'][1]['file']],'3g')
        launch('worker0',shared+[manifest['shards'][0]['file']],'3g')
        client=launch('client',shared+['tokenizer.json'],'768m')
        waited=docker('wait',client,capture_output=True,text=True,timeout=240)
        if waited.stdout.strip()!='0':
            raise RuntimeError('Client failed; inspect the container logs in the output directory.')
        docker('cp',f'{client}:/reports/container-generation.json',str(output/'container-generation.json'),stdout=subprocess.DEVNULL)
        inspection=json.loads(docker('inspect',*containers,capture_output=True,text=True).stdout)
        report=json.loads((output/'container-generation.json').read_text())
        evidence=[]
        for item in inspection:
            host=item['HostConfig']
            assert item['Config']['User']=='10001:10001'
            assert host['ReadonlyRootfs'] and not host.get('PortBindings')
            assert 'ALL' in host['CapDrop'] and 'no-new-privileges=true' in host['SecurityOpt']
            files=sorted(m['Destination'].split('/')[-1] for m in item['Mounts'] if m['Destination'].startswith('/model/'))
            role=item['Config']['Hostname']
            expected=shared+(['tokenizer.json'] if role=='client' else [manifest['shards'][int(role[-1])]['file']])
            assert files==sorted(expected)
            assert all(not m['RW'] for m in item['Mounts'] if m['Destination'].startswith('/model/') or m['Destination']=='/credentials')
            namespace=report['client_network_namespace'] if role=='client' else docker('exec',item['Id'],'readlink','/proc/self/ns/net',capture_output=True,text=True).stdout.strip()
            assert namespace.startswith('net:[')
            evidence.append({'role':role,'container_id':item['Id'],'network_namespace':namespace,
                'model_files':files,'uid':item['Config']['User'],'read_only_root':True,'host_ports_published':False,
                'capabilities_dropped':host['CapDrop'],'no_new_privileges':True,'memory_limit':host['Memory'],
                'oom_killed':item['State']['OOMKilled']})
        assert len({x['network_namespace'] for x in evidence})==3
        report['container_isolation']=evidence
        report['image_id']=inspection[0]['Image']
        # Compare with an earlier independently verified real-model run when available.
        reference=Path('docs/test-results/qwen-metal-final.json')
        if reference.exists():
            expected=json.loads(reference.read_text())
            assert report['generation']['distributed_token_ids']==expected['local_token_ids']
            report['checks']['tokens_match_recorded_independent_baseline']=True
        (output/'container-generation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'containers':len(evidence),'text':report['generation']['distributed_text'],
                          'arithmetic':report['arithmetic']['distributed_text'],'report':str(output/'container-generation.json')},indent=2))
    finally:
        for container in containers:
            role=container.removeprefix(prefix+'-')
            logs=subprocess.run(['docker','logs',container],capture_output=True,text=True)
            (output/(role+'.log')).write_text(logs.stdout+logs.stderr)
            subprocess.run(['docker','rm','-f',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if created_network:
            subprocess.run(['docker','network','rm',network],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for volume in created_volumes:
            subprocess.run(['docker','volume','rm',volume],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__':
    main()
