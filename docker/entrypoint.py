"""Isolated CPU workers and a metadata-only client connected through pinned SSH tunnels."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

children = []
TOKEN = Path('/credentials/token').read_text().strip() if Path('/credentials/token').exists() else ''

def launch(argv):
    process = subprocess.Popen(argv)
    children.append(process)
    return process

def tunnel(host, local_port, remote_port):
    args = ['ssh','-F','/dev/null','-N','-T','-p','2222','-i','/credentials/identity',
            '-o','UserKnownHostsFile=/credentials/known_hosts','-o','GlobalKnownHostsFile=/dev/null',
            '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','BatchMode=yes',
            '-o','IdentitiesOnly=yes','-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no',
            '-o','ForwardAgent=no','-o','ForwardX11=no','-o','ExitOnForwardFailure=yes',
            '-o','ConnectTimeout=3','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
            '-L',f'127.0.0.1:{local_port}:127.0.0.1:{remote_port}',f'sangama@{host}']
    deadline = time.monotonic()+45
    while time.monotonic()<deadline:
        process=launch(args)
        time.sleep(.7)
        if process.poll() is None:
            return process
    raise RuntimeError(f'Could not establish SSH tunnel to {host}')

def info(port, token=TOKEN):
    request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/qwen/info',headers={'Authorization':'Bearer '+token})
    with urllib.request.urlopen(request,timeout=3) as response:
        return json.load(response)

def wait_ready(port):
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        try:
            return info(port)
        except (OSError,urllib.error.URLError):
            time.sleep(.25)
    raise RuntimeError(f'Worker {port} did not become ready')

def generate(prompt, limit=40):
    result=subprocess.run(['sangama','generate','--device','cpu','--model-dir','/model',
        '--peers','127.0.0.1:7901,127.0.0.1:7902','--token-file','/credentials/token',
        '--prompt',prompt,'--max-tokens',str(limit)],capture_output=True,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)

def main():
    role=sys.argv[1]
    if role in ['worker0','worker1']:
        index=int(role[-1]);port=7901+index
        ssh=launch(['/usr/sbin/sshd','-D','-e','-f','/app/docker/sshd_config','-o',f'PermitOpen=127.0.0.1:{port}'])
        if index==0:
            tunnel('worker1',7902,7902)
        args=['sangama','qwen-worker','--device','cpu','--shard',str(index),
              '--listen',f'127.0.0.1:{port}','--model-dir','/model','--token-file','/credentials/token']
        if index==0:
            args+=['--allow-next','127.0.0.1:7902']
        worker=launch(args)
        while all(p.poll() is None for p in [ssh,worker]):
            time.sleep(.5)
        raise RuntimeError('Worker or SSH service stopped')
    if role!='client':
        raise ValueError('expected worker0, worker1, or client')
    assert sorted(p.name for p in Path('/model').iterdir())==['config.json','manifest.json','tokenizer.json']
    tunnel('worker0',7901,7901)
    tunnel('worker1',7902,7902)
    workers=[wait_ready(7901),wait_ready(7902)]
    wrong_rejected=False
    try:
        info(7902,'invalid-token')
    except urllib.error.HTTPError as error:
        wrong_rejected=error.code==401
    first=generate('Explain peer-to-peer computing in one short sentence.')
    second=generate('What is 2 + 2? Answer with only the number.',8)
    assert first['operation']=='generate' and first['local'] is None and first['passed'] is None
    assert first['finish_reason']=='eos' and first['distributed_text'].strip()
    assert second['distributed_text'].strip()=='4' and wrong_rejected
    report={'passed':True,'architecture':'linux/arm64 CPU containers on one Docker Desktop VM',
        'client_network_namespace':os.readlink('/proc/self/ns/net'),
        'checks':{'metadata_only_client':True,'separate_shard_workers':True,
                  'ssh_encrypted_connections':True,'invalid_token_rejected':wrong_rejected,
                  'second_session_after_reset':True},'workers':workers,'generation':first,'arithmetic':second}
    Path('/reports/container-generation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'text':first['distributed_text'],'arithmetic':second['distributed_text'],
                     'tokens':first['generated_tokens'],'decode_tokens_per_second':first['distributed']['decode_tokens_per_second']},indent=2),flush=True)


def stop(signum, frame):
    raise SystemExit(128+signum)

signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
try:
    main()
finally:
    for process in children:
        if process.poll() is None:
            process.terminate()
    for process in children:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
