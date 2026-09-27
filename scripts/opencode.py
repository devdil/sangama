#!/usr/bin/env python3
"""Run isolated OpenCode settings against a local Sangama gateway and shard workers."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def private_token(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(secrets.token_hex(32)+'\n')
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise RuntimeError(f'Token file must be private and not a symlink: {path}')
    return path.read_text().strip()

def environment(api_key):
    env = os.environ.copy()
    env.pop("P2P_TOKEN", None)
    env.pop("P2P_TOKEN_FILE", None)
    state = ROOT/'.mesh/opencode'
    for name, folder in [('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('XDG_CACHE_HOME','cache'),('XDG_STATE_HOME','state')]:
        directory = state/folder
        directory.mkdir(parents=True,exist_ok=True)
        env[name] = str(directory)
    env['SANGAMA_API_KEY'] = api_key
    env['OPENCODE_CONFIG_CONTENT'] = (ROOT/'integrations/opencode/opencode.json').read_text()
    return env

def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['metal','cpu'],default='metal')
    parser.add_argument('--peers',help='Existing worker endpoints in shard order (admitted mesh bridges or SSH tunnels for remote peers)')
    parser.add_argument('--worker-token-file',type=Path)
    parser.add_argument('--serve-only',action='store_true',help='Keep gateway/workers alive for API tests')
    parser.add_argument('opencode_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    binary=ROOT/'target/release/sangama'
    cli=ROOT/'.tools/opencode/node_modules/.bin/opencode'
    if not binary.exists(): raise RuntimeError('Build first: ./scripts/cargo build --release --features metal --locked')
    if not args.serve_only and not cli.exists(): raise RuntimeError('Install first: ./scripts/install-opencode.sh')
    if args.peers and not args.worker_token_file: raise RuntimeError('--peers requires --worker-token-file')
    api_path=ROOT/'.secrets/opencode/api.token'
    api_key=private_token(api_path)
    worker_path=(args.worker_token_file.resolve() if args.worker_token_file else ROOT/'.secrets/opencode/worker.token')
    if args.worker_token_file:
        if not worker_path.is_file(): raise RuntimeError('Existing worker token file is missing')
    else: private_token(worker_path)
    children=[]; logs=[]
    def launch(argv,name):
        path=ROOT/'runs/opencode'/f'{name}.log'
        path.parent.mkdir(parents=True,exist_ok=True)
        log=path.open('w');logs.append(log)
        child=subprocess.Popen([str(binary),*argv],cwd=ROOT,stdout=log,stderr=log)
        children.append(child)
        return child
    def ready(url,token,timeout=120):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if any(c.poll() is not None for c in children): raise RuntimeError('Sangama service exited; inspect runs/opencode/*.log')
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer '+token}),timeout=2) as response:
                    return json.load(response)
            except OSError: time.sleep(.2)
        raise RuntimeError('Timed out waiting for Sangama')
    try:
        if args.peers:
            peers=args.peers
        else:
            first,second=free_port(),free_port()
            while first==second: second=free_port()
            peers=f'127.0.0.1:{first},127.0.0.1:{second}'
            for index,port in [(1,second),(0,first)]:
                command=['qwen-worker','--shard',str(index),'--device',args.device,'--listen',f'127.0.0.1:{port}','--token-file',str(worker_path)]
                if index==0:command+=['--allow-next',f'127.0.0.1:{second}']
                launch(command,f'worker{index}')
            for port in [first,second]: ready(f'http://127.0.0.1:{port}/v1/qwen/info',worker_path.read_text().strip())
        launch(['chat-api','--device',args.device,'--peers',peers,'--token-file',str(worker_path),'--api-token-file',str(api_path)],'gateway')
        ready('http://127.0.0.1:8090/v1/models',api_key)
        print('Sangama ready at http://127.0.0.1:8090/v1 (text-only; tools disabled).',file=sys.stderr)
        if args.serve_only:
            while all(c.poll() is None for c in children): time.sleep(.5)
            raise RuntimeError('Sangama service exited')
        workspace=ROOT/'work/opencode-workspace'
        workspace.mkdir(parents=True,exist_ok=True)
        if not (workspace/'.git').exists():subprocess.run(['git','init','-q',str(workspace)],check=True)
        forwarded=args.opencode_args
        if forwarded[:1]==['--']:forwarded=forwarded[1:]
        process=subprocess.Popen([str(cli),*forwarded],cwd=workspace,env=environment(api_key))
        children.append(process)
        return process.wait()
    finally:
        for child in reversed(children):
            if child.poll() is None:child.terminate()
        for child in reversed(children):
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
