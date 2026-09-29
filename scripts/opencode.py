#!/usr/bin/env python3
"""Run isolated OpenCode settings against a local Sangama gateway and shard workers."""
import argparse
import json
import os
import platform
import tempfile
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

def environment(api_key, port):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("OPENCODE_"): del env[key]
    env.pop("P2P_TOKEN", None)
    env.pop("P2P_TOKEN_FILE", None)
    state = ROOT/'.mesh/opencode'
    for name, folder in [('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('XDG_CACHE_HOME','cache'),('XDG_STATE_HOME','state')]:
        directory = state/folder
        directory.mkdir(parents=True,exist_ok=True)
        env[name] = str(directory)
    home = state/'home'
    home.mkdir(parents=True, exist_ok=True)
    env['HOME'] = str(home)
    env['USERPROFILE'] = str(home)
    env['OPENCODE_DISABLE_PROJECT_CONFIG'] = 'true'
    env['OPENCODE_DISABLE_CLAUDE_CODE'] = 'true'
    env['SANGAMA_API_KEY'] = api_key
    config = json.loads((ROOT/'integrations/opencode/opencode.json').read_text())
    config['provider']['sangama']['options']['baseURL'] = f'http://127.0.0.1:{port}/v1'
    env['OPENCODE_CONFIG_CONTENT'] = json.dumps(config)
    return env

def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]

def main():
    defaults_path = ROOT/'settings.json'
    defaults = json.loads(defaults_path.read_text()) if defaults_path.exists() else {}
    parser=argparse.ArgumentParser(prog="sangama-code", description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=defaults.get('model_dir', ROOT/'.models/qwen2.5-0.5b-instruct'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--device',choices=['metal','cuda','cpu'],default=defaults.get('device', 'metal' if platform.system() == 'Darwin' and platform.machine() == 'arm64' else 'cpu'))
    parser.add_argument('--peers',help='Existing worker endpoints in shard order (admitted mesh bridges or SSH tunnels for remote peers)')
    parser.add_argument('--worker-token-file',type=Path)
    parser.add_argument('--serve-only',action='store_true',help='Keep gateway/workers alive for API tests')
    parser.add_argument('opencode_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    binary=ROOT/'target/release/sangama'
    cli=ROOT/'.tools/opencode/node_modules/opencode-ai/bin/opencode.exe'
    if not binary.exists(): raise RuntimeError('Build first: ./scripts/cargo build --release --features metal --locked')
    if not args.serve_only and not cli.exists(): raise RuntimeError('Install first: ./scripts/install-opencode.sh')
    if args.peers and not args.worker_token_file: raise RuntimeError('--peers requires --worker-token-file')
    args.model_dir = args.model_dir.expanduser().resolve()
    if not (args.model_dir/'config.json').is_file(): raise RuntimeError(f'Model missing at {args.model_dir}; prepare it with scripts/fetch-qwen.py or supply --model-dir')
    args.project = args.project.expanduser().resolve()
    if not args.project.is_dir(): raise RuntimeError('Project directory does not exist')
    run_root = ROOT/'runs/opencode'
    run_root.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix='session-', dir=run_root))
    api_path=ROOT/'.secrets/opencode/api.token' if args.serve_only else session/'api.token'
    api_key=private_token(api_path)
    worker_path=(args.worker_token_file.resolve() if args.worker_token_file else ROOT/'.secrets/opencode/worker.token')
    if args.worker_token_file:
        if not worker_path.is_file(): raise RuntimeError('Existing worker token file is missing')
    else: private_token(worker_path)
    children=[]; logs=[]
    def launch(argv,name):
        path=session/f'{name}.log'
        path.parent.mkdir(parents=True,exist_ok=True)
        log=path.open('w');logs.append(log)
        child=subprocess.Popen([str(binary),*argv],cwd=ROOT,stdout=log,stderr=log)
        children.append(child)
        return child
    def ready(url,token,timeout=120):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if any(c.poll() is not None for c in children): raise RuntimeError(f'Sangama service exited; inspect {session}/*.log')
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
                command=['qwen-worker','--model-dir',str(args.model_dir),'--shard',str(index),'--device',args.device,'--listen',f'127.0.0.1:{port}','--token-file',str(worker_path)]
                if index==0:command+=['--allow-next',f'127.0.0.1:{second}']
                launch(command,f'worker{index}')
            for port in [first,second]: ready(f'http://127.0.0.1:{port}/v1/qwen/info',worker_path.read_text().strip())
        port = 8090 if args.serve_only else free_port()
        launch(['chat-api','--listen',f'127.0.0.1:{port}','--model-dir',str(args.model_dir),'--device',args.device,'--peers',peers,'--token-file',str(worker_path),'--api-token-file',str(api_path)],'gateway')
        ready(f'http://127.0.0.1:{port}/v1/models',api_key)
        print(f'Sangama ready at http://127.0.0.1:{port}/v1 (text-only; tools disabled).',file=sys.stderr)
        if args.serve_only:
            while all(c.poll() is None for c in children): time.sleep(.5)
            raise RuntimeError('Sangama service exited')
        forwarded=args.opencode_args
        if forwarded[:1]==['--']:forwarded=forwarded[1:]
        process=subprocess.Popen([str(cli),*forwarded],cwd=args.project,env=environment(api_key, port))
        children.append(process)
        return process.wait()
    finally:
        for child in reversed(children):
            if child.poll() is None:child.terminate()
        for child in reversed(children):
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
        if not args.serve_only: api_path.unlink(missing_ok=True)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except (RuntimeError, OSError) as error:
        print(f"sangama-code: {error}", file=sys.stderr)
        sys.exit(1)
