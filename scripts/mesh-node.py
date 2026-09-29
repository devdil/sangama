#!/usr/bin/env python3
"""Start the configured local worker, admitted mesh, and optional OpenCode gateway."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--binary',type=Path,default=ROOT/'target/release/sangama')
    p.add_argument('--model-dir',type=Path,default=ROOT/'.models/qwen2.5-0.5b-instruct')
    p.add_argument('--shard',type=int)
    p.add_argument('--device',choices=['cpu','metal','cuda'],default='metal')
    p.add_argument('--chat',action='store_true')
    p.add_argument('--api-token-file',type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text());children=[]
    if a.chat and (not a.api_token_file or str(a.api_token_file.resolve())==str(Path(config['token_file']).resolve())):
        p.error('--chat requires a separate --api-token-file')
    base=[str(a.binary),'--token-file',config['token_file']]
    def spawn(cmd):
        child=subprocess.Popen(cmd,start_new_session=True);children.append(child);return child
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        if config.get('worker') and not config.get('managed'):
            if a.shard is None:p.error('worker requires --shard from the prepared manifest')
            cmd=base+['qwen-worker','--model-dir',str(a.model_dir),'--device',a.device,'--shard',str(a.shard),'--listen',config['worker']]
            if config.get('bridges'):cmd+=['--allow-next',','.join(b['listen'] for b in config['bridges'])]
            spawn(cmd)
        spawn([str(a.binary),'mesh','--config',str(a.config.resolve())])
        if a.chat:
            candidates=','.join(b['listen'] for b in config['bridges'])
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if any(c.poll() is not None for c in children):raise RuntimeError('A node process stopped; inspect its error above')
                probe=subprocess.run(base+['mesh-plan','--model-dir',str(a.model_dir),'--candidates',candidates],capture_output=True,text=True)
                if probe.returncode==0:break
                allocation=subprocess.run(base+['mesh-allocate','--model-dir',str(a.model_dir),'--candidates',candidates],capture_output=True,text=True)
                if allocation.returncode==0:print('Prepared shard assignment loaded; checking readiness.',flush=True)
                time.sleep(2)
            else:raise RuntimeError('No complete ready route after 120 seconds')
            plan=json.loads(probe.stdout)
            print('Selected ready shard bridges: '+', '.join(plan['peers']),flush=True)
            spawn(base+['chat-api','--model-dir',str(a.model_dir),'--device',a.device,'--peers',','.join(plan['peers']),'--api-token-file',str(a.api_token_file)])
        print('Sangama node running. Ctrl-C stops its local processes.',flush=True)
        while all(c.poll() is None for c in children):time.sleep(.5)
        raise RuntimeError('A node process stopped; restart after resolving the reported error')
    except KeyboardInterrupt:pass
    finally:
        for c in reversed(children):
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        for c in reversed(children):
            try:c.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
if __name__=='__main__':main()
