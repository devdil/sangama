#!/usr/bin/env python3
"""Run two-worker Qwen routes from a scenario file and record tokens and timings.

Each scenario starts shard 0 and shard 1 as local workers (any engine, device or GPU), runs
`generate` through them, and compares the tokens with the expected reference. A scenario may
instead expect the route to be refused. Remote workers are reached through SSH tunnels that the
caller sets up; list them as "remote" with the loopback port the tunnel exposes.

Scenario file (JSON list):
  {"id": "A6", "name": "Candle CUDA x2", "expect": "f32" | "q4" | "refuse",
   "workers": [{"args": ["--device", "cuda"], "env": {"CUDA_VISIBLE_DEVICES": "0"}, "binary": "..."},
               {"remote": true}]}
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = {
    # Candle Metal, F32, greedy; see docs/test-results/scenario-matrix.md.
    'f32': [30888, 4686, 78597, 24231, 6147, 3847, 311, 4564, 323, 4332, 4963, 2041, 279, 1184, 369,
            264, 8622, 3538, 13, 151645],
    # llama.cpp Metal, Q4_K_M, greedy.
    'q4': [30888, 4686, 78597, 24231, 374, 264, 47963, 11, 3922, 291, 24231, 1614, 1380, 18495, 323,
           7611, 4564, 323, 4332, 4963, 2041, 264, 8622, 11198, 13, 151645],
}


def ready(port, token, deadline, procs):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/qwen/info',
                                     headers={'Authorization': 'Bearer ' + token})
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in procs):
            return None
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return json.load(response)
        except OSError:
            time.sleep(0.5)
    return None


def run(scenario, args, token):
    ports = [args.port, args.port + 1]
    procs, logs = [], []
    started = time.monotonic()
    try:
        for index, worker in enumerate(scenario['workers']):
            if worker.get('remote'):
                continue
            log = tempfile.NamedTemporaryFile('w+', prefix=f"{scenario['id']}-w{index}-", suffix='.log', delete=False)
            logs.append(log.name)
            cmd = [worker.get('binary', args.binary), '--token-file', args.token_file, 'qwen-worker',
                   '--model-dir', str(args.model_dir), '--shard', str(index),
                   '--listen', f'127.0.0.1:{ports[index]}', *worker.get('args', [])]
            if index == 0:
                cmd += ['--allow-next', f'127.0.0.1:{ports[1]}']
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, **worker.get('env', {})}))
        deadline = time.monotonic() + args.load_timeout
        infos = []
        for index, port in enumerate(ports):
            info = ready(port, token, deadline, procs)
            if info is None:
                errors = [Path(p).read_text()[-400:] for p in logs]
                return {'status': 'Fail', 'error': 'worker did not start', 'logs': errors}
            infos.append(info)
        load_s = time.monotonic() - started
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as out:
            report_path = out.name
        result = subprocess.run([args.binary, '--token-file', args.token_file, 'generate',
                                 '--model-dir', str(args.model_dir),
                                 '--peers', f'127.0.0.1:{ports[0]},127.0.0.1:{ports[1]}',
                                 '--max-tokens', '40', '--output', report_path],
                                capture_output=True, text=True, timeout=args.generate_timeout)
        route = [{k: i.get(k) for k in ('engine', 'device', 'precision')} for i in infos]
        if scenario['expect'] == 'refuse':
            refused = result.returncode != 0
            return {'status': 'Pass' if refused else 'Fail', 'route': route,
                    'refusal': (result.stderr.strip().splitlines() or [''])[-1]}
        if result.returncode != 0:
            return {'status': 'Fail', 'route': route, 'error': result.stderr.strip()[-600:]}
        report = json.loads(Path(report_path).read_text())
        tokens = report['distributed_token_ids']
        reference = REFERENCES[scenario['expect']]
        return {
            'status': 'Pass' if tokens == reference else 'Fail',
            'route': route,
            'tokens_match_reference': tokens == reference,
            'tokens': tokens,
            'text': report['distributed_text'],
            'load_s': round(load_s, 1),
            'first_token_ms': round(report['distributed']['first_token_ms'], 1),
            'decode_tokens_per_second': round(report['distributed']['decode_tokens_per_second'], 1),
            'forward_ms': [round(t['forward_ms'], 2) for t in report['last_trace']],
            'gpu_memory_budget_mib': [((i.get('memory') or {}).get('budget_bytes') or 0) // 2**20 for i in infos],
        }
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenarios', type=Path)
    parser.add_argument('--binary', default=str(ROOT/'target/release/sangama'))
    parser.add_argument('--model-dir', type=Path, default=ROOT/'.models/qwen2.5-0.5b-instruct')
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--port', type=int, default=7961, help='shard 0 port; shard 1 uses port+1')
    parser.add_argument('--only', help='comma-separated scenario ids')
    parser.add_argument('--load-timeout', type=float, default=240)
    parser.add_argument('--generate-timeout', type=float, default=300)
    parser.add_argument('--output', type=Path, help='append results as JSON lines')
    parser.add_argument('--environment', default='unspecified', help='environment id from the scenario matrix, e.g. E2')
    args = parser.parse_args()
    token = Path(args.token_file).read_text().strip()
    only = set(args.only.split(',')) if args.only else None
    for scenario in json.loads(args.scenarios.read_text()):
        if only and scenario['id'] not in only:
            continue
        result = {'id': scenario['id'], 'name': scenario['name'], 'environment': args.environment,
                  'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **run(scenario, args, token)}
        print(json.dumps(result), flush=True)
        if args.output:
            with args.output.open('a') as f:
                f.write(json.dumps(result) + '\n')
        if result['status'] != 'Pass':
            print(f"{scenario['id']}: {result['status']}", file=sys.stderr)


if __name__ == '__main__':
    main()
