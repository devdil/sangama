#!/usr/bin/env python3
"""Opt-in real-checkpoint regression: metadata-only client and standalone local shards.

Requires a built Sangama binary and the prepared Qwen checkpoint. No new downloads.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/sangama')
    parser.add_argument('--model-dir', default='.models/qwen2.5-0.5b-instruct')
    parser.add_argument('--device', choices=['cpu', 'metal'], default='metal')
    parser.add_argument('--output', default='runs/standalone-generation-test.json')
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    model = Path(args.model_dir).resolve()
    env = os.environ.copy()
    env.pop('P2P_TOKEN_FILE', None)
    env['P2P_TOKEN'] = secrets.token_hex(32)
    prompt = 'Explain peer-to-peer computing in one short sentence.'
    workers = []
    with tempfile.TemporaryDirectory(prefix='sangama-generate-') as scratch:
        client = Path(scratch) / 'client'
        client.mkdir()
        for name in ['manifest.json', 'config.json', 'tokenizer.json']:
            shutil.copyfile(model / name, client / name)
        manifest = json.loads((model / 'manifest.json').read_text())
        assert len(manifest['shards']) == 2, 'This smoke test expects the default two-shard manifest.'
        sockets = [socket.socket() for _ in range(2)]
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        addresses = [f'127.0.0.1:{sock.getsockname()[1]}' for sock in sockets]
        def run(command, directory, peers=None):
            argv = [binary, command, '--model-dir', str(directory), '--device', args.device,
                    '--prompt', prompt, '--max-tokens', '40']
            if peers:
                argv += ['--peers', ','.join(peers)]
            result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr)
            return json.loads(result.stdout)
        try:
            for index in range(2):
                sockets[index].close()
                argv = [binary, 'qwen-worker', '--model-dir', str(model), '--shard', str(index),
                        '--device', args.device, '--listen', addresses[index]]
                if index == 0:
                    argv += ['--allow-next', addresses[1]]
                with open(Path(scratch) / f'worker-{index}.log', 'w') as log:
                    workers.append(subprocess.Popen(argv, env=env, stdout=log, stderr=log))
            for address in addresses:
                deadline = time.monotonic() + 120
                while True:
                    if any(worker.poll() is not None for worker in workers):
                        raise RuntimeError('A worker exited during startup.')
                    request = urllib.request.Request(f'http://{address}/v1/qwen/info',
                        headers={'Authorization': 'Bearer ' + env['P2P_TOKEN']})
                    try:
                        with urllib.request.urlopen(request, timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError('Worker startup timed out.')
                    time.sleep(.2)
            assert not list(client.glob('*.safetensors'))
            generated = run('generate', client, addresses)
            repeated = run('generate', client, addresses)
            assert generated['operation'] == 'generate' and generated['passed'] is None
            assert generated['local'] is None and generated['local_token_ids'] is None
            assert generated['finish_reason'] == 'eos'
            assert generated['distributed_token_ids'] == repeated['distributed_token_ids']
        finally:
            for sock in sockets:
                sock.close()
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers:
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
        # Autostart also works with physical shards but no original complete checkpoint.
        for shard in manifest['shards']:
            os.link(model / shard['file'], client / shard['file'])
        automatic = run('generate', client)
        assert automatic['distributed_token_ids'] == generated['distributed_token_ids']
        verified = run('qwen-test', model)
        assert verified['passed'] and verified['operation'] == 'verify'
        assert verified['local_token_ids'] == generated['distributed_token_ids']
        report = {'passed': True, 'topology': 'separate worker processes on one physical computer',
                  'checks': {'client_directory_contained_no_weights': True,
                             'generation_did_not_run_a_baseline': True,
                             'second_session_matches_after_reset': True,
                             'autostart_without_original_checkpoint': True,
                             'tokens_match_separate_verification': True},
                  'generation': generated, 'verification': verified}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'passed': True, 'generated_tokens': generated['generated_tokens'],
                          'text': generated['distributed_text'], 'report': str(output)}, indent=2))


if __name__ == '__main__':
    main()
