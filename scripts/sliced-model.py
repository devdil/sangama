#!/usr/bin/env python3
"""Prepare a model published as per-layer GGUF slices (scripts/split-gguf.py) for Sangama.

  manifest  On the operator's machine: pin the slices repo at an exact revision, choose the
            stage layout, fetch the tokenizer, and write manifest.json + tokenizer.json. Give the
            same directory to the client and to every worker.
  stage     On each worker: download only that stage's slices, verify every SHA-256 against the
            pinned model.json, assemble the stage GGUF, and approve it in gguf.json for
            `qwen-worker --engine llamacpp --gguf <file>`.

A shard's sha256 in the manifest is the digest of its slices' names and checksums, so a stage is
pinned to exact slice files. Downloads are public HTTPS from Hugging Face and resume if cut off.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(repo, revision, name):
    return f'https://huggingface.co/{repo}/resolve/{revision}/{name}'


def fetch_bytes(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def download(url, dest, size=None):
    """Download with resume; returns the file path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
    with urllib.request.urlopen(request, timeout=60) as r, open(part, 'ab' if have else 'wb') as f:
        if have and r.status != 206:
            f.truncate(0)
        while block := r.read(1 << 24):
            f.write(block)
    os.replace(part, dest)
    return dest


def stage_slices(model, start, end):
    """Slice names a stage [start, end) needs, in order."""
    total = model['layers_present']
    names = ['embed'] if start == 0 else []
    names += [f'layer-{i:03d}' for i in range(start, end)]
    if end == total:
        names += [s['name'] for s in model['slices'] if s['name'] in ('head', 'extra')]
        if not model.get('separate_output', True) and 'embed' not in names:
            names.insert(0, 'embed')
    return names


def stage_digest(model, names):
    by_name = {s['name']: s for s in model['slices']}
    text = ''.join(f"{n}:{by_name[n]['sha256']}\n" for n in names)
    return hashlib.sha256(text.encode()).hexdigest()


def manifest(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_bytes = fetch_bytes(resolve(args.repo, args.revision, 'model.json'))
    model = json.loads(model_bytes)
    assert model['format'] == 'sangama-gguf-slices-v1', 'not a split-gguf.py manifest'
    layers = model['layers_present']
    bounds = [int(b) for b in args.boundaries.split(',')] if args.boundaries else \
        [round(layers * i / args.stages) for i in range(1, args.stages)]
    edges = [0] + bounds + [layers]
    assert edges == sorted(set(edges)), 'stage boundaries must be increasing and inside the model'
    by_name = {s['name']: s for s in model['slices']}
    shards = []
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        names = stage_slices(model, start, end)
        shards.append({'index': index, 'start': start, 'end': end,
                       'file': f'stage-{index}-layers-{start}-{end}.gguf',
                       'sha256': stage_digest(model, names),
                       'file_bytes': sum(by_name[n]['bytes'] for n in names),
                       'tensor_count': sum(by_name[n]['tensors'] for n in names)})
    tokenizer = fetch_bytes(resolve(args.tokenizer_repo, args.tokenizer_revision, 'tokenizer.json'))
    (out/'tokenizer.json').write_bytes(tokenizer)
    config = json.loads(fetch_bytes(resolve(args.tokenizer_repo, args.tokenizer_revision, 'config.json')))
    text = config.get('text_config', config)
    document = {
        'model_id': model['model_id'],
        'revision': args.revision,
        'weights_sha256': hashlib.sha256(model_bytes).hexdigest(),
        'config_sha256': '',
        'tokenizer_sha256': hashlib.sha256(tokenizer).hexdigest(),
        'shards': shards,
        'sliced': {
            'architecture': model['architecture'],
            'layers': layers,
            'hidden_size': int(text['hidden_size']),
            'vocab_size': int(text['vocab_size']),
            'eos_tokens': args.eos,
            'assistant_prefix': args.assistant_prefix,
            'slices_repo': args.repo,
            'slices_revision': args.revision,
            'slices_manifest_sha256': hashlib.sha256(model_bytes).hexdigest(),
        },
    }
    (out/'manifest.json').write_text(json.dumps(document, indent=2) + '\n')
    for s in shards:
        print(f"stage {s['index']}: layers {s['start']}-{s['end']}, {s['file_bytes'] / 1e9:.1f} GB")
    print(f'wrote {out}/manifest.json and tokenizer.json')


def stage(args):
    model_dir = Path(args.model_dir)
    document = json.loads((model_dir/'manifest.json').read_text())
    manifest_hash = hashlib.sha256((model_dir/'manifest.json').read_bytes()).hexdigest()
    sliced = document['sliced']
    shard = document['shards'][args.shard]
    repo, revision = sliced['slices_repo'], sliced['slices_revision']
    slices_dir = model_dir/'slices'
    model_path = download(resolve(repo, revision, 'model.json'), slices_dir/'model.json')
    if sha256_file(model_path) != sliced['slices_manifest_sha256']:
        sys.exit('model.json does not match the pinned manifest')
    model = json.loads(model_path.read_text())
    names = stage_slices(model, shard['start'], shard['end'])
    if stage_digest(model, names) != shard['sha256']:
        sys.exit('stage slices do not match the manifest')
    by_name = {s['name']: s for s in model['slices']}
    for name in names:
        entry = by_name[name]
        path = download(resolve(repo, revision, entry['file']), slices_dir/entry['file'], entry['bytes'])
        if sha256_file(path) != entry['sha256']:
            path.unlink()
            sys.exit(f"{entry['file']} failed its checksum; removed, rerun to download again")
        print(f"verified {entry['file']}", flush=True)
    out = model_dir/shard['file']
    env = dict(os.environ, PYTHONPATH=str(Path(args.gguf_py)))
    subprocess.run([sys.executable, str(ROOT/'scripts/split-gguf.py'), 'assemble', str(model_path),
                    '--start', str(shard['start']), '--end', str(shard['end']), '--out', str(out),
                    '--skip-verify'], check=True, env=env)
    index_path = model_dir/'gguf.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    files = index.get('files', []) if index.get('model_hash') == manifest_hash else []
    files = [f for f in files if f['file'] != out.name]
    files.append({'file': out.name, 'sha256': sha256_file(out),
                  'precision': model['quantization'].lower(), 'file_bytes': out.stat().st_size})
    index_path.write_text(json.dumps({'model_hash': manifest_hash, 'files': files}, indent=2) + '\n')
    if not args.keep_slices:
        for name in names:
            (slices_dir/by_name[name]['file']).unlink(missing_ok=True)
    print(f"stage {args.shard} ready: {out.name}; run qwen-worker --shard {args.shard} --engine llamacpp --gguf {out.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest')
    m.add_argument('--repo', required=True, help='slices repo, e.g. diljitpr/Qwen3.5-397B-A17B-Q4_K_M-slices')
    m.add_argument('--revision', required=True, help='exact commit of the slices repo')
    m.add_argument('--tokenizer-repo', required=True, help='original model repo, e.g. Qwen/Qwen3.5-397B-A17B')
    m.add_argument('--tokenizer-revision', required=True, help='exact commit of the original model repo')
    m.add_argument('--stages', type=int, default=3)
    m.add_argument('--boundaries', help='explicit first layers of stages 2.., e.g. 20,40')
    m.add_argument('--eos', nargs='+', default=['<|im_end|>', '<|endoftext|>'])
    m.add_argument('--assistant-prefix', default='<think>\n\n</think>\n\n',
                   help='text after the assistant header (default turns off Qwen3.5 thinking)')
    m.add_argument('--out', required=True)
    m.set_defaults(func=manifest)
    s = sub.add_parser('stage')
    s.add_argument('--model-dir', required=True, help='directory holding manifest.json')
    s.add_argument('--shard', type=int, required=True)
    s.add_argument('--gguf-py', default=str(ROOT/'.tools/llama.cpp/gguf-py'))
    s.add_argument('--keep-slices', action='store_true', help='keep downloaded slices after assembly')
    s.set_defaults(func=stage)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
