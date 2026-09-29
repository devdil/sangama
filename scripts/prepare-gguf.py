#!/usr/bin/env python3
"""Convert the pinned Qwen checkpoint to GGUF for the llama.cpp engine and approve it in gguf.json.

Needs the prepared checkpoint (scripts/fetch-qwen.py), the llama.cpp source
(scripts/fetch-llama-cpp.sh) and a Python with llama.cpp's converter requirements:
  pip install -r .tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
Quantizing also builds llama-quantize with CMake.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
# Output types the converter writes directly, and llama-quantize types the worker recognises.
OUTTYPES = ['f32', 'bf16', 'f16', 'q8_0']
QUANTS = ['Q4_0', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'.models/qwen2.5-0.5b-instruct')
    parser.add_argument('--llama-cpp', type=Path, default=ROOT/'.tools/llama.cpp')
    parser.add_argument('--outtype', choices=OUTTYPES, default='f32',
                        help='f32 keeps the exact values the Candle engine uses')
    parser.add_argument('--quantize', choices=QUANTS, help='also write a quantized copy, e.g. Q4_K_M')
    parser.add_argument('--python', default=sys.executable, help='interpreter with the converter requirements')
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    source = args.llama_cpp.resolve()
    if not (source/'.sangama-pin').is_file():
        sys.exit('llama.cpp source missing; run scripts/fetch-llama-cpp.sh')
    manifest = model_dir/'manifest.json'
    if not manifest.is_file():
        sys.exit('checkpoint missing; run python3 scripts/fetch-qwen.py')
    model_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    written = []
    base = model_dir/f'qwen2.5-0.5b-instruct-{args.outtype}.gguf'
    subprocess.run([args.python, str(source/'convert_hf_to_gguf.py'), str(model_dir),
                    '--outtype', args.outtype, '--outfile', str(base)], check=True)
    written.append((base, args.outtype))
    if args.quantize:
        build = source/'build-tools'
        subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                        '-DLLAMA_CURL=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
                        '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=OFF'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--config', 'Release', '--target', 'llama-quantize'], check=True)
        binary = next(p for p in [build/'bin/llama-quantize', build/'bin/Release/llama-quantize.exe'] if p.exists())
        quantized = model_dir/f'qwen2.5-0.5b-instruct-{args.quantize.lower()}.gguf'
        subprocess.run([str(binary), str(base), str(quantized), args.quantize], check=True)
        written.append((quantized, args.quantize.lower()))

    index_path = model_dir/'gguf.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    files = index.get('files', []) if index.get('model_hash') == model_hash else []
    for path, precision in written:
        files = [f for f in files if f['file'] != path.name]
        files.append({'file': path.name, 'sha256': sha256(path), 'precision': precision,
                      'file_bytes': path.stat().st_size})
    index_path.write_text(json.dumps({'model_hash': model_hash, 'files': files}, indent=2) + '\n')
    for path, precision in written:
        print(f'approved {path.name} ({precision}) in {index_path}')


if __name__ == '__main__':
    main()
