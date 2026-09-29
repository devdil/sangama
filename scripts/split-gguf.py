#!/usr/bin/env python3
"""Split a GGUF model into per-layer slices with a checksummed manifest, and assemble a
worker's layer range from those slices.

  split     GGUF files (one model, possibly split into parts) -> embed.gguf, layer-NNN.gguf,
            head.gguf (and extra.gguf for any tensor outside those groups) plus model.json.
  assemble  model.json + the slices a worker downloaded -> one GGUF holding only the layers
            [start, end) (with the embedding if start == 0 and the head if end == n_layer),
            loadable by Sangama's layer-range llama.cpp.

Every slice keeps the model's full metadata, so each is a valid GGUF by itself. Needs gguf-py
from the pinned llama.cpp (scripts/fetch-llama-cpp.sh): PYTHONPATH=.tools/llama.cpp/gguf-py.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import gguf

LAYER = re.compile(r'^blk\.(\d+)\.')
EMBED = ('token_embd.',)
HEAD = ('output.', 'output_norm.')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def group_of(name):
    m = LAYER.match(name)
    if m:
        return f'layer-{int(m.group(1)):03d}'
    if name.startswith(EMBED):
        return 'embed'
    if name.startswith(HEAD):
        return 'head'
    return 'extra'


def copy_metadata(reader, writer):
    """Copy every key except those the writer adds itself and the split bookkeeping."""
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith(('GGUF.', 'split.')):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def write_gguf(path, meta_reader, tensors):
    """tensors: list of (ReaderTensor, source reader) pairs."""
    arch = meta_reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(path, arch)
    copy_metadata(meta_reader, writer)
    for t, _ in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t, reader in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()


def split(args):
    sources = sorted(Path(p) for p in args.gguf)
    readers = [gguf.GGUFReader(p) for p in sources]
    meta = readers[0]
    arch = meta.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    n_layer = int(meta.get_field(f'{arch}.block_count').contents())
    groups = {}
    for reader in readers:
        for t in reader.tensors:
            groups.setdefault(group_of(t.name), []).append((t, reader))
    layers = sorted(int(g.split('-')[1]) for g in groups if g.startswith('layer-'))
    assert layers == list(range(len(layers))), 'layer numbers must be contiguous from 0'
    assert 'embed' in groups, 'no token embedding found'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slices = []
    order = ['embed'] + [f'layer-{i:03d}' for i in layers] + [g for g in ('head', 'extra') if g in groups]
    for name in order:
        path = out / f'{name}.gguf'
        write_gguf(path, meta, groups[name])
        entry = {'name': name, 'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path),
                 'tensors': len(groups[name])}
        if name.startswith('layer-'):
            entry['layer'] = int(name.split('-')[1])
        slices.append(entry)
        print(f"{name}: {entry['tensors']} tensors, {entry['bytes'] / 1e9:.2f} GB", flush=True)
    manifest = {
        'format': 'sangama-gguf-slices-v1',
        'model_id': args.model_id,
        'architecture': arch,
        'block_count': n_layer,
        'layers_present': len(layers),
        'separate_output': any(t.name == 'output.weight' for t, _ in groups.get('head', [])),
        'quantization': args.quantization,
        'source': [{'file': p.name, 'bytes': p.stat().st_size} for p in sources],
        'slices': slices,
    }
    (out / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"wrote {len(slices)} slices and model.json to {out}")


def assemble(args):
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    total = manifest['layers_present']
    assert 0 <= args.start < args.end <= total, f'range must be within 0..{total}'
    wanted = [f'layer-{i:03d}' for i in range(args.start, args.end)]
    if args.start == 0:
        wanted.insert(0, 'embed')
    if args.end == total:
        wanted += [s['name'] for s in manifest['slices'] if s['name'] in ('head', 'extra')]
        # A model without output.weight reuses the token embedding as its LM head (tied).
        if not manifest.get('separate_output', True) and 'embed' not in wanted:
            wanted.insert(0, 'embed')
    by_name = {s['name']: s for s in manifest['slices']}
    readers, tensors = [], []
    for name in wanted:
        entry = by_name[name]
        path = base / entry['file']
        if not args.skip_verify and sha256(path) != entry['sha256']:
            sys.exit(f'{path} does not match the manifest checksum')
        reader = gguf.GGUFReader(path)
        readers.append(reader)
        tensors += [(t, reader) for t in reader.tensors]
    write_gguf(args.out, readers[0], tensors)
    print(f'assembled {len(wanted)} slices ({len(tensors)} tensors) into {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split', help='split GGUF parts into per-layer slices')
    s.add_argument('gguf', nargs='+', help='all parts of one GGUF model')
    s.add_argument('--out', required=True)
    s.add_argument('--model-id', required=True, help='e.g. unsloth/Qwen3.5-397B-A17B-GGUF')
    s.add_argument('--quantization', required=True, help='e.g. Q4_K_M')
    s.set_defaults(func=split)
    a = sub.add_parser('assemble', help='build one GGUF for a layer range from slices')
    a.add_argument('manifest', help='model.json written by split')
    a.add_argument('--start', type=int, required=True)
    a.add_argument('--end', type=int, required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--skip-verify', action='store_true', help='do not re-hash slices (they were verified on download)')
    a.set_defaults(func=assemble)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
