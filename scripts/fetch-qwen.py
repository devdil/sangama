#!/usr/bin/env python3
"""Download a pinned Qwen checkpoint and produce actual per-worker safetensors files.

Uses only Python's standard library plus curl. Inference remains entirely Rust.
SafeTensors are partitioned without executing model code or importing torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHTS_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CONFIG_GIT_SHA1 = "0dbb161213629a23f0fc00ef286e6b1e366d180f"
TOKENIZER_GIT_SHA1 = "443909a61d429dff23010e5bddd28ff530edda00"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(folder, name):
    dest = folder / name
    if not dest.exists():
        print(f"Downloading {name}", flush=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--proto", "=https",
                        "--tlsv1.2", "--output", str(partial),
                        f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"], check=True)
        partial.replace(dest)
    return dest


def write_shard(source, dest, tensors, data_start):
    header = {}
    offset = 0
    names = sorted(tensors)
    for name in names:
        value = tensors[name]
        size = value["data_offsets"][1] - value["data_offsets"][0]
        header[name] = {"dtype": value["dtype"], "shape": value["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = dest.with_suffix(dest.suffix + ".part")
    with open(source, "rb") as src, open(partial, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for name in names:
            start, end = tensors[name]["data_offsets"]
            src.seek(data_start + start)
            remaining = end - start
            while remaining:
                block = src.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Truncated source weights")
                out.write(block)
                remaining -= len(block)
    partial.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".models/qwen2.5-0.5b-instruct"))
    parser.add_argument("--shards", type=int, default=2, choices=range(1, 9))
    args = parser.parse_args()
    folder = args.model_dir
    folder.mkdir(parents=True, exist_ok=True)
    weights = download(folder, "model.safetensors")
    config_path = download(folder, "config.json")
    tokenizer = download(folder, "tokenizer.json")
    download(folder, "LICENSE")
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Model SHA256 mismatch; remove the corrupt model.safetensors and retry")
    if git_sha1(config_path) != CONFIG_GIT_SHA1 or git_sha1(tokenizer) != TOKENIZER_GIT_SHA1:
        raise ValueError("Config/tokenizer does not match the pinned Hugging Face revision")
    cfg = json.loads(config_path.read_text())
    layers = cfg["num_hidden_layers"]
    with open(weights, "rb") as src:
        header_size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(header_size))
    manifest = {"model_id": MODEL, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
                "config_sha256": sha256(config_path), "tokenizer_sha256": sha256(tokenizer), "shards": []}
    for index in range(args.shards):
        start, end = index * layers // args.shards, (index + 1) * layers // args.shards
        selected = {}
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name.startswith("model.layers.") and start <= int(name.split(".")[2]) < end:
                selected[name] = tensor
            elif name == "model.embed_tokens.weight" and (start == 0 or end == layers):
                # Qwen2.5 0.5B ties its output projection to the input embeddings.
                selected[name] = tensor
            elif name in ("model.norm.weight", "lm_head.weight") and end == layers:
                selected[name] = tensor
        filename = f"shard-{index}-of-{args.shards}.safetensors"
        dest = folder / filename
        print(f"Writing {filename}: layers [{start}, {end})", flush=True)
        write_shard(weights, dest, selected, header_size + 8)
        manifest["shards"].append({"index": index, "start": start, "end": end, "file": filename,
                                   "sha256": sha256(dest), "file_bytes": dest.stat().st_size,
                                   "tensor_count": len(selected)})
    manifest_path = folder / "manifest.json"
    temporary = folder / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print(f"Ready: {manifest_path}\nPinned checkpoint and all shard hashes verified.")


if __name__ == "__main__":
    main()
