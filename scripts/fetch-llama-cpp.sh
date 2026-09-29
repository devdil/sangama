#!/bin/sh
# Download the pinned llama.cpp source with layer-range stages, verify it and apply
# Sangama's patches. `crates/llama-stage` builds from the result (.tools/llama.cpp).
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# unslothai/llama.cpp PR #180: layer-range stages with hidden-state input and output.
revision=bc230ecdf285dbf76bb9db1dfd9c1deb2189103f
sha256=74aa5b044ff030a6166121ef624f35b6bc0c93ec13016bfabaf4a56732ff6892
dest=$project_dir/.tools/llama.cpp
pin=$revision+$(cat "$project_dir"/crates/llama-stage/patches/*.patch | shasum -a 256 | cut -d' ' -f1)

if [ -f "$dest/.sangama-pin" ] && [ "$(cat "$dest/.sangama-pin")" = "$pin" ]; then
    echo "llama.cpp already prepared at $dest"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --retry 5 --retry-all-errors -o "$tmp/llama.tgz" \
    "https://codeload.github.com/unslothai/llama.cpp/tar.gz/$revision"
echo "$sha256  $tmp/llama.tgz" | shasum -a 256 -c -
mkdir "$tmp/src"
tar -xzf "$tmp/llama.tgz" -C "$tmp/src" --strip-components=1
for patch in "$project_dir"/crates/llama-stage/patches/*.patch; do
    patch -d "$tmp/src" -p1 --forward < "$patch"
done
echo "$pin" > "$tmp/src/.sangama-pin"
rm -rf "$dest"
mkdir -p "$(dirname "$dest")"
mv "$tmp/src" "$dest"
echo "prepared llama.cpp at $dest"
