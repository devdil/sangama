#!/bin/sh
# Rebuild the llama.cpp layer-split spike from scratch in work/ (git-ignored).
# Needs a C++ toolchain and Python 3; the prepared Qwen checkpoint must exist
# (python3 scripts/fetch-qwen.py). Set NGL=0 to run on CPU instead of the GPU.
set -eu
cd "$(dirname "$0")/../.."
root=$PWD
work=$root/work
spike=$root/spikes/llamacpp
# unslothai/llama.cpp PR #180 head: layer-range stages with hidden-state input/output.
sha=bc230ecdf285dbf76bb9db1dfd9c1deb2189103f
mkdir -p "$work/gguf" "$work/spike"

if [ ! -d "$work/llama.cpp" ]; then
    curl -fL --retry 5 --retry-all-errors -o "$work/llama.tgz" \
        "https://codeload.github.com/unslothai/llama.cpp/tar.gz/$sha"
    mkdir "$work/llama.cpp"
    tar -xzf "$work/llama.tgz" -C "$work/llama.cpp" --strip-components=1
    patch -d "$work/llama.cpp" -p1 < "$root/crates/llama-stage/patches/qwen2-layer-split.patch"
fi

if [ ! -x "$work/venv/bin/cmake" ]; then
    python3 -m venv "$work/venv"
    "$work/venv/bin/pip" install -q cmake ninja
    "$work/venv/bin/pip" install -q -r "$work/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
fi
PATH=$work/venv/bin:$PATH

cmake -S "$work/llama.cpp" -B "$work/llama.cpp/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF > /dev/null
cmake --build "$work/llama.cpp/build" --target test-layer-split llama-quantize > /dev/null

f32=$work/gguf/qwen2.5-0.5b-f32.gguf
q4=$work/gguf/qwen2.5-0.5b-q4_k_m.gguf
[ -f "$f32" ] || python "$work/llama.cpp/convert_hf_to_gguf.py" .models/qwen2.5-0.5b-instruct \
    --outtype f32 --outfile "$f32"
[ -f "$q4" ] || "$work/llama.cpp/build/bin/llama-quantize" "$f32" "$q4" Q4_K_M > /dev/null 2>&1

lib=$work/llama.cpp/build/bin
c++ -std=c++17 -O2 "$spike/pipeline-gen.cpp" -I"$work/llama.cpp/include" \
    -I"$work/llama.cpp/ggml/include" -L"$lib" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$lib" -o "$work/spike/pipeline-gen"

# Sangama's formatted prompt for "Explain peer-to-peer computing in one short sentence."
prompt=151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,840,20772,14397,4686,78597,24231,304,825,2805,11652,13,151645,198,151644,77091,198
for model in "$f32" "$q4"; do
    for split in 1 12 23; do
        "$lib/test-layer-split" "$model" "$split" \
            "Explain peer-to-peer computing in one short sentence." 2>/dev/null | tail -1
    done
    NOMMAP=1 "$work/spike/pipeline-gen" "$model" 12 40 "$prompt"
done
