// Checks that a layer range loaded from a per-stage slice file computes exactly what the same
// range computes when loaded from the original model files.
//
// usage: stage-compare <original.gguf (first part)> <assembled-range.gguf> <beg> <end>
//        stage-compare chain <model.gguf> <beg> <mid> <end>
//
// The chain form runs [beg, end) as one stage and as two chained stages [beg, mid) -> [mid, end),
// which checks the layer-range wiring itself (hidden-state handoff, final norm and LM head).
//
// The first stage is fed a fixed token sequence; later stages get the same deterministic hidden
// states from both files. Prints the maximum absolute difference of the stage output (hidden
// states, or logits for the final stage) and exits non-zero unless it is exactly zero.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> run_stage(const char * path, int beg, int end, int * n_layer_out,
                                    const std::vector<float> * hidden = nullptr) {
    setenv("LLAMA_PP_IL_BEG", std::to_string(beg).c_str(), 1);
    setenv("LLAMA_PP_IL_END", std::to_string(end).c_str(), 1);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;  // read only this range's tensors into memory
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", path); exit(2); }
    const int n_layer = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    *n_layer_out = n_layer;
    const bool last = end >= n_layer;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 16;
    if (!last) { cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); exit(2); }

    const int n = 8;
    // M-RoPE models read four position sections per token when the batch carries hidden states.
    const llama_rope_type rope = llama_model_rope_type(model);
    const int sections = (beg != 0 && (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE)) ? 4 : 1;
    llama_batch b = llama_batch_init(n * sections, beg == 0 ? 0 : n_embd, 1);
    for (int j = 0; j < sections; ++j) {
        for (int i = 0; i < n; ++i) {
            b.pos[j * n + i] = i;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (beg == 0) {
            b.token[i] = 1000 + 37 * i;
        }
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = last ? (i == n - 1) : 1;
    }
    if (beg != 0 && hidden) {
        memcpy(b.embd, hidden->data(), (size_t) n * n_embd * sizeof(float));
    } else if (beg != 0) {
        for (int i = 0; i < n * n_embd; ++i) {
            b.embd[i] = 0.02f * std::sin(0.001f * (float) i + 0.3f * (float) (i % 7));
        }
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(2); }
    const float * out = last ? llama_get_logits_ith(ctx, n - 1) : llama_get_embeddings(ctx);
    std::vector<float> result(out, out + (last ? (size_t) n_vocab : (size_t) n * n_embd));
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s original.gguf assembled.gguf beg end | chain model.gguf beg mid end\n", argv[0]); return 2; }
    if (!getenv("LOGS")) llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    const bool chain = strcmp(argv[1], "chain") == 0;
    if (chain && argc < 6) { fprintf(stderr, "chain needs model beg mid end\n"); return 2; }
    const int beg = atoi(argv[3]), end = atoi(argv[chain ? 5 : 4]);
    int n_layer = 0;
    std::vector<float> a, b;
    if (chain) {
        const int mid = atoi(argv[4]);
        a = run_stage(argv[2], beg, end, &n_layer);
        std::vector<float> first = run_stage(argv[2], beg, mid, &n_layer);
        b = run_stage(argv[2], mid, end, &n_layer, &first);
        printf("chain [%d, %d) -> [%d, %d) vs single stage: ", beg, mid, mid, end);
    } else {
        a = run_stage(argv[1], beg, end, &n_layer);
        b = run_stage(argv[2], beg, end, &n_layer);
    }
    if (a.size() != b.size()) { printf("FAIL: output sizes differ (%zu vs %zu)\n", a.size(), b.size()); return 1; }
    double max_diff = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
        max_diff = std::fmax(max_diff, std::fabs((double) a[i] - (double) b[i]));
    }
    printf("range [%d, %d) of %d: %zu outputs, finite=%s, max |diff| = %g -> %s\n", beg, end, n_layer, a.size(),
           finite ? "yes" : "no", max_diff, finite && max_diff == 0 ? "PASS" : "FAIL");
    return finite && max_diff == 0 ? 0 : 1;
}
