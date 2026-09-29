// Spike: greedy generation through two llama.cpp stages, each loading only its own layers
// and keeping its own KV cache, compared with the unsplit model.
//
// usage: pipeline-gen <model.gguf> <split_layer> <max_tokens> <comma-separated prompt ids>

#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const llama_token EOS = 151645;  // <|im_end|> for Qwen2.5-Instruct

struct Stage {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
};

static Stage load_stage(const char * path, int beg, int end, bool hidden_out) {
    // The patched loader and context read the layer range from the environment.
    if (beg >= 0) {
        setenv("LLAMA_PP_IL_BEG", std::to_string(beg).c_str(), 1);
        setenv("LLAMA_PP_IL_END", std::to_string(end).c_str(), 1);
    } else {
        unsetenv("LLAMA_PP_IL_BEG");
        unsetenv("LLAMA_PP_IL_END");
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 999;
    if (getenv("NOMMAP")) mp.load_mode = LLAMA_LOAD_MODE_NONE;
    Stage s;
    s.model = llama_model_load_from_file(path, mp);
    if (!s.model) { fprintf(stderr, "load failed\n"); exit(1); }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_threads = cp.n_threads_batch = 4;
    if (hidden_out) {
        cp.embeddings   = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    s.ctx = llama_init_from_model(s.model, cp);
    if (!s.ctx) { fprintf(stderr, "context failed\n"); exit(1); }
    return s;
}

static llama_token argmax(const float * logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

// Stage A: tokens at positions [pos, pos+n) -> hidden states (n x n_embd).
static std::vector<float> run_a(Stage & a, const std::vector<llama_token> & toks, int pos, int n_embd) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = (int) toks.size();
    if (llama_decode(a.ctx, b) != 0) { fprintf(stderr, "stage A decode failed\n"); exit(1); }
    std::vector<float> h(toks.size() * n_embd);
    memcpy(h.data(), llama_get_embeddings(a.ctx), h.size() * sizeof(float));
    llama_batch_free(b);
    return h;
}

// Stage B: hidden states at positions [pos, pos+n) -> next token from the last position.
static llama_token run_b(Stage & bstage, const std::vector<float> & h, int n, int pos, int n_embd, int n_vocab) {
    llama_batch b = llama_batch_init(n, n_embd, 1);
    memcpy(b.embd, h.data(), (size_t) n * n_embd * sizeof(float));
    for (int i = 0; i < n; ++i) {
        b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == n - 1);
    }
    b.n_tokens = n;
    if (llama_decode(bstage.ctx, b) != 0) { fprintf(stderr, "stage B decode failed\n"); exit(1); }
    llama_token t = argmax(llama_get_logits_ith(bstage.ctx, n - 1), n_vocab);
    llama_batch_free(b);
    return t;
}

static llama_token run_full(Stage & f, const std::vector<llama_token> & toks, int pos, int n_vocab) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (i == toks.size() - 1);
    }
    b.n_tokens = (int) toks.size();
    if (llama_decode(f.ctx, b) != 0) { fprintf(stderr, "full decode failed\n"); exit(1); }
    llama_token t = argmax(llama_get_logits_ith(f.ctx, (int) toks.size() - 1), n_vocab);
    llama_batch_free(b);
    return t;
}

static void print_ids(const char * label, const std::vector<llama_token> & ids, double secs) {
    printf("%s (%zu tokens, %.1f tok/s):", label, ids.size(), secs > 0 ? ids.size() / secs : 0.0);
    for (auto t : ids) printf(" %d", t);
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model split max_tokens ids\n", argv[0]); return 1; }
    const char * path = argv[1];
    const int split = atoi(argv[2]);
    const int max_tokens = atoi(argv[3]);
    std::vector<llama_token> prompt;
    for (char * p = strtok(argv[4], ","); p; p = strtok(nullptr, ",")) prompt.push_back(atoi(p));

    if (!getenv("LOGS")) llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    // Unsplit reference.
    Stage full = load_stage(path, -1, -1, false);
    const int n_layer = llama_model_n_layer(full.model);
    const int n_embd  = llama_model_n_embd(full.model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(full.model));
    std::vector<llama_token> ref;
    auto t0 = std::chrono::steady_clock::now();
    llama_token t = run_full(full, prompt, 0, n_vocab);
    int pos = (int) prompt.size();
    while (true) {
        ref.push_back(t);
        if (t == EOS || (int) ref.size() >= max_tokens) break;
        t = run_full(full, {t}, pos++, n_vocab);
    }
    double ref_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    llama_free(full.ctx); llama_model_free(full.model);

    // Two stages, each with only its own layers and KV cache.
    Stage a = load_stage(path, 0, split, true);
    Stage b = load_stage(path, split, n_layer, false);
    std::vector<llama_token> got;
    t0 = std::chrono::steady_clock::now();
    auto h = run_a(a, prompt, 0, n_embd);
    t = run_b(b, h, (int) prompt.size(), 0, n_embd, n_vocab);
    pos = (int) prompt.size();
    while (true) {
        got.push_back(t);
        if (t == EOS || (int) got.size() >= max_tokens) break;
        h = run_a(a, {t}, pos, n_embd);
        t = run_b(b, h, 1, pos, n_embd, n_vocab);
        ++pos;
    }
    double split_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("layers=%d split=%d prompt_tokens=%zu\n", n_layer, split, prompt.size());
    printf("weights loaded: stage A %.1f MiB, stage B %.1f MiB\n", llama_model_size(a.model) / 1048576.0, llama_model_size(b.model) / 1048576.0);
    print_ids("unsplit", ref, ref_s);
    print_ids("split  ", got, split_s);
    printf("%s\n", ref == got ? "MATCH: split generation equals unsplit" : "MISMATCH");
    llama_free(a.ctx); llama_model_free(a.model);
    llama_free(b.ctx); llama_model_free(b.model);
    return ref == got ? 0 : 1;
}
