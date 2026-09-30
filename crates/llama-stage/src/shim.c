// A narrow C interface over one llama.cpp layer-range stage. Compiling against llama.h here
// keeps llama.cpp's struct layouts out of the Rust FFI, which only sees plain types.
#include "ggml-backend.h"
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct sg_stage {
    struct llama_model * model;
    struct llama_context * ctx;
    int il_beg;
    int il_end;
    int n_layer;
    int n_embd;
    int n_vocab;
    int n_seq;
    // The final stage reports every position, not only the last: its MTP head reads them all.
    int output_all;
} sg_stage;

static void quiet_log(enum ggml_log_level level, const char * text, void * data) {
    (void) level;
    (void) text;
    (void) data;
}

static void set_env(const char * name, int value) {
    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%d", value);
#ifdef _WIN32
    _putenv_s(name, buffer);
#else
    setenv(name, buffer, 1);
#endif
}

static void clear_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

void sg_init(int verbose) {
    if (!verbose) {
        llama_log_set(quiet_log, NULL);
    }
    llama_backend_init();
}

// Free and total memory of the first GPU device llama.cpp found; returns 0 if there is none.
int sg_gpu_memory(size_t * free, size_t * total, char * name, size_t name_len) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ggml_backend_dev_memory(dev, free, total);
            snprintf(name, name_len, "%s", ggml_backend_dev_name(dev));
            return 1;
        }
    }
    return 0;
}

// n_ctx is per sequence; the context holds n_seq independent sequences (KV and recurrent state).
// n_rollback is how many positions a sequence can be rewound without a saved state: llama.cpp
// keeps that many earlier recurrent states on the device (models that support it; else 0).
sg_stage * sg_stage_open(const char * path, int il_beg, int il_end, int n_gpu_layers, int n_ctx,
                         int n_seq, int n_rollback, int n_threads, char * err, size_t err_len) {
    if (il_beg < 0 || il_beg >= il_end) {
        snprintf(err, err_len, "empty or negative layer range [%d, %d)", il_beg, il_end);
        return NULL;
    }
    if (n_seq < 1 || n_seq > 256) { // llama.cpp's LLAMA_MAX_SEQ
        snprintf(err, err_len, "sequence slots must be 1-256, got %d", n_seq);
        return NULL;
    }
    // The patched loader and context read the range from the environment; the Rust caller
    // serialises opens so no other stage sees these values.
    set_env("LLAMA_PP_IL_BEG", il_beg);
    set_env("LLAMA_PP_IL_END", il_end);

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    // Read the file instead of mapping it, so memory use reflects only this stage's layers.
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    struct llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        clear_env("LLAMA_PP_IL_BEG");
        clear_env("LLAMA_PP_IL_END");
        snprintf(err, err_len, "llama.cpp could not load %s", path);
        return NULL;
    }
    const int n_layer = llama_model_n_layer(model);
    if (il_end > n_layer) {
        clear_env("LLAMA_PP_IL_BEG");
        clear_env("LLAMA_PP_IL_END");
        llama_model_free(model);
        snprintf(err, err_len, "layer range [%d, %d) outside model with %d layers", il_beg, il_end,
                 n_layer);
        return NULL;
    }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx * (uint32_t) n_seq;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = (uint32_t) n_seq;
    // Separate KV per sequence, so each keeps the full n_ctx. With separate KV llama.cpp only
    // merges sequences into one step when their ids are consecutive; SANGAMA_KV_UNIFIED=1 shares
    // one attention buffer of n_ctx * n_seq cells instead, which lifts that rule.
    const char * unified = getenv("SANGAMA_KV_UNIFIED");
    cp.kv_unified = unified && unified[0] == '1';
    cp.n_rs_seq = (uint32_t) (n_rollback > 0 ? n_rollback : 0);
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    if (il_end < n_layer) {
        // A non-final stage returns every token's residual stream through the embeddings API.
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    struct llama_context * ctx = llama_init_from_model(model, cp);
    clear_env("LLAMA_PP_IL_BEG");
    clear_env("LLAMA_PP_IL_END");
    if (!ctx) {
        llama_model_free(model);
        snprintf(err, err_len, "llama.cpp could not create a context");
        return NULL;
    }

    sg_stage * s = calloc(1, sizeof(sg_stage));
    if (!s) {
        llama_free(ctx);
        llama_model_free(model);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    s->model = model;
    s->ctx = ctx;
    s->il_beg = il_beg;
    s->il_end = il_end;
    s->n_layer = n_layer;
    s->n_embd = llama_model_n_embd(model);
    s->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    s->n_seq = n_seq;
    return s;
}

int sg_stage_n_layer(const sg_stage * s) { return s->n_layer; }
int sg_stage_n_embd(const sg_stage * s) { return s->n_embd; }
int sg_stage_n_vocab(const sg_stage * s) { return s->n_vocab; }
int sg_stage_ftype(const sg_stage * s) { return (int) llama_model_ftype(s->model); }

int sg_stage_architecture(const sg_stage * s, char * buf, size_t len) {
    return llama_model_meta_val_str(s->model, "general.architecture", buf, len);
}

int sg_stage_n_seq(const sg_stage * s) { return s->n_seq; }
// Positions a sequence can be rewound with sg_stage_rollback; 0 if the model cannot.
int sg_stage_n_rollback(const sg_stage * s) { return (int) llama_n_rs_seq(s->ctx); }
// Discards a sequence's positions from pos on, right after the batch that decoded them, without
// a saved state. Returns 1 on success; 0 if more positions are dropped than the context keeps.
int sg_stage_rollback(sg_stage * s, int seq, int pos) {
    if (seq < 0 || seq >= s->n_seq || pos < 1) {
        return 0;
    }
    return llama_memory_seq_rm(llama_get_memory(s->ctx), seq, pos, -1) ? 1 : 0;
}
void sg_stage_output_all(sg_stage * s, int on) { s->output_all = on; }
// The stage's llama.cpp context, for the MTP head (shim_mtp.cpp).
struct llama_context * sg_stage_context(sg_stage * s) { return s->ctx; }

// Decodes n positions of sequence seq starting at pos. all_logits asks the final stage for
// every position's logits rather than only the last.
static int run(sg_stage * s, int seq, const int32_t * tokens, const float * hidden, int n, int pos,
               int all_logits, char * err, size_t err_len) {
    const int last = s->il_end == s->n_layer;
    const int takes_tokens = s->il_beg == 0;
    if (n <= 0) {
        snprintf(err, err_len, "bad decode size: n=%d", n);
        return -1;
    }
    if (seq < 0 || seq >= s->n_seq) {
        snprintf(err, err_len, "sequence %d outside 0-%d", seq, s->n_seq - 1);
        return -1;
    }
    if ((takes_tokens && !tokens) || (!takes_tokens && !hidden)) {
        snprintf(err, err_len, "stage input mismatch: %s expected", takes_tokens ? "tokens" : "hidden states");
        return -1;
    }
    // M-RoPE models (Qwen3.5) read n_pos_per_embd position sections per token when the batch
    // carries hidden states; supply the same text position in every section, or llama.cpp
    // reads uninitialised positions and the stage's output silently changes run to run.
    const enum llama_rope_type rope = llama_model_rope_type(s->model);
    const int sections = (!takes_tokens && (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE)) ? 4 : 1;
    struct llama_batch batch = llama_batch_init(n * sections, takes_tokens ? 0 : s->n_embd, 1);
    if (takes_tokens) {
        memcpy(batch.token, tokens, (size_t) n * sizeof(int32_t));
    } else {
        memcpy(batch.embd, hidden, (size_t) n * (size_t) s->n_embd * sizeof(float));
    }
    for (int j = 0; j < sections; ++j) {
        for (int i = 0; i < n; ++i) {
            batch.pos[j * n + i] = pos + i;
        }
    }
    for (int i = 0; i < n; ++i) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = last ? (all_logits || s->output_all || i == n - 1) : 1;
    }
    batch.n_tokens = n;
    const int rc = llama_decode(s->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        snprintf(err, err_len, "llama_decode failed with %d", rc);
        return -1;
    }
    return 0;
}

// Runs n positions starting at pos. The first stage takes tokens, later stages take
// n * n_embd hidden values. A non-final stage writes n * n_embd hidden values to out; the
// final stage writes the logits of the last position (n_vocab values).
int sg_stage_decode(sg_stage * s, int seq, const int32_t * tokens, const float * hidden, int n, int pos,
                    float * out, size_t out_len, char * err, size_t err_len) {
    const int last = s->il_end == s->n_layer;
    const size_t expected = last ? (size_t) s->n_vocab : (size_t) n * (size_t) s->n_embd;
    if (n <= 0 || out_len != expected) {
        snprintf(err, err_len, "bad decode size: n=%d out_len=%zu expected=%zu", n, out_len, expected);
        return -1;
    }
    if (run(s, seq, tokens, hidden, n, pos, 0, err, err_len) != 0) {
        return -1;
    }
    const float * result = last ? llama_get_logits_ith(s->ctx, n - 1) : llama_get_embeddings(s->ctx);
    if (!result) {
        snprintf(err, err_len, "llama.cpp returned no %s", last ? "logits" : "hidden states");
        return -1;
    }
    memcpy(out, result, expected * sizeof(float));
    return 0;
}

// Final stage only: the greedy next token after each of the n positions, to verify drafts.
int sg_stage_decode_greedy(sg_stage * s, int seq, const int32_t * tokens, const float * hidden, int n, int pos,
                           int32_t * ids, char * err, size_t err_len) {
    if (s->il_end != s->n_layer) {
        snprintf(err, err_len, "only the final stage samples");
        return -1;
    }
    if (run(s, seq, tokens, hidden, n, pos, 1, err, err_len) != 0) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        const float * logits = llama_get_logits_ith(s->ctx, i);
        if (!logits) {
            snprintf(err, err_len, "llama.cpp returned no logits for position %d", i);
            return -1;
        }
        int best = 0;
        for (int v = 1; v < s->n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        ids[i] = best;
    }
    return 0;
}

// Decodes one frame for each of n_items sequences in a single llama_decode, so the device works
// on several sessions at once. Item i has ns[i] positions of sequence seqs[i] starting at
// poss[i]; tokens (first stage) or hidden states (later stages) are concatenated in item order.
// A non-final stage writes every position's hidden state to out, in the same order. The final
// stage writes the greedy token after every position to ids.
int sg_stage_decode_many(sg_stage * s, int n_items, const int * seqs, const int * ns, const int * poss,
                         const int32_t * tokens, const float * hidden, float * out, size_t out_len,
                         int32_t * ids, char * err, size_t err_len) {
    const int last = s->il_end == s->n_layer;
    const int takes_tokens = s->il_beg == 0;
    int total = 0;
    for (int k = 0; k < n_items; ++k) {
        if (ns[k] <= 0 || seqs[k] < 0 || seqs[k] >= s->n_seq) {
            snprintf(err, err_len, "bad batch item %d: seq=%d n=%d", k, seqs[k], ns[k]);
            return -1;
        }
        total += ns[k];
    }
    if (n_items <= 0 || total > 512) {
        snprintf(err, err_len, "bad batch: %d items, %d positions", n_items, total);
        return -1;
    }
    if ((takes_tokens && !tokens) || (!takes_tokens && !hidden) || (last && !ids) ||
        (!last && out_len != (size_t) total * (size_t) s->n_embd)) {
        snprintf(err, err_len, "batch input or output does not match this stage");
        return -1;
    }
    const enum llama_rope_type rope = llama_model_rope_type(s->model);
    const int sections = (!takes_tokens && (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE)) ? 4 : 1;
    struct llama_batch batch = llama_batch_init(total * sections, takes_tokens ? 0 : s->n_embd, 1);
    if (takes_tokens) {
        memcpy(batch.token, tokens, (size_t) total * sizeof(int32_t));
    } else {
        memcpy(batch.embd, hidden, (size_t) total * (size_t) s->n_embd * sizeof(float));
    }
    int row = 0;
    for (int k = 0; k < n_items; ++k) {
        for (int i = 0; i < ns[k]; ++i, ++row) {
            for (int j = 0; j < sections; ++j) {
                batch.pos[j * total + row] = poss[k] + i;
            }
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = seqs[k];
            batch.logits[row] = 1;
        }
    }
    batch.n_tokens = total;
    const int rc = llama_decode(s->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        snprintf(err, err_len, "llama_decode failed with %d", rc);
        return -1;
    }
    for (int i = 0; i < total; ++i) {
        if (last) {
            const float * logits = llama_get_logits_ith(s->ctx, i);
            if (!logits) {
                snprintf(err, err_len, "llama.cpp returned no logits for row %d", i);
                return -1;
            }
            int best = 0;
            for (int v = 1; v < s->n_vocab; ++v) {
                if (logits[v] > logits[best]) {
                    best = v;
                }
            }
            ids[i] = best;
        } else {
            const float * row_out = llama_get_embeddings_ith(s->ctx, i);
            if (!row_out) {
                snprintf(err, err_len, "llama.cpp returned no hidden state for row %d", i);
                return -1;
            }
            memcpy(out + (size_t) i * (size_t) s->n_embd, row_out, (size_t) s->n_embd * sizeof(float));
        }
    }
    return 0;
}

// A sequence's cached state (attention KV and recurrent state), to roll back rejected drafts.
size_t sg_stage_state_size(sg_stage * s, int seq) { return llama_state_seq_get_size(s->ctx, seq); }
size_t sg_stage_state_save(sg_stage * s, int seq, uint8_t * buf, size_t len) {
    return llama_state_seq_get_data(s->ctx, buf, len, seq);
}
// Replaces a sequence's state with a saved one; returns 0 on failure.
size_t sg_stage_state_load(sg_stage * s, int seq, const uint8_t * buf, size_t len) {
    return llama_state_seq_set_data(s->ctx, buf, len, seq);
}

// Drops one sequence's cache so a new session can use its slot from position zero.
void sg_stage_clear(sg_stage * s, int seq) {
    llama_memory_seq_rm(llama_get_memory(s->ctx), seq, -1, -1);
}

void sg_stage_free(sg_stage * s) {
    if (s) {
        llama_free(s->ctx);
        llama_model_free(s->model);
        free(s);
    }
}
