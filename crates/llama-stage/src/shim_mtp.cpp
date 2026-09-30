// The model's multi-token-prediction (MTP) head, attached to a final layer-split stage so the
// stage can draft the next few tokens in the same pass that samples one. C++ because the
// next-token hidden states come from llama.cpp's staging API (llama-ext.h).
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct sg_mtp {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    int             n_embd  = 0;
    int             n_vocab = 0;
    int             n_seq   = 0;
    // Per sequence: the target's last kept hidden state, which pairs with the next token.
    std::vector<float> pending;
};

extern "C" {

// Loads an MTP-only GGUF and makes a draft context with n_seq sequences of n_ctx each. The
// target context is switched to report every position's next-token hidden state.
sg_mtp * sg_mtp_open(llama_context * target, const char * path, int n_gpu_layers, int n_ctx, int n_seq,
                     int n_threads, char * err, size_t err_len) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.load_mode    = LLAMA_LOAD_MODE_NONE;
    mp.load_mtp     = true;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        snprintf(err, err_len, "llama.cpp could not load the MTP head %s", path);
        return nullptr;
    }
    if (llama_model_n_layer_nextn(model) < 1) {
        llama_model_free(model);
        snprintf(err, err_len, "%s has no MTP layer", path);
        return nullptr;
    }
    const llama_model * target_model = llama_get_model(target);
    if (llama_model_n_embd_out(model) != llama_model_n_embd_out(target_model) ||
        llama_vocab_n_tokens(llama_model_get_vocab(model)) != llama_vocab_n_tokens(llama_model_get_vocab(target_model))) {
        llama_model_free(model);
        snprintf(err, err_len, "the MTP head's width or vocabulary differs from the model's");
        return nullptr;
    }

    llama_context_params cp = llama_context_default_params();
    cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx           = (uint32_t) n_ctx * (uint32_t) n_seq;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = (uint32_t) n_seq;
    cp.kv_unified      = false;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.ctx_other       = target;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        snprintf(err, err_len, "llama.cpp could not create the MTP context");
        return nullptr;
    }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    llama_set_embeddings_nextn(target, true, /*masked*/ false);

    sg_mtp * m = new sg_mtp();
    m->model   = model;
    m->ctx     = ctx;
    m->n_embd  = llama_model_n_embd_out(model);
    m->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    m->n_seq   = n_seq;
    m->pending.assign((size_t) n_seq * m->n_embd, 0.0f);
    return m;
}

static void set_row(llama_batch & batch, int i, llama_token token, llama_pos pos, int seq, const float * h, int n_embd,
                    bool output) {
    batch.token[i]     = token;
    batch.pos[i]       = pos;
    batch.n_seq_id[i]  = 1;
    batch.seq_id[i][0] = seq;
    batch.logits[i]    = output;
    memcpy(batch.embd + (size_t) i * n_embd, h, (size_t) n_embd * sizeof(float));
}

// After the target decoded a batch for `seq` starting at `pos`, whose first `keep` inputs were
// `tokens` (the rest were rejected drafts): feed those positions to the MTP head, then draft
// up to n_draft tokens after `next`, the token the target chose after them. A draft is kept
// only while the head's probability for it is at least p_min. Returns the number of drafts.
int sg_mtp_step(sg_mtp * m, llama_context * target, int seq, const int32_t * tokens, int keep, int pos,
                int32_t next, int n_draft, float p_min, int32_t * drafts, char * err, size_t err_len) {
    if (seq < 0 || seq >= m->n_seq || keep < 1 || keep > 512 || n_draft < 0) {
        snprintf(err, err_len, "bad MTP step: seq=%d keep=%d n_draft=%d", seq, keep, n_draft);
        return -1;
    }
    const int n_embd  = m->n_embd;
    float *   pending = m->pending.data() + (size_t) seq * n_embd;
    // Draft positions from the previous step, and anything past this batch, are stale.
    llama_memory_seq_rm(llama_get_memory(m->ctx), seq, pos, -1);

    // Position q of the head pairs the target's hidden state at q-1 with the token at q.
    llama_batch batch = llama_batch_init(keep, n_embd, 1);
    batch.token       = (llama_token *) malloc(sizeof(llama_token) * (size_t) keep);
    for (int i = 0; i < keep; ++i) {
        const float * h = i == 0 ? pending : llama_get_embeddings_nextn_ith(target, i - 1);
        if (!h) {
            llama_batch_free(batch);
            snprintf(err, err_len, "the target produced no next-token hidden state");
            return -1;
        }
        set_row(batch, i, tokens[i], pos + i, seq, h, n_embd, false);
    }
    batch.n_tokens = keep;
    int rc = llama_decode(m->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        snprintf(err, err_len, "MTP catch-up decode failed with %d", rc);
        return -1;
    }
    const float * last = llama_get_embeddings_nextn_ith(target, keep - 1);
    if (!last) {
        snprintf(err, err_len, "the target produced no next-token hidden state");
        return -1;
    }
    memcpy(pending, last, (size_t) n_embd * sizeof(float));
    if (n_draft == 0) {
        return 0;
    }

    // Draft: each step feeds the head its own hidden state and the token it just proposed.
    std::vector<float> h(pending, pending + n_embd);
    llama_batch one = llama_batch_init(1, n_embd, 1);
    one.token       = (llama_token *) malloc(sizeof(llama_token));
    llama_token token = next;
    int         n     = 0;
    for (; n < n_draft; ++n) {
        set_row(one, 0, token, pos + keep + n, seq, h.data(), n_embd, true);
        one.n_tokens = 1;
        if (llama_decode(m->ctx, one) != 0) {
            break;
        }
        const float * logits = llama_get_logits_ith(m->ctx, 0);
        const float * h_next = llama_get_embeddings_nextn_ith(m->ctx, 0);
        if (!logits || !h_next) {
            break;
        }
        int best = 0;
        for (int v = 1; v < m->n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        if (p_min > 0.0f) {
            double sum = 0.0;
            for (int v = 0; v < m->n_vocab; ++v) {
                sum += exp((double) logits[v] - (double) logits[best]);
            }
            if (1.0 / sum < p_min) {
                break;
            }
        }
        drafts[n] = best;
        token     = best;
        h.assign(h_next, h_next + n_embd);
    }
    llama_batch_free(one);
    return n;
}

// Forgets a sequence, so a new session can use it from position zero.
void sg_mtp_clear(sg_mtp * m, int seq) {
    if (seq < 0 || seq >= m->n_seq) {
        return;
    }
    llama_memory_seq_rm(llama_get_memory(m->ctx), seq, -1, -1);
    std::fill(m->pending.begin() + (size_t) seq * m->n_embd, m->pending.begin() + (size_t) (seq + 1) * m->n_embd, 0.0f);
}

void sg_mtp_free(sg_mtp * m) {
    if (m) {
        llama_free(m->ctx);
        llama_model_free(m->model);
        delete m;
    }
}

}  // extern "C"
