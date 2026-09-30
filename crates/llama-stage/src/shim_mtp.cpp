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
#include <thread>
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
    // Rows are then indexed by batch position, whatever order the device ran them in. The
    // stage must mark every position as an output (sg_stage_output_all).
    llama_set_embeddings_nextn(target, true, /*masked*/ true);

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

// The most likely token of each of the batch's first n rows and, when p_min > 0, whether its
// probability reaches p_min. A row is 250k logits, so several rows are scanned on threads.
static bool pick_rows(sg_mtp * m, int n, float p_min, std::vector<int32_t> & best, std::vector<char> & sure) {
    std::vector<const float *> rows(n);
    for (int r = 0; r < n; ++r) {
        rows[r] = llama_get_logits_ith(m->ctx, r);
        if (!rows[r]) {
            return false;
        }
    }
    best.assign(n, 0);
    sure.assign(n, 1);
    const int n_vocab = m->n_vocab;
    auto scan = [&](int r) {
        const float * logits = rows[r];
        int           top    = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[top]) {
                top = v;
            }
        }
        best[r] = top;
        if (p_min > 0.0f) {
            // The draft's probability is 1 / sum(exp(logit - best)). Entries more than 16
            // below the best add under 3% in total over a 250k vocabulary, so skip their exp.
            const float floor = logits[top] - 16.0f;
            float       sum   = 0.0f;
            for (int v = 0; v < n_vocab; ++v) {
                if (logits[v] > floor) {
                    sum += expf(logits[v] - logits[top]);
                }
            }
            sure[r] = 1.0f / sum >= p_min;
        }
    };
    const int n_threads = std::min(n, 8);
    if (n_threads < 2) {
        scan(0);
        return true;
    }
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            for (int r = t; r < n; r += n_threads) {
                scan(r);
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    return true;
}

// After the target decoded a batch holding one frame of each of n_items sequences, given in
// increasing sequence order. Item i's frame began at batch row row0[i] and position pos[i];
// its first keep[i] inputs were kept (the rest were rejected drafts) and lie one item after
// another in `tokens`. Feeds those positions to the MTP head, then drafts up to n_draft[i]
// tokens after next[i], the token the target chose after them. A draft is kept only while the
// head's probability for it is at least p_min. Item i's drafts go to drafts[i * max_draft ..]
// and their number to counts[i]. Every sequence shares each device call.
int sg_mtp_step_many(sg_mtp * m, llama_context * target, int n_items, const int * seq, const int * row0,
                     const int * keep, const int * pos, const int32_t * next, const int * n_draft,
                     const int32_t * tokens, float p_min, int max_draft, int32_t * drafts, int * counts, char * err,
                     size_t err_len) {
    const int n_embd = m->n_embd;
    int       steps  = 0;
    for (int i = 0; i < n_items; ++i) {
        if (seq[i] < 0 || seq[i] >= m->n_seq || row0[i] < 0 || keep[i] < 1 || keep[i] > 512 || n_draft[i] < 0 ||
            n_draft[i] > max_draft || (i > 0 && seq[i] <= seq[i - 1])) {
            snprintf(err, err_len, "bad MTP step: seq=%d keep=%d n_draft=%d", seq[i], keep[i], n_draft[i]);
            return -1;
        }
        steps     = std::max(steps, n_draft[i]);
        counts[i] = 0;
        // Draft positions from the previous step, and anything past this batch, are stale.
        llama_memory_seq_rm(llama_get_memory(m->ctx), seq[i], pos[i], -1);
    }

    // Position q of the head pairs the target's hidden state at q-1 with the token at q.
    const int   capacity = 512;
    llama_batch batch    = llama_batch_init(capacity, n_embd, 1);
    batch.token          = (llama_token *) malloc(sizeof(llama_token) * (size_t) capacity);
    batch.n_tokens       = 0;
    auto flush = [&]() {
        if (batch.n_tokens == 0) {
            return 0;
        }
        const int rc   = llama_decode(m->ctx, batch);
        batch.n_tokens = 0;
        return rc;
    };
    int rc = 0;
    for (int i = 0, t = 0; i < n_items && rc == 0; ++i) {
        for (int k = 0; k < keep[i] && rc == 0; ++k, ++t) {
            const float * h = k == 0 ? m->pending.data() + (size_t) seq[i] * n_embd
                                     : llama_get_embeddings_nextn_ith(target, row0[i] + k - 1);
            if (!h) {
                llama_batch_free(batch);
                snprintf(err, err_len, "the target produced no next-token hidden state");
                return -1;
            }
            set_row(batch, batch.n_tokens++, tokens[t], pos[i] + k, seq[i], h, n_embd, false);
            if (batch.n_tokens == capacity) {
                rc = flush();
            }
        }
    }
    if (rc == 0) {
        rc = flush();
    }
    if (rc != 0) {
        llama_batch_free(batch);
        snprintf(err, err_len, "MTP catch-up decode failed with %d", rc);
        return -1;
    }
    for (int i = 0; i < n_items; ++i) {
        const float * last = llama_get_embeddings_nextn_ith(target, row0[i] + keep[i] - 1);
        if (!last) {
            llama_batch_free(batch);
            snprintf(err, err_len, "the target produced no next-token hidden state");
            return -1;
        }
        memcpy(m->pending.data() + (size_t) seq[i] * n_embd, last, (size_t) n_embd * sizeof(float));
    }

    // Draft: each step feeds the head its own hidden state and the token it just proposed. A
    // sequence the head became unsure of stays in the batch, so the batch keeps its shape and
    // the device reuses its graph; what it drafts from then on is ignored.
    std::vector<float>       h((size_t) n_items * n_embd);
    std::vector<llama_token> token(next, next + n_items);
    std::vector<char>        open(n_items, 1);
    std::vector<int>         rows;
    std::vector<int32_t>     best;
    std::vector<char>        sure;
    for (int i = 0; i < n_items; ++i) {
        memcpy(h.data() + (size_t) i * n_embd, m->pending.data() + (size_t) seq[i] * n_embd,
               (size_t) n_embd * sizeof(float));
    }
    for (int step = 0; step < steps; ++step) {
        rows.clear();
        for (int i = 0; i < n_items && (int) rows.size() < capacity; ++i) {
            if (n_draft[i] > step) {
                set_row(batch, (int) rows.size(), token[i], pos[i] + keep[i] + step, seq[i],
                        h.data() + (size_t) i * n_embd, n_embd, true);
                rows.push_back(i);
            }
        }
        batch.n_tokens = (int) rows.size();
        if (llama_decode(m->ctx, batch) != 0 || !pick_rows(m, (int) rows.size(), p_min, best, sure)) {
            break;
        }
        bool any = false;
        for (int r = 0; r < (int) rows.size(); ++r) {
            const int     i      = rows[r];
            const float * h_next = llama_get_embeddings_nextn_ith(m->ctx, r);
            if (!h_next || !sure[r]) {
                open[i] = 0;
            }
            if (h_next) {
                memcpy(h.data() + (size_t) i * n_embd, h_next, (size_t) n_embd * sizeof(float));
            }
            token[i] = best[r];
            if (open[i]) {
                drafts[(size_t) i * max_draft + counts[i]++] = best[r];
                any = true;
            }
        }
        if (!any) {
            break;
        }
    }
    batch.n_tokens = 0;
    llama_batch_free(batch);
    return 0;
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
