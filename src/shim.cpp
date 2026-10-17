// Narrow C ABI over llama.cpp. Everything layout-sensitive (llama_context_params,
// llama_batch, ggml_tensor) stays on this side; Rust only sees the hd_* structs below.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// Mirrored in src/ffi.rs; keep field order identical.
struct hd_info {
    int32_t n_layer, n_embd, n_vocab, n_ctx_train;
    uint64_t n_params, size_bytes;
    char desc[128];
    int32_t n_head;
};

struct hd_cfg {
    int32_t n_threads;
    int32_t n_ubatch;    // 0 = whole sequence in one ubatch
    int32_t flash_attn;  // -1 auto, 0 off, 1 on
    int32_t type_k;      // ggml_type
    int32_t type_v;
};

struct hd_hooks {
    int32_t n;
    const char * const * names;
    float * const * out;        // per hook: capture buffer (n_tok*width) or null
    const float * const * src;  // per hook: patch source (n_tok*width) or null
    int32_t width;
};

namespace {

std::mutex g_log_mu;
std::string g_log;  // recent warnings/errors from llama.cpp, surfaced in error messages

void log_cb(ggml_log_level level, const char * text, void *) {
    if (getenv("HETERODYNE_VERBOSE")) fputs(text, stderr);
    if (level != GGML_LOG_LEVEL_ERROR && level != GGML_LOG_LEVEL_WARN) return;
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log += text;
    if (g_log.size() > 4096) g_log.erase(0, g_log.size() - 4096);
}

void set_err(char * err, int32_t errlen, const std::string & msg) {
    if (!err || errlen <= 0) return;
    std::string full = msg;
    {
        std::lock_guard<std::mutex> lk(g_log_mu);
        if (!g_log.empty()) { full += "\nllama.cpp log:\n" + g_log; g_log.clear(); }
    }
    snprintf(err, errlen, "%s", full.c_str());
}

int32_t copy_out(const std::string & s, char * buf, int32_t cap) {
    if ((int32_t) s.size() + 1 > cap) return -(int32_t) (s.size() + 1);
    memcpy(buf, s.data(), s.size());
    buf[s.size()] = 0;
    return (int32_t) s.size();
}

std::once_flag g_init;
void init_once() {
    std::call_once(g_init, [] {
        llama_log_set(log_cb, nullptr);
        llama_backend_init();
    });
}

// One hooked tensor: captured into `out` or overwritten from `src`, row-block by row-block
// (llama.cpp evaluates one ubatch at a time, so a tensor fires once per ubatch).
struct Hook {
    std::string name;
    float * out = nullptr;
    const float * src = nullptr;
    int64_t rows_done = 0;
};

struct HookState {
    std::vector<Hook> hooks;
    int64_t width = 0;   // expected ne[0] (n_embd)
    int64_t n_rows = 0;  // total rows expected (tokens)
    std::string error;
};

bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (HookState *) ud;
    const char * name = ggml_get_name(t);
    Hook * h = nullptr;
    for (auto & x : st->hooks) if (x.name == name) { h = &x; break; }
    if (ask) return h != nullptr;
    if (!h) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != st->width || t->ne[2] != 1 || t->ne[3] != 1) {
        st->error = std::string("tensor '") + name + "' has unexpected type/shape";
        return false;
    }
    const int64_t rows = t->ne[1];
    if (h->rows_done + rows > st->n_rows) {
        st->error = std::string("tensor '") + name + "' produced more rows than tokens";
        return false;
    }
    const size_t off = (size_t) h->rows_done * st->width;
    const size_t nbytes = (size_t) rows * st->width * sizeof(float);
    if (h->out) ggml_backend_tensor_get(t, h->out + off, 0, nbytes);
    if (h->src) ggml_backend_tensor_set(t, h->src + off, 0, nbytes);
    h->rows_done += rows;
    return true;
}

void clear_log() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log.clear();
}

llama_context * make_ctx(llama_model * m, int32_t n_ctx, const hd_cfg * cfg, ggml_backend_sched_eval_callback cb, void * ud) {
    clear_log();
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = cfg->n_ubatch > 0 ? std::min(cfg->n_ubatch, n_ctx) : n_ctx;
    cp.n_seq_max = 1;
    cp.n_threads = cfg->n_threads;
    cp.n_threads_batch = cfg->n_threads;
    cp.flash_attn_type = (llama_flash_attn_type) cfg->flash_attn;
    cp.type_k = (ggml_type) cfg->type_k;
    cp.type_v = (ggml_type) cfg->type_v;
    cp.no_perf = true;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = ud;
    return llama_init_from_model(m, cp);
}

bool any_non_finite(const float * x, size_t n) {
    for (size_t i = 0; i < n; i++) if (!std::isfinite(x[i])) return true;
    return false;
}

}  // namespace

extern "C" {

void * hd_model_load(const char * path, char * err, int32_t errlen) {
    init_once();
    clear_log();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(path, mp);
    if (!m) set_err(err, errlen, std::string("failed to load model: ") + path);
    return m;
}

void hd_model_free(void * m) { llama_model_free((llama_model *) m); }

void hd_model_info(void * mv, hd_info * out) {
    auto * m = (llama_model *) mv;
    memset(out, 0, sizeof(*out));
    out->n_layer = llama_model_n_layer(m);
    out->n_embd = llama_model_n_embd(m);
    out->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    out->n_ctx_train = llama_model_n_ctx_train(m);
    out->n_params = llama_model_n_params(m);
    out->size_bytes = llama_model_size(m);
    llama_model_desc(m, out->desc, sizeof(out->desc));
    out->n_head = llama_model_n_head(m);
}

int32_t hd_meta(void * m, const char * key, char * buf, int32_t cap) {
    return llama_model_meta_val_str((llama_model *) m, key, buf, cap);
}

int32_t hd_tokenize(void * m, const char * text, int32_t len, int32_t add_special, int32_t parse_special, int32_t * out, int32_t cap) {
    return llama_tokenize(llama_model_get_vocab((llama_model *) m), text, len, out, cap, add_special != 0, parse_special != 0);
}

int32_t hd_piece(void * m, int32_t tok, char * buf, int32_t cap) {
    return llama_token_to_piece(llama_model_get_vocab((llama_model *) m), tok, buf, cap, 0, true);
}

int32_t hd_is_eog(void * m, int32_t tok) {
    return llama_vocab_is_eog(llama_model_get_vocab((llama_model *) m), tok) ? 1 : 0;
}

// 1 if both vocabularies have identical token texts, else 0.
int32_t hd_vocab_equal(void * a, void * b) {
    auto * va = llama_model_get_vocab((llama_model *) a);
    auto * vb = llama_model_get_vocab((llama_model *) b);
    const int32_t n = llama_vocab_n_tokens(va);
    if (n != llama_vocab_n_tokens(vb)) return 0;
    for (int32_t i = 0; i < n; i++) {
        if (strcmp(llama_vocab_get_text(va, i), llama_vocab_get_text(vb, i)) != 0) return 0;
    }
    return 1;
}

// Wraps one user message in the model's built-in chat template. Returns length,
// -needed if buf is too small, or INT32_MIN if the model has no usable template.
int32_t hd_chat(void * m, const char * user, char * buf, int32_t cap) {
    const char * tmpl = llama_model_chat_template((llama_model *) m, nullptr);
    if (!tmpl) return INT32_MIN;
    llama_chat_message msg = {"user", user};
    std::vector<char> tmp(strlen(user) * 2 + 4096);
    int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, tmp.data(), tmp.size());
    if (n < 0) return INT32_MIN;
    if (n > (int32_t) tmp.size()) {
        tmp.resize(n + 1);
        n = llama_chat_apply_template(tmpl, &msg, 1, true, tmp.data(), tmp.size());
    }
    return copy_out(std::string(tmp.data(), n), buf, cap);
}

const char * hd_system_info(void) {
    init_once();
    return llama_print_system_info();
}

}  // extern "C"
