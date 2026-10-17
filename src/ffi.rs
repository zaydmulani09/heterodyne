//! Safe wrapper over the C ABI in `shim.cpp`.
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

#[repr(C)]
struct HdInfo {
    n_layer: i32,
    n_embd: i32,
    n_vocab: i32,
    n_ctx_train: i32,
    n_params: u64,
    size_bytes: u64,
    desc: [c_char; 128],
    n_head: i32,
}

/// Runtime settings for one llama.cpp context (mirrors `hd_cfg`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfg {
    pub n_threads: i32,
    /// 0 = the whole sequence is one ubatch.
    pub n_ubatch: i32,
    /// -1 auto, 0 off, 1 on.
    pub flash_attn: i32,
    pub type_k: i32,
    pub type_v: i32,
}

#[repr(C)]
struct HdHooks {
    n: i32,
    names: *const *const c_char,
    out: *const *mut f32,
    src: *const *const f32,
    width: i32,
}

extern "C" {
    fn hd_model_load(path: *const c_char, err: *mut c_char, errlen: i32) -> *mut c_void;
    fn hd_model_free(m: *mut c_void);
    fn hd_model_info(m: *mut c_void, out: *mut HdInfo);
    fn hd_meta(m: *mut c_void, key: *const c_char, buf: *mut c_char, cap: i32) -> i32;
    fn hd_tokenize(m: *mut c_void, text: *const c_char, len: i32, add_special: i32, parse_special: i32, out: *mut i32, cap: i32) -> i32;
    fn hd_piece(m: *mut c_void, tok: i32, buf: *mut c_char, cap: i32) -> i32;
    fn hd_is_eog(m: *mut c_void, tok: i32) -> i32;
    fn hd_vocab_equal(a: *mut c_void, b: *mut c_void) -> i32;
    fn hd_chat(m: *mut c_void, user: *const c_char, buf: *mut c_char, cap: i32) -> i32;
    fn hd_system_info() -> *const c_char;
    fn hd_eval(
        m: *mut c_void,
        cfg: *const Cfg,
        toks: *const i32,
        n: i32,
        hooks: *const HdHooks,
        logits: *mut f32,
        err: *mut c_char,
        errlen: i32,
    ) -> i32;
    fn hd_generate(
        m: *mut c_void,
        cfg: *const Cfg,
        prompt: *const i32,
        n_prompt: i32,
        n_gen: i32,
        out: *mut i32,
        err: *mut c_char,
        errlen: i32,
    ) -> i32;
    fn hd_flash_attn_used(m: *mut c_void, cfg: *const Cfg, tok: i32, err: *mut c_char, errlen: i32) -> i32;
    fn hd_quantize(input: *const c_char, output: *const c_char, ftype: i32, nthread: i32) -> i32;
}
