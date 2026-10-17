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
