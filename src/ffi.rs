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
