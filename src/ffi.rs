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

const ERRLEN: usize = 8192;

fn err_string(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

fn cstr(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|_| "string contains a NUL byte".to_string())
}

#[derive(Clone, Debug)]
pub struct Info {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub n_params: u64,
    pub size_bytes: u64,
    pub desc: String,
    /// Per-head K and V widths (from GGUF metadata, else n_embd / n_head).
    pub head_k: usize,
    pub head_v: usize,
}

/// A loaded GGUF model. llama.cpp models are safe to share across threads;
/// each `eval` builds its own context.
pub struct Model {
    ptr: *mut c_void,
    pub info: Info,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

/// One named activation to read and/or overwrite during `eval`.
pub struct Hook<'a> {
    pub name: String,
    pub capture: Option<&'a mut [f32]>,
    pub patch: Option<&'a [f32]>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Model, String> {
        // Check the magic ourselves so llama.cpp's error (which echoes the first bytes) never
        // sees a non-GGUF file.
        let mut magic = [0u8; 4];
        std::fs::File::open(path)
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if &magic != b"GGUF" {
            return Err(format!("{} is not a GGUF file", path.display()));
        }
        let p = cstr(&path.to_string_lossy())?;
        let mut err = vec![0 as c_char; ERRLEN];
        let ptr = unsafe { hd_model_load(p.as_ptr(), err.as_mut_ptr(), ERRLEN as i32) };
        if ptr.is_null() {
            return Err(err_string(&err));
        }
        let mut hi = HdInfo { n_layer: 0, n_embd: 0, n_vocab: 0, n_ctx_train: 0, n_params: 0, size_bytes: 0, desc: [0; 128], n_head: 0 };
        unsafe { hd_model_info(ptr, &mut hi) };
        let info = Info {
            n_layer: hi.n_layer as usize,
            n_embd: hi.n_embd as usize,
            n_vocab: hi.n_vocab as usize,
            n_ctx_train: hi.n_ctx_train as usize,
            n_params: hi.n_params,
            size_bytes: hi.size_bytes,
            desc: err_string(&hi.desc),
            head_k: 0,
            head_v: 0,
        };
        let mut m = Model { ptr, info };
        let fallback = m.info.n_embd / (hi.n_head.max(1) as usize);
        let arch = m.meta("general.architecture").unwrap_or_default();
        let dim = |k: &str| m.meta(&format!("{arch}.attention.{k}")).and_then(|v| v.parse().ok()).unwrap_or(fallback);
        (m.info.head_k, m.info.head_v) = (dim("key_length"), dim("value_length"));
        Ok(m)
    }

    /// Reject settings that would trip a ggml assert (which aborts the whole process).
    pub fn check_cfg(&self, cfg: &Cfg) -> Result<(), String> {
        for (t, dim, which) in [(cfg.type_k, self.info.head_k, "K"), (cfg.type_v, self.info.head_v, "V")] {
            let name = CACHE_TYPES.iter().find(|c| c.1 == t).map(|c| c.0).ok_or_else(|| format!("unsupported {which}-cache type {t}"))?;
            let block = if matches!(name, "f32" | "f16" | "bf16") { 1 } else { 32 };
            if dim % block != 0 {
                return Err(format!(
                    "{which}-cache type {name} needs a head size divisible by {block}; this model's {which} head size is {dim}"
                ));
            }
        }
        Ok(())
    }

    pub fn meta(&self, key: &str) -> Option<String> {
        let k = cstr(key).ok()?;
        let mut buf = vec![0 as c_char; 512];
        let n = unsafe { hd_meta(self.ptr, k.as_ptr(), buf.as_mut_ptr(), buf.len() as i32) };
        (n >= 0).then(|| err_string(&buf))
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>, String> {
        // parse_special=true so chat-template control tokens become single tokens.
        let mut out = vec![0i32; text.len() + 16];
        let call = |out: &mut Vec<i32>| unsafe {
            hd_tokenize(
                self.ptr,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                add_special as i32,
                1,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        let mut n = call(&mut out);
        if n < 0 {
            out.resize((-n) as usize, 0);
            n = call(&mut out);
        }
        if n < 0 {
            return Err("tokenization failed".into());
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Token text as raw bytes (a piece can be a partial UTF-8 sequence).
    pub fn piece(&self, tok: i32) -> Vec<u8> {
        let mut buf = vec![0 as c_char; 64];
        let mut n = unsafe { hd_piece(self.ptr, tok, buf.as_mut_ptr(), buf.len() as i32) };
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe { hd_piece(self.ptr, tok, buf.as_mut_ptr(), buf.len() as i32) };
        }
        buf[..n.max(0) as usize].iter().map(|&c| c as u8).collect()
    }

    pub fn is_eog(&self, tok: i32) -> bool {
        unsafe { hd_is_eog(self.ptr, tok) != 0 }
    }

    pub fn same_vocab(&self, other: &Model) -> bool {
        unsafe { hd_vocab_equal(self.ptr, other.ptr) != 0 }
    }

    /// The prompt wrapped in the model's built-in chat template (None if it has none).
    pub fn chat(&self, user: &str) -> Option<String> {
        let u = cstr(user).ok()?;
        let mut buf = vec![0 as c_char; user.len() * 2 + 4096];
        let mut n = unsafe { hd_chat(self.ptr, u.as_ptr(), buf.as_mut_ptr(), buf.len() as i32) };
        if n == i32::MIN {
            return None;
        }
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe { hd_chat(self.ptr, u.as_ptr(), buf.as_mut_ptr(), buf.len() as i32) };
        }
        (n >= 0).then(|| err_string(&buf))
    }

    /// Teacher-forced pass: logits for every position (`n_tok * n_vocab`), with optional hooks.
    pub fn eval(&self, cfg: &Cfg, toks: &[i32], hooks: &mut [Hook]) -> Result<Vec<f32>, String> {
        let mut logits = vec![0f32; toks.len() * self.info.n_vocab];
        self.eval_into(cfg, toks, hooks, &mut logits)?;
        Ok(logits)
    }

    pub fn eval_into(&self, cfg: &Cfg, toks: &[i32], hooks: &mut [Hook], logits: &mut [f32]) -> Result<(), String> {
        self.check_cfg(cfg)?;
        let n = toks.len();
        assert_eq!(logits.len(), n * self.info.n_vocab);
        let width = self.info.n_embd;
        for h in hooks.iter() {
            let ok = h.capture.as_ref().map_or(true, |c| c.len() == n * width) && h.patch.map_or(true, |p| p.len() == n * width);
            if !ok {
                return Err(format!("hook '{}' buffer size mismatch", h.name));
            }
        }
        let names: Vec<CString> = hooks.iter().map(|h| cstr(&h.name)).collect::<Result<_, _>>()?;
        let name_ptrs: Vec<*const c_char> = names.iter().map(|c| c.as_ptr()).collect();
        let outs: Vec<*mut f32> = hooks.iter_mut().map(|h| h.capture.as_mut().map_or(std::ptr::null_mut(), |c| c.as_mut_ptr())).collect();
        let srcs: Vec<*const f32> = hooks.iter().map(|h| h.patch.map_or(std::ptr::null(), |p| p.as_ptr())).collect();
        let hh = HdHooks { n: hooks.len() as i32, names: name_ptrs.as_ptr(), out: outs.as_ptr(), src: srcs.as_ptr(), width: width as i32 };
        let mut err = vec![0 as c_char; ERRLEN];
        let rc = unsafe { hd_eval(self.ptr, cfg, toks.as_ptr(), n as i32, &hh, logits.as_mut_ptr(), err.as_mut_ptr(), ERRLEN as i32) };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(())
    }

    /// Whether llama.cpp actually builds a flash-attention graph for these settings
    /// ("auto" is resolved internally and not exposed by llama.h).
    pub fn flash_attn_used(&self, cfg: &Cfg, tok: i32) -> Result<bool, String> {
        self.check_cfg(cfg)?;
        let mut err = vec![0 as c_char; ERRLEN];
        match unsafe { hd_flash_attn_used(self.ptr, cfg, tok, err.as_mut_ptr(), ERRLEN as i32) } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(err_string(&err)),
        }
    }

    /// Greedy incremental decoding (temperature 0), stopping at end-of-generation.
    pub fn generate(&self, cfg: &Cfg, prompt: &[i32], n_gen: usize) -> Result<Vec<i32>, String> {
        self.check_cfg(cfg)?;
        if prompt.is_empty() {
            return Err("prompt is empty".into());
        }
        let mut out = vec![0i32; n_gen];
        let mut err = vec![0 as c_char; ERRLEN];
        let n = unsafe {
            hd_generate(
                self.ptr,
                cfg,
                prompt.as_ptr(),
                prompt.len() as i32,
                n_gen as i32,
                out.as_mut_ptr(),
                err.as_mut_ptr(),
                ERRLEN as i32,
            )
        };
        if n < 0 {
            return Err(err_string(&err));
        }
        out.truncate(n as usize);
        Ok(out)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { hd_model_free(self.ptr) }
    }
}

pub fn system_info() -> String {
    unsafe { CStr::from_ptr(hd_system_info()) }.to_string_lossy().trim().to_string()
}
