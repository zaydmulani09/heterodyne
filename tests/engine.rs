//! Numerical invariants of the llama.cpp hook path, on a real (tiny) model.
//! Fixture: tests/fixtures/stories260K.gguf, or $HETERODYNE_TEST_MODEL.
//! Set HETERODYNE_REQUIRE_FIXTURE=1 (CI does) to fail instead of skip when it is missing.
use heterodyne::analysis::{self, Arm, Input, Models, Request};
use heterodyne::bundle::{self, Notes};
use heterodyne::ffi::{self, Cfg, Hook, Model};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

fn fixture() -> Option<PathBuf> {
    let p = std::env::var_os("HETERODYNE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stories260K.gguf"));
    if p.is_file() {
        return Some(p);
    }
    if std::env::var_os("HETERODYNE_REQUIRE_FIXTURE").is_some() {
        panic!("test model missing: {}", p.display());
    }
    eprintln!("skipping: no test model at {}", p.display());
    None
}

/// A Q8_0 copy of the fixture, made once with llama.cpp's own quantizer.
fn quantized() -> &'static PathBuf {
    static Q: OnceLock<PathBuf> = OnceLock::new();
    Q.get_or_init(|| {
        let src = fixture().unwrap();
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("heterodyne-test-q8_0-{}.gguf", std::process::id()));
        ffi::quantize(&src, &out, 7 /* LLAMA_FTYPE_MOSTLY_Q8_0 */, 2).unwrap();
        out
    })
}

const TEXT: &str = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her dog. One day, she found a big red ball under a tree.";

fn cfg() -> Cfg {
    Cfg { n_threads: 2, n_ubatch: 0, flash_attn: 0, type_k: 1, type_v: 1 }
}

fn toks(m: &Model) -> Vec<i32> {
    m.tokenize(TEXT, true).unwrap()
}

fn capture(m: &Model, c: &Cfg, t: &[i32]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let w = m.info.n_embd;
    let mut bufs: Vec<Vec<f32>> = (0..m.info.n_layer).map(|_| vec![0f32; t.len() * w]).collect();
    let mut hooks: Vec<Hook> =
        bufs.iter_mut().enumerate().map(|(l, b)| Hook { name: format!("l_out-{l}"), capture: Some(b), patch: None }).collect();
    let logits = m.eval(c, t, &mut hooks).unwrap();
    drop(hooks);
    (logits, bufs)
}

fn patch(m: &Model, c: &Cfg, t: &[i32], layer: usize, src: &[f32]) -> Vec<f32> {
    m.eval(c, t, &mut [Hook { name: format!("l_out-{layer}"), capture: None, patch: Some(src) }]).unwrap()
}

#[test]
fn observing_does_not_perturb() {
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let plain = m.eval(&cfg(), &t, &mut []).unwrap();
    let (hooked, caps) = capture(&m, &cfg(), &t);
    assert!(plain == hooked, "capturing every l_out changed the logits");
    assert!(caps.iter().all(|c| c.iter().any(|&x| x != 0.0)), "a capture buffer stayed empty");
}

#[test]
fn self_patch_is_identity_and_writes_reach_downstream() {
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let (base, caps) = capture(&m, &cfg(), &t);
    let mid = m.info.n_layer / 2;
    assert!(patch(&m, &cfg(), &t, mid, &caps[mid]) == base, "writing a tensor's own values back changed the logits");
    let zeros = vec![0f32; caps[mid].len()];
    assert!(patch(&m, &cfg(), &t, mid, &zeros) != base, "zeroing a residual had no effect: writes are not reaching downstream ops");
}

#[test]
fn last_layer_transplant_restores_reference_when_weights_match() {
    // Same weights, different K-cache type: everything after the last residual is identical,
    // so writing A's final residual into B must reproduce A's logits bit for bit.
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let a = cfg();
    let b = Cfg { type_k: 30, ..cfg() }; // bf16 K cache (fixture head dim 8 rules out q8_0 blocks)
    let (la, caps) = capture(&m, &a, &t);
    let lb = m.eval(&b, &t, &mut []).unwrap();
    assert!(la != lb, "bf16 K cache produced identical logits; the test would prove nothing");
    let last = m.info.n_layer - 1;
    assert!(patch(&m, &b, &t, last, &caps[last]) == la);
}

#[test]
fn self_patch_is_identity_across_ubatches() {
    // Writes land in the right rows when a chunk is computed in several ubatches.
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let c = Cfg { n_ubatch: 7, ..cfg() };
    let (base, caps) = capture(&m, &c, &t);
    let mid = m.info.n_layer / 2;
    assert!(patch(&m, &c, &t, mid, &caps[mid]) == base);
}
