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

/// Every layer's `<kind>-<layer>` tensor: `l_out` (residual after the layer) or `ffn_inp` (after its attention).
fn capture(m: &Model, c: &Cfg, t: &[i32], kind: &str) -> (Vec<f32>, Vec<Vec<f32>>) {
    let w = m.info.n_embd;
    let mut bufs: Vec<Vec<f32>> = (0..m.info.n_layer).map(|_| vec![0f32; t.len() * w]).collect();
    let mut hooks: Vec<Hook> =
        bufs.iter_mut().enumerate().map(|(l, b)| Hook { name: format!("{kind}-{l}"), capture: Some(b), patch: None }).collect();
    let logits = m.eval(c, t, &mut hooks).unwrap();
    drop(hooks);
    (logits, bufs)
}

fn patch(m: &Model, c: &Cfg, t: &[i32], kind: &str, layer: usize, src: &[f32]) -> Vec<f32> {
    m.eval(c, t, &mut [Hook { name: format!("{kind}-{layer}"), capture: None, patch: Some(src) }]).unwrap()
}

#[test]
fn observing_does_not_perturb() {
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let plain = m.eval(&cfg(), &t, &mut []).unwrap();
    let (hooked, caps) = capture(&m, &cfg(), &t, "l_out");
    assert!(plain == hooked, "capturing every l_out changed the logits");
    assert!(caps.iter().all(|c| c.iter().any(|&x| x != 0.0)), "a capture buffer stayed empty");
}

#[test]
fn self_patch_is_identity_and_writes_reach_downstream() {
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let (base, caps) = capture(&m, &cfg(), &t, "l_out");
    let mid = m.info.n_layer / 2;
    assert!(patch(&m, &cfg(), &t, "l_out", mid, &caps[mid]) == base, "writing a tensor's own values back changed the logits");
    let zeros = vec![0f32; caps[mid].len()];
    assert!(
        patch(&m, &cfg(), &t, "l_out", mid, &zeros) != base,
        "zeroing a residual had no effect: writes are not reaching downstream ops"
    );
}

#[test]
fn post_attention_residual_is_observed_and_patched_exactly() {
    // `ffn_inp-<l>`: the residual between a layer's attention and MLP blocks, the split point
    // for the attention/MLP shares.
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let plain = m.eval(&cfg(), &t, &mut []).unwrap();
    let (base, caps) = capture(&m, &cfg(), &t, "ffn_inp");
    assert!(plain == base, "capturing every ffn_inp changed the logits");
    for l in [0, m.info.n_layer - 1] {
        assert!(patch(&m, &cfg(), &t, "ffn_inp", l, &caps[l]) == base, "writing ffn_inp-{l}'s own values back changed the logits");
    }
    let zeros = vec![0f32; caps[0].len()];
    assert!(patch(&m, &cfg(), &t, "ffn_inp", 0, &zeros) != base, "zeroing ffn_inp-0 had no effect");
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
    let (la, caps) = capture(&m, &a, &t, "l_out");
    let lb = m.eval(&b, &t, &mut []).unwrap();
    assert!(la != lb, "bf16 K cache produced identical logits; the test would prove nothing");
    let last = m.info.n_layer - 1;
    assert!(patch(&m, &b, &t, "l_out", last, &caps[last]) == la);
}

#[test]
fn self_patch_is_identity_across_ubatches() {
    // Writes land in the right rows when a chunk is computed in several ubatches.
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let c = Cfg { n_ubatch: 7, ..cfg() };
    let (base, caps) = capture(&m, &c, &t, "l_out");
    let mid = m.info.n_layer / 2;
    assert!(patch(&m, &c, &t, "l_out", mid, &caps[mid]) == base);
}

#[test]
fn multi_ubatch_capture_fills_every_row() {
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let t = toks(&m);
    let (_, whole) = capture(&m, &cfg(), &t, "l_out");
    let (_, split) = capture(&m, &Cfg { n_ubatch: 7, ..cfg() }, &t, "l_out");
    let w = m.info.n_embd;
    // A row placed at the wrong offset would differ by O(1). Different ubatch sizes take
    // different kernels, so small differences are expected: up to ~1e-3 on Apple silicon.
    for l in 0..m.info.n_layer {
        for (i, (ra, rb)) in whole[l].chunks(w).zip(split[l].chunks(w)).enumerate() {
            let err = heterodyne::stats::rel_err_rows(ra, rb, w)[0];
            assert!(err < 1e-2, "layer {l} row {i}: rel err {err} between ubatch splits");
        }
    }
}

fn request(a: &str, b: &str) -> Request {
    let arm =
        |m: &str| Arm { model: m.into(), threads: 2, ubatch: 0, flash_attn: "off".into(), cache_k: "f16".into(), cache_v: "f16".into() };
    Request {
        a: arm(a),
        b: arm(b),
        input: Input::Text { text: TEXT.repeat(3) },
        chunk_tokens: 48,
        max_chunks: 3,
        layers: true,
        repair: true,
        repair_stride: 1,
        components: false,
        attribution: false,
        floor: true,
        top_k: 4,
        skip_first: 4,
    }
}

#[test]
fn identical_arms_diff_to_exactly_zero() {
    let Some(p) = fixture() else { return };
    let p = p.to_string_lossy().into_owned();
    let r = analysis::run(&request(&p, &p), &mut Models::default(), &AtomicBool::new(false), &|_| {}).unwrap();
    assert_eq!(r.summary.mean_kl, 0.0);
    assert_eq!(r.summary.flips, 0);
    assert!(r.layer_err_mean.unwrap().iter().all(|&x| x == 0.0));
    assert!(r.repair_denoise_mean.unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn quantized_variant_diverges_and_transplant_repairs_it() {
    let Some(p) = fixture() else { return };
    let (a, b) = (p.to_string_lossy().into_owned(), quantized().to_string_lossy().into_owned());
    let r = analysis::run(&request(&a, &b), &mut Models::default(), &AtomicBool::new(false), &|_| {}).unwrap();
    assert!(r.summary.mean_kl > 0.0, "q8_0 should differ from f32");
    // Totals count only scored positions: each chunk's first `skip_first` are left out.
    assert_eq!(r.summary.n_positions, r.chunks.iter().map(|c| c.pos.len() - 4).sum::<usize>());
    assert!(r.chunks.len() >= 2 && r.summary.mean_kl_ci95.is_some());
    let den = r.repair_denoise_mean.unwrap();
    // Writing A's final residual into B leaves only the quantized output head's error.
    assert!(den[den.len() - 1] < r.summary.mean_kl as f32, "final-layer transplant did not reduce divergence");
    let noi = r.repair_noise_mean.unwrap();
    assert!(noi.iter().all(|&x| x >= 0.0));
}

#[test]
fn floor_stride_and_attribution_are_consistent() {
    let Some(p) = fixture() else { return };
    let p = p.to_string_lossy().into_owned();
    let mut req = request(&p, &p);
    req.b.cache_k = "bf16".into();
    req.b.ubatch = 7;
    req.b.flash_attn = "auto".into();
    req.a.flash_attn = "on".into();
    req.attribution = true;
    req.repair_stride = 2;
    let r = analysis::run(&req, &mut Models::default(), &AtomicBool::new(false), &|_| {}).unwrap();
    // auto and on resolve to the same kernel on CPU, so flash attention is not a factor.
    let fa = r.flash_attn_used.unwrap();
    assert_eq!(fa[0], fa[1]);
    let at = r.attribution.unwrap();
    assert_eq!(at.factors, vec!["ubatch", "cache_k"]);
    assert_eq!(at.cells.len(), 4);
    assert_eq!(at.cells[0].mean_kl, 0.0);
    assert_eq!(at.cells[3].mean_kl, r.summary.mean_kl);
    assert!(at.cells.iter().all(|c| c.error.is_none()));
    // The floor split is odd and recorded per chunk; transplants hit every other layer plus the last.
    for c in &r.chunks {
        let ub = c.floor_ubatch.unwrap();
        assert!(ub % 2 == 1 && (ub as usize) < c.tokens.len());
    }
    let ls = r.repair_layers.unwrap();
    assert!(ls.windows(2).all(|w| w[1] - w[0] <= 2));
    assert_eq!(*ls.last().unwrap(), r.chunks[0].layer_err.as_ref().unwrap().len() - 1);
}

#[test]
fn bad_requests_are_rejected_before_running() {
    let Some(p) = fixture() else { return };
    let p = p.to_string_lossy().into_owned();
    let mut req = request(&p, &p);
    req.a.threads = 100_000;
    assert!(analysis::run(&req, &mut Models::default(), &AtomicBool::new(false), &|_| {}).unwrap_err().contains("threads"));
    let mut req = request(&p, &p);
    req.top_k = usize::MAX;
    assert!(req.validate().unwrap_err().contains("top_k"));
    let mut req = request(&p, &p);
    req.a.model = env!("CARGO_MANIFEST_DIR").to_string() + "/Cargo.toml";
    assert!(analysis::run(&req, &mut Models::default(), &AtomicBool::new(false), &|_| {}).unwrap_err().contains("not a GGUF"));
}

#[test]
fn experiment_roundtrips_and_verifies_exactly() {
    let Some(p) = fixture() else { return };
    let (a, b) = (p.to_string_lossy().into_owned(), quantized().to_string_lossy().into_owned());
    let mut req = request(&a, &b);
    req.repair = false;
    let mut models = Models::default();
    let e = bundle::record(req, Notes::default(), &mut models, &AtomicBool::new(false), &|_| {}).unwrap();
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("heterodyne-test-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_string_pretty(&e).unwrap()).unwrap();
    let back = bundle::load(&path).unwrap();
    assert!(back == e, "JSON roundtrip changed the experiment");
    let rep = bundle::verify(&back, &[], &mut models, &|_| {}).unwrap();
    assert!(rep.integrity_ok);
    assert_eq!(rep.outcome, "exact", "{:?}", rep.notes);

    let mut tampered = back.clone();
    tampered.results.summary.mean_kl *= 0.5;
    assert_ne!(bundle::integrity(&tampered.request, &tampered.provenance, &tampered.results), tampered.integrity);

    // Moving the model files is not tampering: locations are outside the hash, identity is not.
    let mut moved = back.clone();
    moved.request.a.model = "elsewhere/model.gguf".into();
    assert_eq!(bundle::integrity(&moved.request, &moved.provenance, &moved.results), moved.integrity);
    moved.provenance.b.sha256 = "0".repeat(64);
    assert_ne!(bundle::integrity(&moved.request, &moved.provenance, &moved.results), moved.integrity);
}

#[test]
fn incompatible_cache_type_is_an_error_not_an_abort() {
    // The fixture's head size (8) is not a multiple of q8_0's 32-element block; ggml would assert.
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    if m.info.head_k % 32 == 0 {
        eprintln!("skipping: head size {} is block-compatible", m.info.head_k);
        return;
    }
    let e = m.eval(&Cfg { type_k: 8, ..cfg() }, &toks(&m), &mut []).unwrap_err();
    assert!(e.contains("divisible by 32"), "{e}");
}

#[test]
fn greedy_generation_is_deterministic() {
    let Some(p) = fixture() else { return };
    let m = Model::load(&p).unwrap();
    let prompt = m.tokenize("Once upon a time", true).unwrap();
    let g1 = m.generate(&cfg(), &prompt, 24).unwrap();
    let g2 = m.generate(&cfg(), &prompt, 24).unwrap();
    assert!(!g1.is_empty());
    assert_eq!(g1, g2);
}
