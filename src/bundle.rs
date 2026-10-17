//! Experiment files: request + provenance + results, with an integrity hash and replay.
use crate::analysis::{self, Models, Request, Results};
use crate::ffi;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA: &str = "heterodyne/experiment@1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelId {
    pub path: String,
    pub file_name: String,
    pub sha256: String,
    pub file_bytes: u64,
    pub desc: String,
    pub general_name: Option<String>,
    pub n_params: u64,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Provenance {
    pub heterodyne: String,
    pub llama_cpp_commit: String,
    pub build: String,
    /// llama.cpp's own CPU feature report; numerics depend on these code paths.
    pub system_info: String,
    pub os: String,
    pub arch: String,
    pub a: ModelId,
    pub b: ModelId,
}

/// Free text written by the person running the experiment. Kept apart from measured data
/// and outside the integrity hash: interpretation is not evidence.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Notes {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub hypothesis: String,
    #[serde(default)]
    pub interpretation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Experiment {
    pub schema: String,
    pub created_unix: u64,
    #[serde(default)]
    pub notes: Notes,
    pub request: Request,
    pub provenance: Provenance,
    pub elapsed_ms: u64,
    pub results: Results,
    /// sha256 of the canonical JSON of `request` (model paths blanked), `provenance.a/b.sha256` and `results`.
    pub integrity: String,
}

static HASHES: Mutex<Option<HashMap<(String, u64, u128), String>>> = Mutex::new(None);

/// sha256 of a file, memoized by (path, size, mtime) for the life of the process.
pub fn file_sha256(path: &Path) -> Result<String, String> {
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mtime = md.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
    let key = (path.to_string_lossy().into_owned(), md.len(), mtime);
    if let Some(h) = HASHES.lock().unwrap().get_or_insert_with(HashMap::new).get(&key) {
        return Ok(h.clone());
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let h = hex(&hasher.finalize());
    HASHES.lock().unwrap().as_mut().unwrap().insert(key, h.clone());
    Ok(h)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn model_id(models: &mut Models, path: &str) -> Result<ModelId, String> {
    let m = models.get(path)?;
    let p = Path::new(path);
    Ok(ModelId {
        path: path.to_string(),
        file_name: p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        sha256: file_sha256(p)?,
        file_bytes: std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
        desc: m.info.desc.clone(),
        general_name: m.meta("general.name"),
        n_params: m.info.n_params,
        n_layer: m.info.n_layer,
        n_embd: m.info.n_embd,
        n_vocab: m.info.n_vocab,
    })
}

pub fn provenance(models: &mut Models, req: &Request) -> Result<Provenance, String> {
    Ok(Provenance {
        heterodyne: env!("CARGO_PKG_VERSION").into(),
        llama_cpp_commit: env!("HD_LLAMA_COMMIT").into(),
        build: env!("HD_BUILD_FLAGS").into(),
        system_info: ffi::system_info(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        a: model_id(models, &req.a.model)?,
        b: model_id(models, &req.b.model)?,
    })
}

/// An unkeyed checksum: it detects edits and corruption, it is not a signature.
/// Model *locations* are left out (moving a file must not invalidate an experiment); everything
/// else in the request, the provenance (model hashes, llama.cpp build, CPU features) and the
/// results is in.
pub fn integrity(req: &Request, prov: &Provenance, res: &Results) -> String {
    let (mut req, mut prov) = (req.clone(), prov.clone());
    req.a.model.clear();
    req.b.model.clear();
    prov.a.path.clear();
    prov.b.path.clear();
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(&req).unwrap());
    h.update(serde_json::to_vec(&prov).unwrap());
    h.update(serde_json::to_vec(res).unwrap());
    hex(&h.finalize())
}

/// Run a request end to end and package it.
pub fn record(
    req: Request,
    notes: Notes,
    models: &mut Models,
    cancel: &AtomicBool,
    progress: &dyn Fn(analysis::Progress),
) -> Result<Experiment, String> {
    let t0 = std::time::Instant::now();
    let provenance = provenance(models, &req)?;
    let results = analysis::run(&req, models, cancel, progress)?;
    let integrity = integrity(&req, &provenance, &results);
    Ok(Experiment {
        schema: SCHEMA.into(),
        created_unix: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        notes,
        request: req,
        provenance,
        elapsed_ms: t0.elapsed().as_millis() as u64,
        results,
        integrity,
    })
}

/// Write an experiment into `dir` as `<created>-<title slug>.json`; returns the file name.
pub fn save_to(dir: &Path, e: &Experiment) -> Result<String, String> {
    let slug: String = e.notes.title.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    let slug: String = slug.trim_matches('-').chars().take(40).collect();
    let name = format!("{}-{}.json", e.created_unix, if slug.is_empty() { "experiment" } else { &slug });
    std::fs::write(dir.join(&name), serde_json::to_string(e).unwrap()).map_err(|err| err.to_string())?;
    Ok(name)
}

pub fn load(path: &Path) -> Result<Experiment, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let e: Experiment = serde_json::from_str(&s).map_err(|e| format!("not a heterodyne experiment: {e}"))?;
    if e.schema != SCHEMA {
        return Err(format!("unsupported schema '{}' (this build reads {SCHEMA})", e.schema));
    }
    Ok(e)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyReport {
    /// The stored results still hash to the stored integrity value.
    pub integrity_ok: bool,
    /// "exact" | "approximate" | "not-reproducible"
    pub outcome: String,
    pub same_llama_build: bool,
    pub same_system_info: bool,
    pub max_abs_kl_diff: f64,
    pub top1_changes: usize,
    /// Largest absolute difference per stored array (per-token KL, floor, layer map, transplants).
    #[serde(default)]
    pub max_abs_diff: std::collections::BTreeMap<String, f64>,
    pub notes: Vec<String>,
}

fn max_diff_rows(a: &Option<Vec<Vec<f32>>>, b: &Option<Vec<Vec<f32>>>) -> Option<f64> {
    let (a, b) = (a.as_ref()?, b.as_ref()?);
    Some(a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (*p as f64 - *q as f64).abs())).fold(0.0, f64::max))
}

/// Locate a model by stored path, else by file name under `search` dirs, checked by sha256.
fn find_model(id: &ModelId, search: &[PathBuf]) -> Result<String, String> {
    let mut candidates = vec![PathBuf::from(&id.path)];
    for d in search {
        candidates.push(d.join(&id.file_name));
        // One level deep, like the model list in the UI.
        for sub in std::fs::read_dir(d).into_iter().flatten().flatten().filter(|e| e.path().is_dir()) {
            candidates.push(sub.path().join(&id.file_name));
        }
    }
    for c in candidates {
        if c.is_file() && file_sha256(&c)? == id.sha256 {
            return Ok(c.to_string_lossy().into_owned());
        }
    }
    Err(format!("no file with sha256 {} found for '{}' (pass --search DIR)", &id.sha256[..12], id.file_name))
}

/// Re-run the stored request and compare every number.
pub fn verify(
    e: &Experiment,
    search: &[PathBuf],
    models: &mut Models,
    progress: &dyn Fn(analysis::Progress),
) -> Result<VerifyReport, String> {
    let integrity_ok = integrity(&e.request, &e.provenance, &e.results) == e.integrity;
    let mut req = e.request.clone();
    req.a.model = find_model(&e.provenance.a, search)?;
    req.b.model = find_model(&e.provenance.b, search)?;
    let now = provenance(models, &req)?;
    let res = analysis::run(&req, models, &AtomicBool::new(false), progress)?;
    let mut notes = vec![];
    if !integrity_ok {
        notes.push("stored results do not match their integrity hash: the file was edited or corrupted".into());
    }
    let same_llama_build = now.llama_cpp_commit == e.provenance.llama_cpp_commit && now.build == e.provenance.build;
    let same_system_info = now.system_info == e.provenance.system_info;
    if !same_llama_build {
        notes.push(format!(
            "llama.cpp build differs: recorded {} ({}), now {} ({})",
            e.provenance.llama_cpp_commit, e.provenance.build, now.llama_cpp_commit, now.build
        ));
    }
    if !same_system_info {
        notes.push("CPU feature set differs; kernels with different summation order can change low-order bits".into());
    }
    let (mut max_abs, mut flips, mut shape_ok) = (0f64, 0usize, res.chunks.len() == e.results.chunks.len());
    let mut by_array = std::collections::BTreeMap::new();
    let mut note_max = |name: &str, d: Option<f64>| {
        if let Some(d) = d {
            let m = by_array.entry(name.to_string()).or_insert(0f64);
            *m = m.max(d);
        }
    };
    for (c0, c1) in e.results.chunks.iter().zip(&res.chunks) {
        shape_ok &= c0.tokens == c1.tokens;
        for (p0, p1) in c0.pos.iter().zip(&c1.pos) {
            max_abs = max_abs.max((p0.kl - p1.kl).abs());
            flips += (p0.top_a != p1.top_a || p0.top_b != p1.top_b) as usize;
        }
        note_max("kl", Some(max_abs));
        note_max("floor", max_diff_rows(&c0.floor_kl.clone().map(|v| vec![v]), &c1.floor_kl.clone().map(|v| vec![v])));
        note_max("layer map", max_diff_rows(&c0.layer_err, &c1.layer_err));
        note_max("transplant A into B", max_diff_rows(&c0.repair_denoise, &c1.repair_denoise));
        note_max("transplant B into A", max_diff_rows(&c0.repair_noise, &c1.repair_noise));
    }
    let outcome = if !shape_ok {
        notes.push("token sequences differ (tokenizer or generation changed)".into());
        "not-reproducible"
    } else if res == e.results {
        "exact"
    } else {
        "approximate"
    };
    Ok(VerifyReport {
        integrity_ok,
        outcome: outcome.into(),
        same_llama_build,
        same_system_info,
        max_abs_kl_diff: max_abs,
        top1_changes: flips,
        max_abs_diff: by_array,
        notes,
    })
}
