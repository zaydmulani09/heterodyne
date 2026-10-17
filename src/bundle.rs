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
