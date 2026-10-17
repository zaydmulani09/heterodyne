//! The differential experiment: two arms, identical tokens, every number from llama.cpp.
use crate::ffi::{cache_type, Cfg, Hook, Model};
use crate::stats::{self, PosStat, Summary};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One side of the comparison: a GGUF file plus the runtime settings that can change its numbers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Arm {
    pub model: String,
    #[serde(default = "Arm::default_threads")]
    pub threads: i32,
    /// Physical batch size; 0 = the whole chunk in one ubatch.
    #[serde(default)]
    pub ubatch: i32,
    /// "auto" | "on" | "off"
    #[serde(default = "d_auto")]
    pub flash_attn: String,
    #[serde(default = "d_f16")]
    pub cache_k: String,
    #[serde(default = "d_f16")]
    pub cache_v: String,
}

impl Arm {
    pub fn default_threads() -> i32 {
        std::thread::available_parallelism().map_or(4, |n| n.get().min(8) as i32)
    }
}
fn d_auto() -> String {
    "auto".into()
}
fn d_f16() -> String {
    "f16".into()
}
fn t() -> bool {
    true
}

impl Arm {
    pub fn cfg(&self) -> Result<Cfg, String> {
        let fa = match self.flash_attn.as_str() {
            "auto" => -1,
            "off" => 0,
            "on" => 1,
            o => return Err(format!("flash_attn must be auto|on|off, got '{o}'")),
        };
        let kt = |n: &str| cache_type(n).ok_or_else(|| format!("unknown cache type '{n}'"));
        if !(1..=256).contains(&self.threads) {
            return Err(format!("threads must be 1 to 256, got {}", self.threads));
        }
        if !(0..=16384).contains(&self.ubatch) {
            return Err(format!("ubatch must be 0 to 16384, got {}", self.ubatch));
        }
        Ok(Cfg { n_threads: self.threads, n_ubatch: self.ubatch, flash_attn: fa, type_k: kt(&self.cache_k)?, type_v: kt(&self.cache_v)? })
    }

    /// Settings in which `other` differs from `self`, by name.
    fn differences(&self, other: &Arm) -> Vec<&'static str> {
        let mut d = vec![];
        if self.model != other.model {
            d.push("model")
        }
        if self.ubatch != other.ubatch {
            d.push("ubatch")
        }
        if self.flash_attn != other.flash_attn {
            d.push("flash_attn")
        }
        if self.cache_k != other.cache_k {
            d.push("cache_k")
        }
        if self.cache_v != other.cache_v {
            d.push("cache_v")
        }
        if self.threads != other.threads {
            d.push("threads")
        }
        d
    }

    fn with(&self, other: &Arm, factor: &str) -> Arm {
        let mut a = self.clone();
        match factor {
            "model" => a.model = other.model.clone(),
            "ubatch" => a.ubatch = other.ubatch,
            "flash_attn" => a.flash_attn = other.flash_attn.clone(),
            "cache_k" => a.cache_k = other.cache_k.clone(),
            "cache_v" => a.cache_v = other.cache_v.clone(),
            "threads" => a.threads = other.threads,
            _ => unreachable!(),
        }
        a
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Input {
    /// Score fixed text (split into chunks).
    Text { text: String },
    /// Greedy-generate from a prompt with both arms, find the fork, then score A's continuation.
    Generate {
        prompt: String,
        #[serde(default = "t")]
        chat: bool,
        #[serde(default = "d_new")]
        max_new: usize,
    },
}
fn d_new() -> usize {
    96
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Request {
    pub a: Arm,
    pub b: Arm,
    pub input: Input,
    #[serde(default = "d_chunk")]
    pub chunk_tokens: usize,
    #[serde(default = "d_max_chunks")]
    pub max_chunks: usize,
    /// Capture every layer's residual stream in both arms.
    #[serde(default = "t")]
    pub layers: bool,
    /// Activation transplant per layer (denoise B with A's residual, noise A with B's).
    #[serde(default)]
    pub repair: bool,
    /// Transplant every n-th layer (the last layer is always included).
    #[serde(default = "d_one")]
    pub repair_stride: usize,
    /// Factorial attribution over the settings that differ between arms.
    #[serde(default)]
    pub attribution: bool,
    /// Reference perturbation: arm A again with its ubatch split differently.
    #[serde(default = "t")]
    pub floor: bool,
    #[serde(default = "d_topk")]
    pub top_k: usize,
    /// Leading positions of each chunk that are shown but left out of every total: predictions
    /// from a handful of tokens of context are erratic and would dominate the averages.
    #[serde(default = "d_skip")]
    pub skip_first: usize,
}
fn d_skip() -> usize {
    16
}
fn d_chunk() -> usize {
    256
}
fn d_max_chunks() -> usize {
    8
}
fn d_one() -> usize {
    1
}
fn d_topk() -> usize {
    8
}

impl Request {
    /// Bounds on everything a caller controls, checked before any allocation or llama.cpp call.
    pub fn validate(&self) -> Result<(), String> {
        self.a.cfg()?;
        self.b.cfg()?;
        let bound = |name: &str, v: usize, lo: usize, hi: usize| {
            if (lo..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("{name} must be {lo} to {hi}, got {v}"))
            }
        };
        bound("tokens per chunk", self.chunk_tokens, 2, 4096)?;
        bound("max chunks", self.max_chunks, 1, 256)?;
        bound("top_k", self.top_k, 0, 64)?;
        bound("repair stride", self.repair_stride, 1, 1024)?;
        if self.skip_first + 2 > self.chunk_tokens {
            return Err(format!("unscored leading tokens ({}) must leave room in a {}-token chunk", self.skip_first, self.chunk_tokens));
        }
        match &self.input {
            Input::Text { text } => bound("text length (bytes)", text.len(), 1, 8 << 20),
            Input::Generate { prompt, max_new, .. } => {
                bound("prompt length (bytes)", prompt.len(), 1, 64 << 10)?;
                bound("tokens to generate", *max_new, 1, 4096)
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub tokens: Vec<i32>,
    pub pos: Vec<PosStat>,
    /// KL(A || A') per position, A' = arm A with a different ubatch split.
    pub floor_kl: Option<Vec<f32>>,
    /// The ubatch size A' used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor_ubatch: Option<i32>,
    /// [layer][pos] ||h_B - h_A|| / ||h_A|| of `l_out-<layer>`.
    pub layer_err: Option<Vec<Vec<f32>>>,
    /// Layers transplanted (indices into the model's layers).
    pub repair_layers: Option<Vec<usize>>,
    /// [i][pos] KL(A || B with A's l_out-<repair_layers[i]> written in).
    pub repair_denoise: Option<Vec<Vec<f32>>>,
    /// [i][pos] KL(A || A with B's l_out-<repair_layers[i]> written in).
    pub repair_noise: Option<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Generation {
    pub prompt_tokens: Vec<i32>,
    pub a: Vec<i32>,
    pub b: Vec<i32>,
    /// Index into the generated tokens where the arms first disagree.
    pub fork: Option<usize>,
    pub a_text: String,
    pub b_text: String,
    /// Whether batched teacher forcing reproduces both arms' choices at the fork
    /// (incremental decoding and batched scoring can disagree when choices are close).
    pub fork_reproduced: Option<bool>,
    /// Under A's teacher-forced distribution at the fork: log p(A's token) - log p(B's token).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_gap_a: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cell {
    /// Factors switched from A's value to B's value.
    pub switched: Vec<String>,
    pub mean_kl: f64,
    pub flip_rate: f64,
    /// Set when llama.cpp rejects this combination (e.g. a quantized V cache without flash attention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Attribution {
    pub factors: Vec<String>,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Results {
    pub summary: Summary,
    pub floor: Option<Summary>,
    /// Per layer, mean over all positions.
    pub layer_err_mean: Option<Vec<f32>>,
    pub repair_layers: Option<Vec<usize>>,
    pub repair_denoise_mean: Option<Vec<f32>>,
    pub repair_noise_mean: Option<Vec<f32>>,
    pub chunks: Vec<Chunk>,
    pub generation: Option<Generation>,
    pub attribution: Option<Attribution>,
    /// Whether each arm's graph actually used flash attention ("auto" resolved by llama.cpp).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash_attn_used: Option<[bool; 2]>,
    /// Text for every token id that appears in the results.
    pub vocab: BTreeMap<i32, String>,
    pub warnings: Vec<String>,
}

/// Loaded models, keyed by path + size + mtime so a file rewritten in place is reloaded.
#[derive(Default)]
pub struct Models(HashMap<(String, u64, u128), Arc<Model>>);

impl Models {
    pub fn get(&mut self, path: &str) -> Result<Arc<Model>, String> {
        let md = std::fs::metadata(path).map_err(|_| format!("model file not found: {path}"))?;
        if !md.is_file() {
            return Err(format!("not a file: {path}"));
        }
        let mtime = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        let key = (path.to_string(), md.len(), mtime);
        if let Some(m) = self.0.get(&key) {
            return Ok(m.clone());
        }
        if self.0.len() >= 4 {
            self.0.clear(); // ponytail: drop all; LRU if people juggle many models
        }
        let m = Arc::new(Model::load(Path::new(path))?);
        self.0.insert(key, m.clone());
        Ok(m)
    }
}

pub struct Progress<'a> {
    pub done: usize,
    pub total: usize,
    pub stage: &'a str,
}

fn l_out(l: usize) -> String {
    format!("l_out-{l}")
}
