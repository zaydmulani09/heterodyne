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

/// Token text; byte-level pieces that are not valid UTF-8 on their own are shown as <XX> bytes.
fn piece(m: &Model, t: i32) -> String {
    match String::from_utf8(m.piece(t)) {
        Ok(s) => s,
        Err(e) => e
            .into_bytes()
            .iter()
            .map(|&b| if b.is_ascii_graphic() || b == b' ' { (b as char).to_string() } else { format!("<{b:02X}>") })
            .collect(),
    }
}

/// Token chunks for scoring: [special prefix] + body slice, at most `chunk` tokens each.
/// Also returns how many body tokens did not fit in `max_chunks`.
fn chunks_for(m: &Model, text: &str, chunk: usize, max_chunks: usize) -> Result<(Vec<Vec<i32>>, usize), String> {
    let prefix = m.tokenize("", true)?;
    let body = m.tokenize(text, false)?;
    let step = chunk.saturating_sub(prefix.len()).max(1);
    let out: Vec<Vec<i32>> =
        body.chunks(step).map(|c| prefix.iter().chain(c).copied().collect::<Vec<_>>()).filter(|c| c.len() >= 2).take(max_chunks).collect();
    if out.is_empty() {
        return Err("input is too short: need at least 2 tokens".into());
    }
    Ok((out, body.len().saturating_sub(step * max_chunks)))
}

/// An odd ubatch near half of `ub` (or of the chunk): odd sizes force llama.cpp's CPU kernels
/// onto their remainder paths, so this split changes summation order wherever it can.
fn floor_split(cfg_ubatch: i32, n: usize) -> i32 {
    let base = if cfg_ubatch <= 0 || cfg_ubatch as usize >= n { n as i32 } else { cfg_ubatch };
    let half = (base + 1) / 2;
    (if half % 2 == 0 { half + 1 } else { half }).clamp(1, (n as i32 - 1).max(1))
}

/// Capture buffers for every layer's residual output.
fn capture_all(m: &Model, cfg: &Cfg, toks: &[i32], n_layer: usize) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
    let w = m.info.n_embd;
    let mut bufs: Vec<Vec<f32>> = (0..n_layer).map(|_| vec![0f32; toks.len() * w]).collect();
    let mut hooks: Vec<Hook> = bufs.iter_mut().enumerate().map(|(l, b)| Hook { name: l_out(l), capture: Some(b), patch: None }).collect();
    let logits = m.eval(cfg, toks, &mut hooks)?;
    drop(hooks);
    Ok((logits, bufs))
}

fn patched(m: &Model, cfg: &Cfg, toks: &[i32], layer: usize, src: &[f32], out: &mut [f32]) -> Result<(), String> {
    let mut hooks = [Hook { name: l_out(layer), capture: None, patch: Some(src) }];
    m.eval_into(cfg, toks, &mut hooks, out)
}

pub fn run(req: &Request, models: &mut Models, cancel: &AtomicBool, progress: &dyn Fn(Progress)) -> Result<Results, String> {
    req.validate()?;
    let cfg_a = req.a.cfg()?;
    let cfg_b = req.b.cfg()?;
    let ma = models.get(&req.a.model)?;
    let mb = models.get(&req.b.model)?;
    let mut warnings = vec![];
    if ma.info.n_vocab != mb.info.n_vocab || !ma.same_vocab(&mb) {
        return Err("the two models have different vocabularies; token-level comparison needs an identical tokenizer".into());
    }
    let same_shape = ma.info.n_layer == mb.info.n_layer && ma.info.n_embd == mb.info.n_embd;
    let mut layers = (req.layers || req.repair) && same_shape;
    if (req.layers || req.repair) && !same_shape {
        warnings.push("models differ in layer count or width: layer map and transplants disabled".into());
    }
    let (n_layer, v) = (ma.info.n_layer, ma.info.n_vocab);
    let skip = req.skip_first;
    let check = || if cancel.load(Ordering::Relaxed) { Err("cancelled".to_string()) } else { Ok(()) };

    // Rough peak memory: ~4 logit buffers plus both arms' residual captures.
    let n_max = match &req.input {
        Input::Text { .. } => req.chunk_tokens,
        Input::Generate { max_new, .. } => req.chunk_tokens.max(*max_new + 512),
    };
    let gib = (4 * n_max * v * 4 + if layers { 2 * n_layer * n_max * ma.info.n_embd * 4 } else { 0 }) as f64 / (1u64 << 30) as f64;
    if gib > 12.0 {
        return Err(format!("this run would need about {gib:.1} GB of RAM for logits and captures; lower the tokens per chunk"));
    }
    if req.chunk_tokens > ma.info.n_ctx_train {
        warnings.push(format!("{} tokens per chunk exceeds the model's training context ({})", req.chunk_tokens, ma.info.n_ctx_train));
    }

    // Inputs.
    let mut generation = None;
    let seqs = match &req.input {
        Input::Text { text } => {
            let (seqs, dropped) = chunks_for(&ma, text, req.chunk_tokens, req.max_chunks)?;
            if dropped > 0 {
                warnings.push(format!("{dropped} tokens of text did not fit in {} chunks and were not scored", req.max_chunks));
            }
            seqs
        }
        Input::Generate { prompt, chat, max_new } => {
            let text = if *chat {
                ma.chat(prompt).unwrap_or_else(|| {
                    warnings.push(
                        "the model's chat template is missing or not one llama.cpp's built-in formatter recognizes; used the raw prompt"
                            .into(),
                    );
                    prompt.clone()
                })
            } else {
                prompt.clone()
            };
            let ptoks = ma.tokenize(&text, true)?;
            progress(Progress { done: 0, total: 1, stage: "generating with arm A" });
            let ga = ma.generate(&cfg_a, &ptoks, *max_new)?;
            check()?;
            progress(Progress { done: 0, total: 1, stage: "generating with arm B" });
            let gb = mb.generate(&cfg_b, &ptoks, *max_new)?;
            let fork = (0..ga.len().max(gb.len())).find(|&i| ga.get(i) != gb.get(i));
            let text_of = |m: &Model, g: &[i32]| g.iter().map(|&t| m.piece(t)).flatten().collect::<Vec<u8>>();
            let seq: Vec<i32> = ptoks.iter().chain(&ga).copied().collect();
            generation = Some(Generation {
                prompt_tokens: ptoks.clone(),
                a_text: String::from_utf8_lossy(&text_of(&ma, &ga)).into_owned(),
                b_text: String::from_utf8_lossy(&text_of(&mb, &gb)).into_owned(),
                a: ga,
                b: gb,
                fork,
                fork_reproduced: None,
                fork_gap_a: None,
            });
            if seq.len() < 2 {
                return Err("prompt plus generation is shorter than 2 tokens".into());
            }
            vec![seq]
        }
    };

    let mut repair_layers: Vec<usize> = if req.repair && layers {
        let s = req.repair_stride.max(1);
        let mut v: Vec<usize> = (0..n_layer).step_by(s).collect();
        if v.last() != Some(&(n_layer - 1)) {
            v.push(n_layer - 1);
        }
        v
    } else {
        vec![]
    };
    // Flash attention as it actually ran: "auto" is resolved inside llama.cpp.
    let probe_tok = seqs[0][0];
    let fa_used = [ma.flash_attn_used(&cfg_a, probe_tok)?, mb.flash_attn_used(&cfg_b, probe_tok)?];
    let mut factors: Vec<&str> = if req.attribution { req.a.differences(&req.b) } else { vec![] };
    if factors.contains(&"flash_attn") && fa_used[0] == fa_used[1] {
        factors.retain(|f| *f != "flash_attn");
        warnings.push(format!(
            "flash attention '{}' and '{}' both resolved to {} here, so it is not an attribution factor",
            req.a.flash_attn,
            req.b.flash_attn,
            if fa_used[0] { "on" } else { "off" }
        ));
    }
    if factors.len() > 5 {
        return Err(format!("attribution over {} factors would need {} runs per chunk; vary at most 5", factors.len(), 1 << factors.len()));
    }
    if req.attribution && factors.len() < 2 {
        warnings.push("attribution needs at least two settings that differ between arms; skipped".into());
    }
    let cells: Vec<u32> = if factors.len() >= 2 { (1..(1u32 << factors.len()) - 1).collect() } else { vec![] };
    let mut cell_acc: Vec<(f64, usize, usize, Option<String>)> = vec![(0.0, 0, 0, None); cells.len()];

    let per_chunk = 2 + req.floor as usize + 2 * repair_layers.len() + cells.len();
    let total = per_chunk * seqs.len();
    let mut done = 0;
    let mut tick = |stage: &str| {
        done += 1;
        progress(Progress { done, total, stage });
    };

    let mut chunks = vec![];
    let mut floor_stats: Vec<Vec<PosStat>> = vec![];
    for (ci, toks) in seqs.iter().enumerate() {
        check()?;
        let n = toks.len();
        let next = &toks[1..];
        let (la, cap_a) = if layers {
            match capture_all(&ma, &cfg_a, toks, n_layer) {
                Ok(x) => x,
                // This architecture does not name its per-layer residual `l_out-N`.
                Err(e) if e.contains(" saw 0 of ") => {
                    warnings.push(format!("no layer map or transplants: {}", e.lines().next().unwrap_or("")));
                    layers = false;
                    repair_layers.clear();
                    (ma.eval(&cfg_a, toks, &mut [])?, vec![])
                }
                Err(e) => return Err(e),
            }
        } else {
            (ma.eval(&cfg_a, toks, &mut [])?, vec![])
        };
        tick("arm A");
        let (lb, cap_b) = if layers { capture_all(&mb, &cfg_b, toks, n_layer)? } else { (mb.eval(&cfg_b, toks, &mut [])?, vec![]) };
        tick("arm B");
        let pos = stats::compare(&la, &lb, v, next, req.top_k);

        if ci == 0 {
            if let Some(g) = generation.as_mut().filter(|g| g.fork.is_some()) {
                let f = g.fork.unwrap();
                let at = g.prompt_tokens.len() + f - 1;
                if at < n {
                    g.fork_reproduced = Some(Some(pos[at].top_a) == g.a.get(f).copied() && Some(pos[at].top_b) == g.b.get(f).copied());
                    if let (Some(&ta), Some(&tb)) = (g.a.get(f), g.b.get(f)) {
                        g.fork_gap_a = Some(la[at * v + ta as usize] as f64 - la[at * v + tb as usize] as f64);
                    }
                }
            }
        }

        // ponytail: one reference perturbation (an odd split near half), recorded per chunk.
        // A max over several splits would be a tighter reference; it costs a pass each.
        let floor_ubatch = req.floor.then(|| floor_split(cfg_a.n_ubatch, n));
        let floor_kl = if let Some(ub) = floor_ubatch {
            check()?;
            let lf = ma.eval(&Cfg { n_ubatch: ub, ..cfg_a }, toks, &mut [])?;
            tick("numerical floor");
            let fp = stats::compare(&la, &lf, v, next, 0);
            let kl = fp.iter().map(|p| p.kl as f32).collect();
            floor_stats.push(fp);
            Some(kl)
        } else {
            None
        };

        let layer_err = layers.then(|| (0..n_layer).map(|l| stats::rel_err_rows(&cap_a[l], &cap_b[l], ma.info.n_embd)).collect());

        let (mut den, mut noi) = (vec![], vec![]);
        if !repair_layers.is_empty() {
            let mut buf = vec![0f32; n * v];
            for &l in &repair_layers {
                check()?;
                patched(&mb, &cfg_b, toks, l, &cap_a[l], &mut buf)?;
                den.push(stats::kl_rows(&la, &buf, v).into_iter().map(|x| x as f32).collect());
                tick("transplant A→B");
                check()?;
                patched(&ma, &cfg_a, toks, l, &cap_b[l], &mut buf)?;
                noi.push(stats::kl_rows(&la, &buf, v).into_iter().map(|x| x as f32).collect());
                tick("transplant B→A");
            }
        }

        for (k, &mask) in cells.iter().enumerate() {
            check()?;
            let mut arm = req.a.clone();
            for (i, f) in factors.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    arm = arm.with(&req.b, f);
                }
            }
            if cell_acc[k].3.is_none() {
                let m = models.get(&arm.model)?;
                match m.eval(&arm.cfg()?, toks, &mut []) {
                    Ok(lc) => {
                        let kl = stats::kl_rows(&la, &lc, v);
                        cell_acc[k].0 += kl.iter().skip(skip).sum::<f64>();
                        cell_acc[k].1 += la.chunks(v).zip(lc.chunks(v)).skip(skip).filter(|(x, y)| argmax(x) != argmax(y)).count();
                        cell_acc[k].2 += n.saturating_sub(skip);
                    }
                    // An invalid combination (e.g. quantized V cache without flash attention) is a
                    // result to show, not a reason to throw the whole run away.
                    Err(e) => cell_acc[k].3 = Some(e.lines().next().unwrap_or("rejected by llama.cpp").to_string()),
                }
            }
            tick("attribution");
        }

        chunks.push(Chunk {
            tokens: toks.clone(),
            pos,
            floor_kl,
            floor_ubatch,
            layer_err,
            repair_layers: (!repair_layers.is_empty()).then(|| repair_layers.clone()),
            repair_denoise: (!den.is_empty()).then_some(den),
            repair_noise: (!noi.is_empty()).then_some(noi),
        });
    }

    // Aggregates, over scored positions only (each chunk's first `skip` are shown, not counted).
    fn tail<T>(p: &[T], k: usize) -> &[T] {
        &p[k.min(p.len())..]
    }
    let scored = |p| tail(p, skip);
    let summary = stats::summarize(&chunks.iter().map(|c| scored(&c.pos)).collect::<Vec<_>>());
    if summary.n_positions == 0 {
        return Err(format!("nothing left to score after skipping the first {skip} tokens of each chunk"));
    }
    let floor = req.floor.then(|| stats::summarize(&floor_stats.iter().map(|c| scored(c)).collect::<Vec<_>>()));
    let mean_rows = |get: &dyn Fn(&Chunk) -> Option<&Vec<Vec<f32>>>| -> Option<Vec<f32>> {
        let first = get(chunks.first()?)?;
        let rows = first.len();
        let mut sum = vec![0f64; rows];
        let mut cnt = 0usize;
        for c in &chunks {
            let m = get(c)?;
            for (r, row) in m.iter().enumerate() {
                sum[r] += row.iter().skip(skip).map(|&x| x as f64).sum::<f64>();
            }
            cnt += m[0].len().saturating_sub(skip);
        }
        Some(sum.into_iter().map(|s| (s / cnt.max(1) as f64) as f32).collect())
    };
    let layer_err_mean = mean_rows(&|c| c.layer_err.as_ref());
    let repair_denoise_mean = mean_rows(&|c| c.repair_denoise.as_ref());
    let repair_noise_mean = mean_rows(&|c| c.repair_noise.as_ref());
    let attribution = (!cells.is_empty()).then(|| {
        let mut out = vec![Cell { switched: vec![], mean_kl: 0.0, flip_rate: 0.0, error: None }];
        for (k, &mask) in cells.iter().enumerate() {
            let (s, f, n, ref error) = cell_acc[k];
            out.push(Cell {
                switched: factors.iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, f)| f.to_string()).collect(),
                mean_kl: if error.is_some() { 0.0 } else { s / n as f64 },
                flip_rate: if error.is_some() { 0.0 } else { f as f64 / n as f64 },
                error: error.clone(),
            });
        }
        out.push(Cell {
            switched: factors.iter().map(|f| f.to_string()).collect(),
            mean_kl: summary.mean_kl,
            flip_rate: summary.flip_rate,
            error: None,
        });
        Attribution { factors: factors.iter().map(|f| f.to_string()).collect(), cells: out }
    });

    let mut vocab = BTreeMap::new();
    let mut add = |t: i32| {
        vocab.entry(t).or_insert_with(|| piece(&ma, t));
    };
    for c in &chunks {
        c.tokens.iter().for_each(|&t| add(t));
        for p in &c.pos {
            p.topk_a.iter().chain(&p.topk_b).for_each(|&(t, _)| add(t));
            add(p.top_a);
            add(p.top_b);
        }
    }
    if let Some(g) = &generation {
        g.a.iter().chain(&g.b).for_each(|&t| add(t));
    }

    Ok(Results {
        summary,
        floor,
        layer_err_mean,
        repair_layers: (!repair_layers.is_empty()).then_some(repair_layers),
        repair_denoise_mean,
        repair_noise_mean,
        chunks,
        generation,
        attribution,
        flash_attn_used: Some(fa_used),
        vocab,
        warnings,
    })
}

fn argmax(r: &[f32]) -> usize {
    let mut b = 0;
    for (i, &x) in r.iter().enumerate() {
        if x > r[b] {
            b = i;
        }
    }
    b
}
