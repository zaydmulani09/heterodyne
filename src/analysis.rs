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
