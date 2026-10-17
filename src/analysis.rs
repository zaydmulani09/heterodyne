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
