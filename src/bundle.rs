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
