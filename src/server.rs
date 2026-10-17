//! Local web UI: 127.0.0.1 only, per-process token on every API call, one job at a time.
use crate::analysis::{Models, Request};
use crate::bundle::{self, Experiment, Notes};
use crate::ffi;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Response, Server};

const INDEX: &str = include_str!("../web/index.html");
const APP_JS: &str = include_str!("../web/app.js");
const STYLE: &str = include_str!("../web/style.css");
const FONT_UI: &[u8] = include_bytes!("../web/fonts/archivo-wdth.woff2");
const FONT_MONO: &[u8] = include_bytes!("../web/fonts/fragment-mono.woff2");
const MAX_BODY: usize = 64 << 20;

#[derive(Default)]
struct Job {
    status: String, // running | done | error | cancelled
    kind: String,   // run | verify
    done: usize,
    total: usize,
    stage: String,
    result: Option<Value>,
    error: Option<String>,
    cancel: Arc<AtomicBool>,
}

struct State {
    token: String,
    origin_hosts: Vec<String>,
    model_dirs: Vec<PathBuf>,
    workspace: PathBuf,
    models: Arc<Mutex<Models>>,
    jobs: Arc<Mutex<HashMap<u64, Job>>>,
    next_job: u64,
    /// The most recent job, so a reloaded page can reattach to it.
    last_job: Option<u64>,
}
