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

/// Lock a mutex even if a panicking thread poisoned it; the data is still usable here.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn random_token() -> String {
    let h = |i: u64| {
        let mut s = std::collections::hash_map::RandomState::new().build_hasher();
        s.write_u64(i);
        s.finish()
    };
    format!("{:016x}{:016x}", h(1), h(2))
}

pub struct Options {
    pub port: u16,
    pub model_dirs: Vec<PathBuf>,
    pub workspace: PathBuf,
    pub open: bool,
}

pub fn serve(opts: Options) -> Result<(), String> {
    std::fs::create_dir_all(&opts.workspace).map_err(|e| format!("{}: {e}", opts.workspace.display()))?;
    let server = Server::http(("127.0.0.1", opts.port)).map_err(|e| format!("cannot bind 127.0.0.1:{}: {e}", opts.port))?;
    let port = server.server_addr().to_ip().map_or(opts.port, |a| a.port());
    let url = format!("http://127.0.0.1:{port}/");
    let mut st = State {
        token: random_token(),
        origin_hosts: vec![format!("127.0.0.1:{port}"), format!("localhost:{port}")],
        model_dirs: opts.model_dirs,
        workspace: opts.workspace,
        models: Arc::new(Mutex::new(Models::default())),
        jobs: Arc::new(Mutex::new(HashMap::new())),
        next_job: 1,
        last_job: None,
    };
    eprintln!("heterodyne {} — llama.cpp {}", env!("CARGO_PKG_VERSION"), &env!("HD_LLAMA_COMMIT")[..env!("HD_LLAMA_COMMIT").len().min(10)]);
    eprintln!("workspace: {}", st.workspace.display());
    eprintln!("open {url}");
    if opts.open {
        open_browser(&url);
    }
    for mut req in server.incoming_requests() {
        let resp = handle(&mut st, &mut req);
        let _ = req.respond(resp);
    }
    Ok(())
}

fn open_browser(url: &str) {
    let r = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if r.is_err() {
        eprintln!("could not open a browser; visit {url}");
    }
}
