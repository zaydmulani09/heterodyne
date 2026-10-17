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

type Resp = Response<std::io::Cursor<Vec<u8>>>;

fn header(k: &str, v: &str) -> Header {
    Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
}

fn text(code: u16, ctype: &str, body: String) -> Resp {
    Response::from_string(body)
        .with_status_code(code)
        .with_header(header("Content-Type", ctype))
        .with_header(header("Cache-Control", "no-store"))
        .with_header(header("X-Content-Type-Options", "nosniff"))
        .with_header(header("X-Frame-Options", "DENY"))
        .with_header(header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        ))
}

fn font(bytes: &[u8]) -> Resp {
    Response::from_data(bytes.to_vec())
        .with_header(header("Content-Type", "font/woff2"))
        .with_header(header("Cache-Control", "max-age=86400"))
}

fn ok(v: Value) -> Resp {
    text(200, "application/json", v.to_string())
}

fn err(code: u16, msg: &str) -> Resp {
    text(code, "application/json", json!({ "error": msg }).to_string())
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        match (b[i], b.get(i + 1).copied().and_then(hex), b.get(i + 2).copied().and_then(hex)) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            (b'+', _, _) => {
                out.push(b' ');
                i += 1;
            }
            (c, _, _) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query(url: &str, key: &str) -> Option<String> {
    let q = url.split_once('?')?.1;
    q.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

fn handle(st: &mut State, req: &mut tiny_http::Request) -> Resp {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();
    fn hdr(req: &tiny_http::Request, name: &'static str) -> Option<String> {
        req.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str().to_string())
    }
    let get = |name: &'static str| hdr(req, name);
    // DNS-rebinding guard: only answer requests addressed to our own loopback origin.
    if !get("Host").map_or(false, |h| st.origin_hosts.contains(&h)) {
        return err(403, "bad host");
    }
    match (req.method(), path.as_str()) {
        (Method::Get, "/") => return text(200, "text/html; charset=utf-8", INDEX.replace("__HD_TOKEN__", &st.token)),
        (Method::Get, "/app.js") => return text(200, "text/javascript; charset=utf-8", APP_JS.into()),
        (Method::Get, "/style.css") => return text(200, "text/css; charset=utf-8", STYLE.into()),
        (Method::Get, "/fonts/archivo-wdth.woff2") => return font(FONT_UI),
        (Method::Get, "/fonts/fragment-mono.woff2") => return font(FONT_MONO),
        _ => {}
    }
    if !path.starts_with("/api/") {
        return err(404, "not found");
    }
    if get("X-Heterodyne-Token").as_deref() != Some(st.token.as_str()) {
        return err(403, "missing or wrong token");
    }
    let mut body = String::new();
    if req.method() == &Method::Post {
        if req.body_length().unwrap_or(0) > MAX_BODY
            || req.as_reader().take(MAX_BODY as u64 + 1).read_to_string(&mut body).is_err()
            || body.len() > MAX_BODY
        {
            return err(413, "body too large or not UTF-8");
        }
    }
    match (req.method(), path.as_str()) {
        (Method::Get, "/api/state") => ok(state_json(st)),
        (Method::Post, "/api/run") => start_run(st, &body),
        (Method::Get, "/api/job") => job_json(st, query(&url, "id").and_then(|s| s.parse().ok())),
        (Method::Post, "/api/cancel") => {
            if let Some(j) =
                query(&url, "id").and_then(|s| s.parse::<u64>().ok()).and_then(|id| lock(&st.jobs).get(&id).map(|j| j.cancel.clone()))
            {
                j.store(true, Ordering::Relaxed);
            }
            ok(json!({}))
        }
        (Method::Post, "/api/save") => save(st, &body),
        (Method::Get, "/api/experiment") => match query(&url, "file").and_then(|f| ws_file(st, &f)) {
            Some(p) => match std::fs::read_to_string(&p) {
                Ok(s) => text(200, "application/json", s),
                Err(e) => err(404, &e.to_string()),
            },
            None => err(400, "bad file name"),
        },
        (Method::Post, "/api/verify") => start_verify(st, &body),
        _ => err(404, "not found"),
    }
}

/// A file name inside the workspace (no separators, .json only).
fn ws_file(st: &State, name: &str) -> Option<PathBuf> {
    let ok = !name.is_empty() && name.ends_with(".json") && !name.contains(['/', '\\', ':']) && !name.starts_with('.');
    ok.then(|| st.workspace.join(name))
}

fn list_models(dirs: &[PathBuf]) -> Vec<Value> {
    let mut out = vec![];
    let mut visit = |d: &Path| {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("gguf")) {
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    out.push(json!({ "path": p.to_string_lossy(), "name": e.file_name().to_string_lossy(), "bytes": size }));
                }
            }
        }
    };
    for d in dirs {
        visit(d);
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten().filter(|e| e.path().is_dir()) {
                visit(&e.path());
            }
        }
    }
    out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    out
}

fn list_experiments(ws: &Path) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(ws)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().map_or(false, |x| x == "json"))
        .filter_map(|e| {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()?;
            (v["schema"] == bundle::SCHEMA).then(|| {
                json!({
                    "file": e.file_name().to_string_lossy(),
                    "title": v["notes"]["title"],
                    "created_unix": v["created_unix"],
                    "a": v["provenance"]["a"]["file_name"],
                    "b": v["provenance"]["b"]["file_name"],
                    "mean_kl": v["results"]["summary"]["mean_kl"],
                    "flip_rate": v["results"]["summary"]["flip_rate"],
                })
            })
        })
        .collect();
    out.sort_by(|a, b| b["created_unix"].as_u64().cmp(&a["created_unix"].as_u64()));
    out
}

fn state_json(st: &State) -> Value {
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "llama_cpp_commit": env!("HD_LLAMA_COMMIT"),
        "build": env!("HD_BUILD_FLAGS"),
        "system_info": ffi::system_info(),
        "model_dirs": st.model_dirs.iter().map(|d| d.to_string_lossy()).collect::<Vec<_>>(),
        "models": list_models(&st.model_dirs),
        "cache_types": ffi::CACHE_TYPES.iter().map(|t| t.0).collect::<Vec<_>>(),
        "workspace": st.workspace.to_string_lossy(),
        "experiments": list_experiments(&st.workspace),
        "busy": lock(&st.jobs).values().any(|j| j.status == "running"),
        "job": st.last_job,
    })
}

fn new_job(st: &mut State, kind: &str) -> Result<(u64, Arc<AtomicBool>), Resp> {
    let mut jobs = lock(&st.jobs);
    if jobs.values().any(|j| j.status == "running") {
        return Err(err(409, "another job is running; one at a time keeps CPU timings and memory sane"));
    }
    jobs.retain(|_, j| j.status == "running"); // keep memory bounded: only the newest finished job is polled
    let id = st.next_job;
    st.next_job += 1;
    st.last_job = Some(id);
    let cancel = Arc::new(AtomicBool::new(false));
    jobs.insert(
        id,
        Job { status: "running".into(), kind: kind.into(), stage: "starting".into(), cancel: cancel.clone(), ..Default::default() },
    );
    Ok((id, cancel))
}

fn spawn<F>(st: &State, id: u64, f: F)
where
    F: FnOnce(&mut Models, &dyn Fn(crate::analysis::Progress)) -> Result<Value, String> + Send + 'static,
{
    let jobs = st.jobs.clone();
    let models = st.models.clone();
    std::thread::spawn(move || {
        let jobs2 = jobs.clone();
        let progress = move |p: crate::analysis::Progress| {
            if let Some(j) = lock(&jobs2).get_mut(&id) {
                (j.done, j.total, j.stage) = (p.done, p.total, p.stage.to_string());
            }
        };
        // A panic must not leave the job "running" forever and block every later run.
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut lock(&models), &progress))).unwrap_or_else(|p| {
            Err(format!(
                "internal error: {}",
                p.downcast_ref::<&str>().copied().or(p.downcast_ref::<String>().map(|s| s.as_str())).unwrap_or("panic")
            ))
        });
        if let Some(j) = lock(&jobs).get_mut(&id) {
            match out {
                Ok(v) => (j.status, j.result) = ("done".into(), Some(v)),
                Err(e) if e == "cancelled" => j.status = "cancelled".into(),
                Err(e) => (j.status, j.error) = ("error".into(), Some(e)),
            }
        }
    });
}

#[derive(Deserialize)]
struct RunBody {
    request: Request,
    #[serde(default)]
    notes: Notes,
}

fn start_run(st: &mut State, body: &str) -> Resp {
    let rb: RunBody = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return err(400, &format!("bad request: {e}")),
    };
    if let Err(e) = rb.request.validate() {
        return err(400, &e);
    }
    let (id, cancel) = match new_job(st, "run") {
        Ok(x) => x,
        Err(r) => return r,
    };
    let ws = st.workspace.clone();
    spawn(st, id, move |models, progress| {
        let e = bundle::record(rb.request, rb.notes, models, &cancel, progress)?;
        // Saved as soon as it finishes, so a closed tab or reload loses nothing.
        let file = bundle::save_to(&ws, &e)?;
        Ok(json!({ "experiment": e, "file": file }))
    });
    ok(json!({ "job": id }))
}

#[derive(Deserialize)]
struct VerifyBody {
    file: String,
}

fn start_verify(st: &mut State, body: &str) -> Resp {
    let Some(path) = serde_json::from_str::<VerifyBody>(body).ok().and_then(|b| ws_file(st, &b.file)) else {
        return err(400, "bad file name");
    };
    let e = match bundle::load(&path) {
        Ok(e) => e,
        Err(m) => return err(400, &m),
    };
    let search = st.model_dirs.clone();
    let (id, _cancel) = match new_job(st, "verify") {
        Ok(x) => x,
        Err(r) => return r,
    };
    spawn(st, id, move |models, progress| {
        let r = bundle::verify(&e, &search, models, progress)?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    });
    ok(json!({ "job": id }))
}

fn job_json(st: &State, id: Option<u64>) -> Resp {
    let jobs = lock(&st.jobs);
    match id.and_then(|id| jobs.get(&id)) {
        Some(j) => ok(
            json!({ "status": j.status, "kind": j.kind, "done": j.done, "total": j.total, "stage": j.stage, "error": j.error, "result": j.result }),
        ),
        None => err(404, "no such job"),
    }
}
