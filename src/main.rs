use clap::{Parser, Subcommand};
use heterodyne::analysis::{Arm, Input, Models, Progress, Request};
use heterodyne::bundle::{self, Notes};
use heterodyne::{ffi, server};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

#[derive(Parser)]
#[command(name = "heterodyne", version, about = "Diff two llama.cpp runs token by token and layer by layer.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the local web UI (127.0.0.1 only).
    Serve {
        #[arg(long, default_value_t = 7878)]
        port: u16,
        /// Directory to list GGUF files from (repeatable; scanned one level deep).
        #[arg(long = "models", value_name = "DIR")]
        models: Vec<PathBuf>,
        /// Where experiments are saved. Default: ~/.heterodyne/experiments
        #[arg(long)]
        workspace: Option<PathBuf>,
        #[arg(long)]
        no_open: bool,
    },
    /// Compare two arms on text or on a generated continuation.
    Diff {
        /// Arm A (reference) GGUF.
        #[arg(long)]
        a: String,
        /// Arm B GGUF (default: same file as A, so only settings differ).
        #[arg(long)]
        b: Option<String>,
        /// Arm A settings, e.g. ctk=f16,ctv=f16,fa=off,ub=0,threads=8
        #[arg(long, default_value = "")]
        a_set: String,
        /// Arm B settings (start from A's settings), e.g. ctk=q8_0,ctv=q8_0,fa=on
        #[arg(long, default_value = "")]
        b_set: String,
        #[arg(long, conflicts_with_all = ["text", "prompt"])]
        text_file: Option<PathBuf>,
        #[arg(long, conflicts_with = "prompt")]
        text: Option<String>,
        /// Generate greedily from this prompt with both arms and find the fork.
        #[arg(long)]
        prompt: Option<String>,
        /// Do not wrap --prompt in the model's chat template.
        #[arg(long)]
        raw: bool,
        #[arg(long, default_value_t = 96)]
        max_new: usize,
        #[arg(long, default_value_t = 256)]
        chunk: usize,
        #[arg(long, default_value_t = 8)]
        max_chunks: usize,
        /// Per-layer activation transplant in both directions (costs 2 passes per layer).
        #[arg(long)]
        repair: bool,
        #[arg(long, default_value_t = 1)]
        repair_stride: usize,
        /// Run every combination of the settings that differ between arms.
        #[arg(long)]
        attribution: bool,
        #[arg(long)]
        no_floor: bool,
        #[arg(long)]
        no_layers: bool,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        /// Leading tokens of each chunk to show but leave out of every total.
        #[arg(long, default_value_t = 16)]
        skip_first: usize,
        #[arg(long, default_value = "")]
        title: String,
        #[arg(long, default_value = "")]
        hypothesis: String,
        /// Write the experiment file here.
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Re-run a saved experiment and compare every number.
    Verify {
        file: PathBuf,
        /// Extra directories to look for the model files in (matched by sha256).
        #[arg(long = "search", value_name = "DIR")]
        search: Vec<PathBuf>,
    },
    /// Print what llama.cpp reports about a GGUF file.
    Info { model: PathBuf },
}

fn parse_set(base: &Arm, spec: &str) -> Result<Arm, String> {
    let mut a = base.clone();
    for kv in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("setting '{kv}' is not key=value"))?;
        match k {
            "ctk" | "cache_k" => a.cache_k = v.into(),
            "ctv" | "cache_v" => a.cache_v = v.into(),
            "fa" | "flash_attn" => a.flash_attn = v.into(),
            "ub" | "ubatch" => a.ubatch = v.parse().map_err(|_| format!("bad ubatch '{v}'"))?,
            "t" | "threads" => a.threads = v.parse().map_err(|_| format!("bad threads '{v}'"))?,
            _ => return Err(format!("unknown setting '{k}' (ctk, ctv, fa, ub, threads)")),
        }
    }
    a.cfg()?;
    Ok(a)
}

fn home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| ".".into())
}

fn progress(p: Progress) {
    eprint!("\r[{:>3}/{:<3}] {:<24}", p.done, p.total, p.stage);
    let _ = std::io::stderr().flush();
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("\nerror: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::Serve { port, models, workspace, no_open } => server::serve(server::Options {
            port,
            model_dirs: if models.is_empty() { vec![std::env::current_dir().map_err(|e| e.to_string())?] } else { models },
            workspace: workspace.unwrap_or_else(|| home().join(".heterodyne").join("experiments")),
            open: !no_open,
        }),
        Cmd::Info { model } => {
            let m = ffi::Model::load(&model)?;
            let i = &m.info;
            println!(
                "{}\n  layers {}  width {}  vocab {}  params {}  ctx_train {}",
                i.desc, i.n_layer, i.n_embd, i.n_vocab, i.n_params, i.n_ctx_train
            );
            println!("  sha256 {}", bundle::file_sha256(&model)?);
            println!("  system {}", ffi::system_info());
            Ok(())
        }
        Cmd::Verify { file, search } => {
            let e = bundle::load(&file)?;
            let r = bundle::verify(&e, &search, &mut Models::default(), &progress)?;
            eprintln!();
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            if r.outcome == "exact" && r.integrity_ok {
                Ok(())
            } else {
                Err(format!("verification outcome: {}", r.outcome))
            }
        }
        Cmd::Diff {
            a,
            b,
            a_set,
            b_set,
            text_file,
            text,
            prompt,
            raw,
            max_new,
            chunk,
            max_chunks,
            repair,
            repair_stride,
            attribution,
            no_floor,
            no_layers,
            top_k,
            skip_first,
            title,
            hypothesis,
            out,
        } => {
            let base = Arm {
                model: a.clone(),
                threads: Arm::default_threads(),
                ubatch: 0,
                flash_attn: "auto".into(),
                cache_k: "f16".into(),
                cache_v: "f16".into(),
            };
            let arm_a = parse_set(&base, &a_set)?;
            let mut arm_b = parse_set(&arm_a, &b_set)?;
            arm_b.model = b.unwrap_or(a);
            let input = match (text_file, text, prompt) {
                (Some(f), _, _) => Input::Text { text: std::fs::read_to_string(&f).map_err(|e| format!("{}: {e}", f.display()))? },
                (_, Some(t), _) => Input::Text { text: t },
                (_, _, Some(p)) => Input::Generate { prompt: p, chat: !raw, max_new },
                _ => return Err("give --text-file, --text or --prompt".into()),
            };
            let req = Request {
                a: arm_a,
                b: arm_b,
                input,
                chunk_tokens: chunk,
                max_chunks,
                layers: !no_layers,
                repair,
                repair_stride,
                attribution,
                floor: !no_floor,
                top_k,
                skip_first,
            };
            let e = bundle::record(
                req,
                Notes { title, hypothesis, interpretation: String::new() },
                &mut Models::default(),
                &AtomicBool::new(false),
                &progress,
            )?;
            eprintln!();
            print_summary(&e);
            if let Some(p) = out {
                std::fs::write(&p, serde_json::to_string(&e).unwrap()).map_err(|e| format!("{}: {e}", p.display()))?;
                println!("wrote {}", p.display());
            }
            Ok(())
        }
    }
}
