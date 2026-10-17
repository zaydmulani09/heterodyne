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
