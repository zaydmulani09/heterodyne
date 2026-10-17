// Builds the pinned llama.cpp (CPU backend only, static) plus our C++ shim.
use std::{env, path::PathBuf, process::Command};

fn main() {
    let llama = env::var("HETERODYNE_LLAMA_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("vendor/llama.cpp"));
    if !llama.join("CMakeLists.txt").exists() {
        panic!("llama.cpp source not found at {} — run `git submodule update --init --depth 1`", llama.display());
    }
    println!("cargo:rerun-if-changed=src/shim.cpp");
    println!("cargo:rerun-if-env-changed=HETERODYNE_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=HETERODYNE_PORTABLE");

    // Portable builds (release binaries) target x86-64-v3 instead of the build machine.
    let portable = env::var("HETERODYNE_PORTABLE").is_ok();
    let x86 = env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
    let mut cfg = cmake::Config::new(&llama);
    cfg.profile("Release");
    for (k, v) in [
        ("BUILD_SHARED_LIBS", "OFF"),
        ("LLAMA_BUILD_COMMON", "OFF"),
        ("LLAMA_BUILD_TESTS", "OFF"),
        ("LLAMA_BUILD_TOOLS", "OFF"),
        ("LLAMA_BUILD_EXAMPLES", "OFF"),
        ("LLAMA_BUILD_SERVER", "OFF"),
        ("LLAMA_BUILD_APP", "OFF"),
        ("LLAMA_OPENSSL", "OFF"),
        ("GGML_OPENMP", "OFF"),
        ("GGML_METAL", "OFF"),
        ("GGML_BLAS", "OFF"),
        // Accelerate would need the framework linked here, and changes kernels; CPU paths only.
        ("GGML_ACCELERATE", "OFF"),
        ("GGML_NATIVE", if portable { "OFF" } else { "ON" }),
    ] {
        cfg.define(k, v);
    }
    if portable && x86 {
        for k in ["GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C"] {
            cfg.define(k, "ON");
        }
    }
    let dst = cfg.build();

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/shim.cpp")
        .include(llama.join("include"))
        .include(llama.join("ggml/include"))
        .compile("hdshim");

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-search=native={}", dst.join("lib64").display());
    for lib in ["llama", "ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // The llama.cpp archives are C++ too: name the C++ runtime after them for GNU ld.
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => println!("cargo:rustc-link-lib=advapi32"),
        Ok("linux") => {
            for l in ["stdc++", "pthread", "m", "dl"] {
                println!("cargo:rustc-link-lib={l}");
            }
        }
        Ok("macos") => println!("cargo:rustc-link-lib=c++"),
        _ => {}
    }

    let commit = Command::new("git")
        .args(["-C", &llama.to_string_lossy(), "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=HD_LLAMA_COMMIT={commit}");
    println!("cargo:rustc-env=HD_BUILD_FLAGS={}", if portable { "portable" } else { "native" });
}
