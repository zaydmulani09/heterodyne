# Contributing

Thanks for looking. heterodyne is small on purpose; the bar for changes is that every number
it shows stays traceable to a llama.cpp pass you could re-run.

## Build and test

```bash
git clone --recursive https://github.com/zaydmulani09/heterodyne
cd heterodyne
mkdir -p tests/fixtures
curl -L -o tests/fixtures/stories260K.gguf https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
cargo test --release -- --test-threads=1
```

You need a Rust toolchain, CMake and a C++17 compiler. On Windows use the MSVC toolchain
(`rustup default stable-x86_64-pc-windows-msvc`) with the Visual Studio Build Tools; CMake
ships with them under `Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.
The first build compiles llama.cpp and takes a while; later builds only recompile Rust.

`HETERODYNE_LLAMA_DIR=/path/to/llama.cpp` builds against another checkout.
`HETERODYNE_VERBOSE=1` prints llama.cpp's log.

## What a good change looks like

- **New measurement:** say what it means in one sentence, add it to the table in
  `docs/ARCHITECTURE.md`, and add a test that would fail if it were computed wrong.
- **New architecture claim:** run the engine tests against that model
  (`HETERODYNE_TEST_MODEL=path/to/model.gguf cargo test --release --test engine`) and paste
  the output in the PR. Only then add it to the supported table in the README.
- **llama.cpp bump:** update the submodule, run all tests, and note any change in the
  numbers of `examples/` experiments (re-run `heterodyne verify`).
- **UI:** draw only fields that exist in the experiment JSON. No synthetic or smoothed data.

Keep code in the style around it: small modules, no new dependency for something a few
lines can do, and explicit errors instead of panics on user input.

## Reporting a numerical problem

Attach the experiment JSON (it carries model hashes, llama.cpp commit and CPU features) and
the output of `heterodyne verify` on your machine.
