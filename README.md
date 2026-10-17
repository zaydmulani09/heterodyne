# heterodyne

[![ci](https://github.com/zaydmulani09/heterodyne/actions/workflows/ci.yml/badge.svg)](https://github.com/zaydmulani09/heterodyne/actions/workflows/ci.yml)

**Diff two llama.cpp runs token by token and layer by layer, then move activations between
them to test which layers' divergence actually reaches the output.**

A quantized GGUF, a quantized KV cache, a different batch size, a fine-tune: each changes
what your model says, and the usual answer to "how much, and where?" is one aggregate
number. heterodyne runs two configurations of llama.cpp (two *arms*: a GGUF file plus the
runtime settings that can change its numbers) on identical tokens and shows:

- **where the outputs differ**: exact full-vocabulary KL(A‖B) and top-1 changes at every
  position, next to a measured numerical floor;
- **where inside the network the difference appears**: the residual stream of both arms
  after every layer;
- **which layers carry it to the output**: activation transplants that write one arm's
  residual stream into the other, layer by layer, in llama.cpp's own numerics;
- **which setting causes it**: every combination of the settings that differ, run and scored;
- **where a greedy answer forks**, and whether that fork survives re-scoring.

Everything runs locally on the CPU, in the same quantized kernels you actually run.
No API keys, no uploads, no Python.

![heterodyne comparing Qwen3-0.6B BF16 with Q4_K_M, inspecting a confident flip in is_prime](docs/screenshot-light.png)

## Why another tool

Interpretability tools (TransformerLens, nnsight, NDIF Workbench, Neuronpedia) run PyTorch.
On CPU, llama.cpp quantizes *activations* to 8 bits before every quantized matmul, so a
PyTorch model with dequantized weights is a different model from the one you run.
llama.cpp itself has no public API for per-layer hidden states (#27112), and its KL tool,
`llama-perplexity --kl-divergence`, reports aggregates, keeps every base logit in memory and on
disk, and stores them compressed (on Qwen2.5-0.5B that storage alone shows up as 0.11% "flips").
The research behind this choice, with sources, is in [docs/DECISION.md](docs/DECISION.md).

| | per-token view | inside the network | interventions | llama.cpp's own numerics | where it runs |
|---|---|---|---|---|---|
| `llama-perplexity --kl-divergence` | no (aggregates) | no | no | yes | any llama.cpp backend |
| [mlx-kld](https://github.com/sammcj/mlx-kld) | per-token arrays, most divergent tokens | no | no | no (MLX) | Apple silicon |
| NDIF Workbench / nnsight / TransformerLens | yes | yes | yes | no (PyTorch) | local PyTorch or NDIF's remote GPUs |
| [jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) | one model | yes (Jacobian lens) | steering, single model | yes | CPU (llama.cpp) |
| [Thireus GGUF-Tool-Suite](https://github.com/Thireus/GGUF-Tool-Suite) | no | per-tensor PPL/KLD sweeps | requantize per tensor | yes (ik_llama.cpp) | llama.cpp backends |
| **heterodyne** | **yes, A vs B** | **yes, A vs B** | **cross-arm transplants** | **yes** | **CPU (llama.cpp)** |

## Quick start

Download a binary from [Releases](https://github.com/zaydmulani09/heterodyne/releases)
(Linux x86-64, Windows x86-64, macOS arm64), or [build from source](#build-from-source).

```bash
heterodyne serve --models ~/models
```

That opens `http://127.0.0.1:7878/`. Pick two GGUF files (or one file and two settings),
paste some text, and run. Every finished run is saved to `~/.heterodyne/experiments`.

You need two GGUFs of the same model with the same tokenizer, for example from
[ggml-org/Qwen3-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF) (BF16) and
[unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) (quants).

### Command line

```bash
# How far is Q4_K_M from BF16 on my text, and which layers carry the difference?
heterodyne diff --a Qwen3-0.6B-BF16.gguf --b Qwen3-0.6B-Q4_K_M.gguf \
  --text-file examples/corpus.txt --chunk 128 --max-chunks 12 --repair -o bf16-vs-q4km.json

# Same file twice: what do a q8_0 KV cache and a different batch split change, and which is it?
heterodyne diff --a Qwen3-0.6B-Q8_0.gguf --b-set ctk=q8_0,ctv=q8_0,ub=63 \
  --text-file examples/corpus.txt --chunk 128 --max-chunks 12 --attribution

# Where do the greedy answers fork?
heterodyne diff --a Qwen3-0.6B-BF16.gguf --b Qwen3-0.6B-Q3_K_M.gguf \
  --prompt "A train leaves at 14:35 and the trip takes 2 hours and 50 minutes. When does it arrive?"

# Re-run a saved experiment and compare every number
heterodyne verify bf16-vs-q4km.json --search ~/models
```

Arm settings: `ctk`/`ctv` (KV cache type: f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0,
iq4_nl), `fa` (flash attention: auto, on, off), `ub` (ubatch size, 0 = whole chunk),
`threads`. `--b-set` starts from A's settings; `--b` defaults to A's file. `--skip-first`
(default 16) sets how many leading tokens of each chunk are shown but not counted.
