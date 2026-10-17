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
