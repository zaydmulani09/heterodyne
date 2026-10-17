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

## What it found on a laptop

All numbers below come from the experiment files in [`examples/`](examples/), recorded on an
Intel i7-1355U (AVX2, CPU only) with llama.cpp b11696 and Qwen3-0.6B: ggml-org's BF16 GGUF as
the reference, unsloth's quants as variants, [`examples/corpus.txt`](examples/corpus.txt)
(prose, arithmetic, Python, SQL, JSON, shell, facts, dialogue) in twelve 128-token chunks,
1,282 scored positions. Intervals are 95% t-intervals over chunks. One model, one text, one
machine: treat these as observations and re-run them on yours.

**How far each quant drifts from BF16.**

| B | mean KL(A‖B) | top-1 changed | of which confident | perplexity (A: 7.83) |
|---|---|---|---|---|
| Q8_0 | 0.0022 (0.0015–0.0028) | 2.2% | 0 | 7.83 |
| UD-Q4_K_XL | 0.058 (0.043–0.073) | 11.3% | 16 | 8.33 |
| Q4_K_M | 0.067 (0.052–0.083) | 11.6% | 25 | 8.53 |
| Q3_K_M | 0.214 (0.169–0.260) | 20.8% | 96 | 9.47 |
| Q2_K | 0.780 (0.636–0.923) | 36.3% | 308 | 16.75 |

"Confident" means B's top token was one BF16 rated at least 1 nat below its own choice; every
Q8_0 flip was a near-tie (within 0.25 nats). Both thresholds are conventions, not tests. The
two 4-bit files' intervals overlap, but they were scored on the same positions, so the right
comparison is paired: Q4_K_M minus UD-Q4_K_XL = 0.0087 (0.0016–0.0158). On this text the
UD file diverges less.

**Where Q4_K_M's divergence comes from.** Writing BF16's residual stream into Q4_K_M after
layer *l* and measuring what divergence is left:

```
layer   0      4      8      12     16     20     24     26     27
KL    0.062  0.052  0.038  0.029  0.021  0.013  0.006  0.004  0.0013
```

It falls steadily over all 28 layers: no single block carries it. The 0.0013 left after the
last layer is what the final norm and output head of the two files add on their own. The
reverse transplant (Q4_K_M's residual written into BF16) rises just as smoothly.

How these totals are computed matters. The first version of this experiment counted every
position, including predictions made from one or two tokens of context, where Q4_K_M
confidently emits UTF-8 byte-order-mark bytes (KL ≈ 7 at each chunk's first position). Those
few positions made up half of the mean and made the last block look responsible for half of
the divergence. heterodyne now shows them but leaves the first 16 tokens of each chunk out of
every total; the setting is recorded in the experiment file.

**A confident flip, in code.** In `is_prime`, after `if n < 2:\n        return`, BF16 puts
97.9% on ` False`; Q4_K_M puts 79.1% on ` True` (screenshot above; KL 1.69). Writing BF16's
residual into Q4_K_M brings that token's KL from 0.48 after layer 6 down to 0.04 after layer 9;
writing Q4_K_M's residual into BF16 reproduces the flip from about layer 12 (KL 0.81, then
1.4 by layer 15). At this token, in this text, the decisive difference forms in layers 6 to 12.

**Which runtime setting changes the output.** Same Q8_0 file twice; B adds a q8_0 K and V
cache and splits each chunk into ubatches of 63. Every combination, scored:

| switched to B's value | mean KL | top-1 changed |
|---|---|---|
| ubatch (whole chunk → 63) | 0.0015 | 1.3% |
| K cache f16 → q8_0 | 0.0035 | 2.0% |
| V cache f16 → q8_0 | 0.0016 | 2.0% |
| K + V | 0.0038 | 3.0% |
| ubatch + K, ubatch + V, all three | same as without ubatch, to every digit | |

On its own, the batch split moves this model about as far as a q8_0 V cache. Once either
cache is quantized, the split adds nothing measurable. A q8_0 KV cache (0.0038) moves this
model further than quantizing its weights from BF16 to Q8_0 (0.0022). Flash attention is not
in the table: on this CPU build `auto` resolves to on, and heterodyne checks what actually ran
before treating it as a setting.

**llama.cpp's CPU path is not batch-invariant at every ubatch size.** Q4_K_M, one 128-token
chunk, all positions counted, flash attention off; A in one ubatch vs B in ubatches of:

| ubatch | 8 | 16 | 32 | 56 | 60 | 62 | 63 | 64 | 65 | 96 | 100 | 127 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 changed | 0 | 0 | 0 | 0 | 0 | 3.1% | 4.7% | 0 | 4.7% | 0 | 0 | 0 |
| KL exactly 0 | yes | yes | yes | yes | yes | no | no | yes | no | yes | yes | no |

In this setup, splits whose ubatches are all multiples of 4 rows are bitwise identical and the
others are not. The code explains it for K-quants: the repacked matmul quantizes activation
rows in interleaved groups of 4 and sends the remainder through a separate path
(`ggml/src/ggml-cpu/repack.cpp:4697-4702`). With flash attention off, Q8_0 and BF16 stay exact
at ubatch 63; with it on, every weight type changes (Q4_K_M: 15.6% of top-1 tokens), so the
flash-attention path has its own size dependence. Stock `llama-perplexity` agrees (Q4_K_M,
ubatch 57 vs 114: KLD 0.0057, 97.4% same top token). This is why heterodyne's numerical floor
uses an odd split and records it.

## Reading the results

| what you see | what it is | what it is not |
|---|---|---|
| KL(A‖B) per token | how much of A's next-token distribution B fails to reproduce, in nats, exact over the full vocabulary | a quality score: B can diverge and still be right |
| numerical floor | the same comparison for A against itself, re-run with odd-sized ubatches that change kernel paths and summation order | a bound on noise; it is one reference perturbation |
| near-tie / confident flips | top-1 changes where A rated B's choice within 0.25 nats / at least 1 nat below its own | evidence on its own; the thresholds are conventions |
| layer map | ‖h_B − h_A‖ / ‖h_A‖ of the residual stream after each layer | a cause: differences propagate whether or not they matter |
| transplant curves | B re-run with A's residual written in after layer l (and the reverse); the KL that results | a general law: it holds for the text you ran |
| attribution table | every subset of differing settings switched to B's value and scored | an additive decomposition: interactions are shown, not split |
| interval | 95% t-interval over chunks, positions within a chunk treated as correlated | exact with few chunks; with fewer than 5 it is flagged as approximate |

Each experiment file stores your hypothesis and interpretation next to the measurements,
outside the integrity checksum: interpretation is not evidence.

## What is implemented

| feature | status |
|---|---|
| two arms = GGUF + KV types + flash-attn + ubatch + threads | implemented, tested |
| teacher-forced exact KL, flips, gaps, top-k, perplexity per arm | implemented, tested |
| chunked text, leading tokens shown but not counted, cluster t-interval | implemented, tested |
| numerical floor (odd ubatch re-split, recorded per chunk) | implemented, tested |
| residual-stream divergence map (`l_out` per layer) | implemented, tested |
| transplants in both directions, every n-th layer | implemented, tested |
| factorial attribution over up to 5 differing settings, rejected combinations recorded | implemented, tested on the fixture; exercised on Qwen3-0.6B |
| flash attention attributed by what actually ran | implemented, tested |
| greedy generation fork + teacher-forced re-check of the fork | implemented; exercised on Qwen3-0.6B |
| experiment files with sha256 provenance, integrity checksum, exact replay | implemented, tested |
| local web UI with auto-save and reattach after reload | implemented; tested by hand in Chromium-based browsers at desktop and phone widths, light and dark |
| GPU backends | not built; not validated |
| component-level transplants (attention vs MLP output) | planned |
| build-to-build comparison (two llama.cpp versions) | planned |

### Tested models

| architecture | model | result |
|---|---|---|
| llama | stories260K (ggml-org/models) | engine test suite; CI on Linux, Windows, macOS |
| qwen2 | Qwen2.5-0.5B-Instruct fp16 (Qwen/Qwen2.5-0.5B-Instruct-GGUF) | engine test suite, local |
| qwen3 | Qwen3-0.6B BF16 (ggml-org) + unsloth Q8_0, UD-Q4_K_XL, Q4_K_M, Q3_K_M, Q2_K | engine test suite, local; experiments in `examples/` |

Other architectures should work when their graph names each layer's residual output
`l_out-<n>` (most decoder-only models in llama.cpp do). When it doesn't, heterodyne still
compares outputs and says why the layer views are missing. Please file an architecture
report when you try one.

## Hardware and cost

CPU only. Peak memory is both models (mmapped) plus about `4 × tokens × vocab × 4` bytes of
logits (≈ 300 MB for a 128-token chunk of a 151k vocabulary) and, with layer views,
`2 × layers × tokens × width × 4` bytes of residual captures (30 MB here; about 1.3 GB per
chunk for an 80-layer, 8192-wide model). heterodyne refuses runs that would need more than
about 12 GB. Time is dominated by forward passes:

- a plain comparison: 2 passes per chunk, +1 for the floor;
- transplants: +2 passes per transplanted layer per chunk;
- attribution: +2^k − 2 passes per chunk for k differing settings.

The BF16 vs Q4_K_M experiment with transplants at all 28 layers (12 chunks, 708 passes) took
15 minutes on the laptop above; the same comparison without transplants takes about a
minute, including hashing both files.

## Limitations

- **Two arms must share a tokenizer**; layer maps and transplants also need the same number
  of layers and width.
- **One text is one text.** Use many chunks and read the interval; transplant curves are
  interventions on your input, not properties of the model.
- **CPU backend only.** GPU kernels compute differently; nothing here says anything about them.
- **Replay is exact on the same machine and build** in our tests. Other CPUs take different
  SIMD paths and can change low-order bits; `verify` reports by how much. Cross-machine replay
  has not been measured.
- **Generation forks** come from incremental decoding, while scoring re-runs the sequence in
  one batch; the two can disagree, and the UI says when they do.
- llama.cpp is pinned (b11696). Newer graph changes could rename tensors.

## Build from source

```bash
git clone --recursive https://github.com/zaydmulani09/heterodyne
cd heterodyne
cargo build --release
./target/release/heterodyne serve --models /path/to/ggufs
```

Requires Rust, CMake and a C++17 compiler (Windows: MSVC toolchain and the Visual Studio
Build Tools). The first build compiles llama.cpp. `HETERODYNE_PORTABLE=1` builds for
x86-64-v3 instead of the build machine. See [CONTRIBUTING.md](CONTRIBUTING.md) for tests.
