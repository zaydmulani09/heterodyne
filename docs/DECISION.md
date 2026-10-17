# Why heterodyne exists (decision record, 2026-10-16)

## The original idea did not survive

The brief proposed "ModelScope", a local "AI MRI" workbench: token stepping, attention and
activation views, logit lens, activation patching in a UI, checkpoint comparison and exported
experiments. Research before any code found that:

- **The core UI already exists.** [NDIF Workbench](https://github.com/ndif-team/workbench)
  (open source, self-hostable) ships logit lens, activation patching ("Patch Lens") and saved
  workspaces. Neuronpedia, TransformerLens, nnsight, circuit-tracer and Inseq cover the rest
  as libraries or hosted tools. Token stepping with candidate probabilities is in mikupad,
  Tapestry Loom, oobabooga and llama-server (`n_probs`).
- **Polished workbenches get abandoned.** Meta's LM Transparency Tool (1.25k★) is archived.
  So are DeepMind's Penzai and OpenAI's Transformer Debugger.
- **The name is taken.** "ModelScope" is Alibaba's model hub. "LLM-MRI" and "AI MRI" are used
  by at least four repos.

## What survived: differential analysis in the runtime people actually use

All of these were verified by reading source code or by running them:

1. **Every interpretability tool runs PyTorch, not llama.cpp.** Code search finds no GGUF
   support in nnsight, Neuronpedia, Workbench, circuit-tracer, diffing-toolkit, Inseq, LIT or
   pyvene. TransformerLens skips GGUF on purpose (`hf_scraper.py:526`).
2. **PyTorch can't stand in for llama.cpp.** llama.cpp's CPU backend quantizes *activations*
   to Q8_K/Q8_0/Q8_1 before every quantized matmul (`ggml-cpu.c` type_traits `vec_dot_type`).
   A dequantized-weight model in PyTorch measures a model nobody runs.
3. **llama.cpp has no public hidden-state API in `llama.h`.** Issue #27112 asking for one is open
   since 2026-08-15. PRs #20785 (server activation capture) and #20820 (capture tool) were
   closed as out of scope. The `llama-eval-callback` example prints truncated tensor values and
   sums. (An internal `llama_set_embeddings_layer_inp` exists in `src/llama-ext.h`, outside
   the public header.)
4. **The only quant-vs-reference tool in llama.cpp reports aggregates.**
   `llama-perplexity --kl-divergence` keeps every base logit in a file and in memory
   (O(context × vocab); #28733 asks to reduce that), stores them as 16-bit compressed
   log-probs, scores only the second half of each
   chunk, and never writes out the per-token values it computes. When I re-ran a model against
   its own base file, it reported 0.11% top-1 "flips". That number is pure storage error.
5. **Users want to know whether a change hurts *their* workload, and each writes their own
   harness to find out.** Examples: mlx-kld (Mac-only), a flip-rate harness on Level1Techs,
   oobabooga patching llama.cpp to get prompt logprobs, llama.cpp's own
   `causal-verify-logits`. Hacker News and r/LocalLLaMA threads say aggregate KLD "does not
   mean much" for a given task.
6. **Runtime settings change outputs, and people can't tell which one did it.** Open
   llama.cpp issues cover quantized-K-cache repeats (#28527), prompt-cache flips (#26817),
   lossy speculative decoding on quantized targets (#25618), batch-invariance (#28111) and
   long-prompt logits (#28211). I measured on Qwen2.5-0.5B (CPU, llama.cpp b11696):
   - Changing thread count leaves logits bitwise identical.
   - Changing `-ub 512` to `-ub 64` flips the top-1 token on about 1.3% of positions.
   - `-ctk q8_0 -ctv q8_0 -fa on` gives a mean KLD of 0.027, against 0.039 for Q4_K_M weights.

## Decision

Build **heterodyne**. It runs two configurations ("arms") of llama.cpp on the same tokens.
An arm is a GGUF file plus runtime settings. heterodyne shows where the two arms diverge,
token by token and layer by layer, and runs causal transplants of activations between them.
Every number comes from llama.cpp's own numerics, on CPU.

- **Problem:** "Did this quant, KV-cache type, batch setting or finetune change my model's
  behavior on my text? Where? Which part of the network did it, and does it matter?"
- **Primary user:** local-model power users choosing quants and settings, people who make
  quants, and llama.cpp contributors chasing divergence bugs. Researchers studying quantization
  are a secondary audience.
- **What existing tools miss:**
  - Per-token exact divergence without a giant base file.
  - Divergence measured against a noise floor.
  - Divergence localized by layer, inside the real runtime.
  - Causal tests of which layer's divergence matters.
  - Which single setting caused a divergence.
- **Technical advantage:** a small C++ shim over llama.cpp's scheduler eval callback captures
  and overwrites named activations mid-graph. Two arms run on identical tokens. Exact
  full-vocabulary statistics are computed in Rust.
- **Closest prior art:** [jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) (90★,
  July 2026) already edits activations through the same callback. It does this for
  single-model Jacobian-lens steering. It does not compare models or configurations.
  heterodyne's contribution is the differential method, not the hook.
- **Strongest counterargument:**
  - The comparisons people argue about most involve 27B+ models, which this laptop can't run.
  - A divergence on a single prompt is usually a near-tie and means little.
  - Mitigations: the tool scales with the user's hardware, since it is just llama.cpp. It
    reports margins, noise floors and corpus-level confidence intervals rather than anecdotes.
- **Smallest experiment that could disprove it:** the hook must meet two conditions, on a
  quant pair:
  - **(a) Observing doesn't perturb:** logits with the capture callback on must be bitwise
    identical to logits with it off.
  - **(b) Writes reach downstream:** writing an arm's own values back into it must be a
    bitwise no-op, and writing zeros must change the output.

  The repair curve must also be informative (not flat noise) on a real quant pair. See
  `docs/VALIDATION.md` for the result.
- **Why a new project instead of contributing:**
  - llama.cpp maintainers closed activation-capture PRs as out of scope.
  - Workbench, nnsight and TransformerLens are built on PyTorch and can't host llama.cpp
    numerics.
  - jlens-gguf is a lens visualizer. A differential engine with its own experiment format is a
    different product, and could feed back into it later.
