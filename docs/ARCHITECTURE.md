# Architecture

```
 browser (web/: vanilla JS, SVG + canvas)          CLI (src/main.rs)
        │  JSON over HTTP, 127.0.0.1 only                │
        ▼                                                ▼
 src/server.rs ──► src/bundle.rs ──► src/analysis.rs ──► src/stats.rs
 (token, Host     (provenance,      (arms, chunks,       (exact KL, margins,
  check, 1 job,    integrity,        floor, layer map,    top-k, cluster
  auto-save)       verify)           transplants,         t-interval, residual
                                                          rel. error)
                                     attribution, fork)
                                          │
                                          ▼
                                    src/ffi.rs  (safe Rust wrapper)
                                          │  C ABI we own: hd_* structs only
                                          ▼
                                    src/shim.cpp
                                          │  llama_context_params.cb_eval
                                          ▼
                              vendor/llama.cpp (pinned submodule, CPU backend, static)
```

## The hook

llama.cpp's scheduler takes an eval callback (`cb_eval`). When it is set, the graph is
computed in pieces: for each node the callback asks "do you want this tensor?"; if yes,
the scheduler computes everything up to that node, synchronizes, and calls back with the
data in place, before computing anything downstream
(`ggml_backend_sched_compute_splits`, `vendor/llama.cpp/ggml/src/ggml-backend.cpp`).

`shim.cpp` uses that second call to either copy the tensor out (capture) or overwrite it
(patch) with `ggml_backend_tensor_get/set`. Because consumers of a node always come after
it in topological order, an overwrite is seen by every downstream op.

Tensors are addressed by their graph names: `l_out-<layer>` is the residual stream after
each block, `ffn_inp-<layer>` the residual between a block's attention and its MLP (the
attention output already added in); the llama, qwen2 and qwen3 graphs at the pinned commit
name both. A tensor fires once per ubatch, so the shim tracks a row offset per hook and
checks at the end that every row of every hooked tensor was seen; missing names become an
error, not silent zeros. On a run's first pass, a kind of tensor the graph does not name is
dropped with a warning saying what is lost (no layer map, or no attention/MLP split), and the
rest of the run goes ahead.

Everything layout-sensitive (`llama_context_params`, `llama_batch`, `ggml_tensor`) stays in
C++. Rust sees only the `hd_*` structs we define, so a llama.cpp update cannot silently
shift a struct field under the FFI.

## Measurements

| quantity | definition | where |
|---|---|---|
| KL(A‖B) per position | Σ p_A(v)(a_v − b_v) − lse(a) + lse(b), summed in f64 from raw f32 logits | `stats::kl_row` |
| flip | argmax_A ≠ argmax_B (ties → lowest token id, same as greedy) | `stats::compare` |
| flip gap | log p_A(A's top) − log p_A(B's top); a flip is "near-tie" below 0.25 nats and "confident" at 1 nat or more (conventions, not tests) | `stats::compare`, constants |
| margin | log p_A(top1) − log p_A(top2), shown for context | `stats::compare` |
| numerical floor | KL(A‖A′), A′ = arm A re-run with an odd ubatch near half the chunk (odd sizes force llama.cpp's CPU remainder paths); the split is recorded per chunk | `analysis::floor_split` |
| flash attention as run | whether the graph contains `GGML_OP_FLASH_ATTN_EXT`, probed once per arm ("auto" is resolved inside llama.cpp and not exposed) | `shim.cpp::hd_flash_attn_used` |
| layer map | ‖h_B − h_A‖ / ‖h_A‖ of `l_out-l` per position | `stats::rel_err_rows` |
| denoise transplant | KL(A ‖ B with A's `l_out-l` written in), every position | `analysis::run` |
| noise transplant | KL(A ‖ A with B's `l_out-l` written in), every position | `analysis::run` |
| attribution | every subset of differing settings switched from A's to B's value; flash attention counts only if it changes what actually runs; combinations llama.cpp rejects are recorded, not fatal | `analysis::run` |
| CI | 95% t-interval for the position-weighted mean with a cluster-robust standard error over chunks (positions within a chunk are correlated), k−1 degrees of freedom | `stats::cluster_t_ci` |

Each teacher-forced pass creates a fresh context with every position's logits requested,
so there is no prefix cache and no state carried between passes. Non-finite logits are an
error, never a number.

## Experiment file

`heterodyne/experiment@1` JSON: request, provenance (GGUF sha256, llama.cpp commit, build
flags, `llama_print_system_info()`, OS/arch), results, user notes, and an `integrity`
sha256 over the request and provenance (model file locations excluded, so files can move)
and the results. It detects edits and corruption; it is a checksum, not a signature. Notes
are excluded on purpose. `serde_json` is built with `float_roundtrip`, so every stored number
parses back to the same bits; `verify` re-runs the request, compares results for exact
equality and reports the largest difference in every stored array. The server writes each
finished run to the workspace immediately.

## Why these choices

- **llama.cpp, not PyTorch.** On CPU, llama.cpp quantizes activations to Q8_K/Q8_0 before
  quantized dot products. A PyTorch model with dequantized weights is a different model.
- **Static llama.cpp, pinned.** The hook depends on scheduler behaviour and graph tensor
  names; pinning makes those testable facts instead of assumptions.
- **No GPU backend (yet).** On GPU backends the callback fires after async dispatch, reading
  forces device→host copies, and kernels differ; none of that has been validated.
- **tiny_http + vanilla JS.** One local user, one job at a time; no async runtime or
  frontend build step is needed.
