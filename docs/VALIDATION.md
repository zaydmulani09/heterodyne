# Validation

What has been checked, how, and on what. Hardware for everything below: Intel i7-1355U
(2 P + 8 E cores, AVX2, no AVX-512), 11.7 GB RAM, Windows 11, CPU backend only,
llama.cpp b11696 (`ceefd16`).

## 1. The hook does what the method assumes (automated, `tests/engine.rs`)

These run against `stories260K.gguf` (ggml-org/models, 1.2 MB, llama architecture), plus a
Q8_0 copy made in-test with llama.cpp's own quantizer. CI runs them on Linux, Windows and
macOS (see the badge in the README for the latest run).

| test | claim | check |
|---|---|---|
| `observing_does_not_perturb` | capturing every `l_out` does not change the computation | logits with the callback == logits without it, **bitwise** |
| `self_patch_is_identity_and_writes_reach_downstream` | an overwrite is consumed downstream, and only an actual change changes anything | writing a layer's own values back is bitwise identical; writing zeros changes the logits |
| `last_layer_transplant_restores_reference_when_weights_match` | transplants move exactly the state they claim to | same weights, B with a bf16 K cache: writing A's final residual into B reproduces A's logits **bitwise** |
| `post_attention_residual_is_observed_and_patched_exactly` | the split point between attention and MLP (`ffn_inp`) is hooked as exactly as `l_out` | capturing every `ffn_inp` leaves the logits **bitwise** unchanged; writing the first and last layer's own values back is **bitwise** identical; writing zeros changes the logits |
| `self_patch_is_identity_across_ubatches` | writes land in the right rows when a chunk spans several ubatches | ubatch 7: writing a layer's own values back is **bitwise** identical |
| `multi_ubatch_capture_fills_every_row` | captures are reassembled correctly across ubatches | ubatch 7 vs whole chunk: every row present, rel. diff < 1e-2 (a misplaced row would differ by O(1); kernel differences between split sizes reached 1.0e-3 on Apple silicon in CI and stayed below 1e-3 on x86) |
| `identical_arms_diff_to_exactly_zero` | no noise floor is introduced by the measurement itself | A == B: mean KL, layer map and transplant map are exactly 0 |
| `quantized_variant_diverges_and_transplant_repairs_it` | the pipeline produces sane, non-trivial numbers | f32 vs Q8_0: KL > 0, interval present, last-layer transplant lowers KL |
| `floor_stride_and_attribution_are_consistent` | the floor, stride and attribution paths do what they say | flash attention "auto" vs "on" resolves to the same kernel and is dropped as a factor; the full attribution cell equals the main result; floor splits are odd; every other layer plus the last is transplanted |
| `bad_requests_are_rejected_before_running` | out-of-range input cannot abort or hang the process | absurd thread counts, top-k and non-GGUF files are rejected with an error |
| `experiment_roundtrips_and_verifies_exactly` | experiment files replay exactly | JSON roundtrip is lossless; `verify` returns `exact`; editing a result breaks the checksum; moving a model file does not |
| `incompatible_cache_type_is_an_error_not_an_abort` | bad settings fail safely | q8_0 K cache on a head size of 8 returns an error instead of a ggml assert abort |
| `greedy_generation_is_deterministic` | generation repeats | two greedy runs give identical tokens |

The same suite also passes locally against Qwen2.5-0.5B-Instruct fp16 (qwen2 architecture,
299 s) and Qwen3-0.6B BF16 (qwen3, 457 s), via `HETERODYNE_TEST_MODEL`, each with a Q8_0
copy made in-test. On those two the block-size test skips itself (their head sizes are
multiples of 32), which is the correct outcome.

The KL code is separately unit-tested (`src/stats.rs`) against a textbook softmax
implementation (agreement within 1e-12 absolute + 1e-9 relative over 5,000-token
vocabularies), an analytic two-token case, logit-shift invariance, and exact zero on
identical rows. The interval code is checked against the classic t-interval it must reduce to
when chunks are equal in size.

## 2. Disproof experiment on a real quant pair (manual, before any product code)

Qwen3-0.6B, ggml-org BF16 GGUF vs unsloth Q4_K_M, 242 tokens of mixed text, one ubatch,
flash attention off:

- capture callback on vs off: logits **bitwise identical**
- variant self-patch at layer 14: **bitwise identical**
- zeroing layer 14 in the variant: mean KL 10.15 (writes reach downstream)
- baseline: mean KL 0.1336, top-1 agreement 85.5%
- transplanting the BF16 residual into Q4_K_M after layer l (mean KL left):
  0.132 (l=0), 0.111 (2), 0.085 (8), 0.074 (11), 0.067 (15), 0.052 (18), 0.040 (22),
  0.035 (26), **0.0012 (27)**

The curve falls steadily, not in one step. In this model and on this text, Q4_K_M's
divergence builds up across many layers rather than coming from one. What is left after
the final layer (0.0012) is the quantized output head's own contribution. This is one
text and one model; it is an observation, not a law.

## 3. Runtime settings change outputs (manual, llama-perplexity, Qwen2.5-0.5B)

Measured with stock `llama-perplexity --kl-divergence`, c=256, Q4_K_M as both base and
variant, before heterodyne existed. These motivated the "arm = file + settings" design:

| B differs from A by | mean KLD | same top token |
|---|---|---|
| `-t 3` instead of `-t 8` | 0.000001 | 99.89% |
| `-ub 64 -b 64` | 0.00086 | 98.65% |
| `-ctk q8_0 -ctv q8_0 -fa on` | 0.0266 | 92.01% |
| (Q4_K_M weights vs F16, for scale) | 0.0387 | 91.29% |

The thread-count row is not zero only because llama-perplexity stores its base logits as
16-bit compressed log-probs; that storage alone produces 0.11% "flips". heterodyne keeps
f32 logits in memory and computes KL in f64, so identical runs give exactly 0 (test above).

## 4. Batch invariance of llama.cpp's CPU path (heterodyne, cross-checked with llama-perplexity)

Qwen3-0.6B, 128-token chunk of `examples/corpus.txt`; A = whole chunk in one ubatch,
B = same file and settings with the given ubatch (`--b-set ub=N`):

- Q4_K_M, flash attention off: KL is **exactly 0** for ubatch 8, 16, 24, 32, 40, 48, 56, 60,
  64, 96, 100, and non-zero for 62, 63, 65, 127 (up to 4.7% top-1 changes, all near-ties).
- ubatch 63 vs 64, flash attention off / on:

  | weights | 63, fa off | 63, fa on | 64, fa off | 64, fa on |
  |---|---|---|---|---|
  | BF16 | 0 | 1.5e-4 (0.8%) | 0 | 0 |
  | Q8_0 | 0 | 2.4e-3 (2.3%) | 0 | 0 |
  | Q3_K_M | 5.9e-3 (1.6%) | 1.9e-2 (5.5%) | 0 | 0 |
  | Q4_K_M | 7.3e-3 (4.7%) | 1.9e-2 (15.6%) | 0 | 0 |

  (mean KL, top-1 changed)
- Independent check with stock `llama-perplexity` (same build), Q4_K_M, `-c 114`, base
  `-ub 114` vs `-ub 57`: mean KLD 0.0057, same top token 97.4%. With `-c 128`, `-ub 128` vs
  `-ub 64`: KLD ≈ 0, same top token 100%.

So the effect is in llama.cpp, not in heterodyne's hook. For the K-quants the code shows the
mechanism: the repacked matmul quantizes activation rows in interleaved groups of 4 and sends
the remainder rows through a separate `from_float` path
(`ggml/src/ggml-cpu/repack.cpp:4697-4702` at b11696), so a ubatch that is not a multiple of 4
puts some rows through different arithmetic. Why every weight type changes with flash
attention on was not traced in the kernel code.

These results are for Qwen3-0.6B, 128-token chunks, on this CPU. Divisibility by 4 is not a
general rule: with flash attention on, an earlier run with 256-token chunks saw a small
difference (mean KL 1.3e-5) at ubatch 64, coming from a 76-token final chunk split 64 + 12,
so the flash-attention path has size dependence of its own.

## 5. Determinism across separate runs

The BF16 vs Q4_K_M comparison in `examples/` was recorded twice by separate processes (once
with layer captures and transplants, once in the quant ladder without): mean KL, flips and
perplexity are identical to every printed digit (0.067050, 149 flips, 7.825 / 8.534).
`verify` compares all stored numbers for exact equality;
`experiment_roundtrips_and_verifies_exactly` exercises it in CI. Every file in `examples/` was
re-run with `heterodyne verify` on the recording machine; see the results in section 7.

## 6. A measurement artifact found and fixed: positions with little context

The first full recording of the BF16 vs Q4_K_M experiment counted every position. At each
chunk's first position (one token of context) Q4_K_M put most of its mass on the bytes of a
UTF-8 byte-order mark, giving KL ≈ 7 there. With all positions counted:

- mean KL was 0.123; with each chunk's first 16 positions excluded it is 0.067, so those 192
  of 1,474 positions made up about half of the total;
- the transplant curve showed a cliff at the last layer (0.063 → 0.0013), suggesting the final
  block carried half the divergence; with them excluded the curve falls smoothly from 0.062
  to 0.0036 over layers 0 to 26 and the last block adds about 0.002.

Both effects are real llama.cpp behavior, but at positions nobody scores in practice.
heterodyne now records `skip_first` in every request (default 16), draws those positions
muted, and leaves them out of every total, interval, floor, layer mean, transplant mean and
attribution cell.

## 7. Replaying the shipped examples

Every experiment file in `examples/` was re-run with `heterodyne verify <file> --search
<models dir>` on the machine that recorded it (same build, same CPU), after the files had
been rewritten as compact JSON. All 8 came back `exact` with the integrity checksum intact:
per-token KL, floor, layer maps and both transplant directions, a maximum absolute
difference of 0 in every array and no top-1 changes. The longest (BF16 vs Q4_K_M with
transplants at all 28 layers) re-ran 708 passes.

Across builds: the v0.1.0 release binary for Windows (built by CI with the portable
x86-64-v3 flags, so `llama_print_system_info()` reports a different feature set) also
replays `examples/qwen3-ladder-Q4_K_M.json`, recorded with a native build, as `exact` on the
same machine. `verify` flags the build and CPU-feature differences and still finds every
number identical. Replay on a different CPU has not been measured.

## 8. Not validated

- GPU backends (CUDA, Metal, Vulkan, SYCL). The build disables them.
- Architectures other than llama (stories260K), qwen2 (Qwen2.5-0.5B) and qwen3 (Qwen3-0.6B). The layer map and
  transplants need `l_out-<layer>` in the graph; heterodyne reports an error when a layer's
  tensor is missing, but other architectures have not been run.
- Mixture-of-experts, recurrent and hybrid models.
- Cross-machine replay. `verify` on the same machine is exact; on a CPU with different SIMD
  paths, low-order bits are expected to differ, and that has not been measured.
