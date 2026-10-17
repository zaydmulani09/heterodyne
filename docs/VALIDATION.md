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
| `self_patch_is_identity_across_ubatches` | writes land in the right rows when a chunk spans several ubatches | ubatch 7: writing a layer's own values back is **bitwise** identical |
| `multi_ubatch_capture_fills_every_row` | captures are reassembled correctly across ubatches | ubatch 7 vs whole chunk: every row present, rel. diff < 1e-3 |
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
