# Native and browser batch release qualification

Release scope: `burn_ardy` 0.1.3, `burn_llama` 0.1.1 and
`bevy_burn_human` 0.6.0. Model weights and CDN manifests are unchanged.
The preceding [pipeline qualification](../performance-2026-09-26/README.md)
continues to cover the unchanged SOMA and GEM-X implementations.

## Changes

ARDY generates 1–8 independent actors in one batch. Prompts, random streams and
waypoints are independent; window lengths, steps and guidance must match.
`generate` uses the same implementation with one actor. CPU conditioning now
covers at most 200 frames per actor, instead of allocating for the entire clip.
The eight-actor maximum uses 4,224,000 bytes for its two conditioning arrays,
including a 12,000-frame request. GPU history retains at most 160 frames per
actor, while CPU history retains only five root features per frame. Final clips
grow with the requested output length. These bounds do not include weights,
backend scratch allocations or final clips.

Llama yields a browser task after each transformer block, without per-block
device synchronization or readback. The studio remains enabled by default.
Core Bevy integrations can disable default features to omit the studio and its
model/UI dependencies. Applications previously disabling default features must
explicitly enable `studio` when using the motion module or browser viewer.

## Numerical and behavioral evidence

The independent baseline is commit
`d26d7a4afa3b8657e85687e83539cbbdbbfca1f0`. Native and browser fixtures were
captured separately from that implementation, with only capture harnesses added.
Five existing Burn Llama embeddings supply actual prompt conditions. The seven
cases cover batch sizes 1/2/4/8, histories 0/4/40/160, 40/44/80/120/240 output
frames, dense/sparse curved paths, wrapped headings, different seeds and a
duplicate seed. Every single-request clip has **zero FK/root position difference**
from its platform's baseline; rotations agree to floating-point roundoff.
Reversing actor order preserves the complete batch output to the same precision.
Cancellation is checked before allocation and after the first window of eight
12,000-frame requests. A final progress notification cannot discard a completed
clip.

Continuous denoising, ten-step DDIM and identical-input decoding are compared
using the existing checkpoint tolerances (0.002, 0.02 and 0.002). The original
PyTorch checkpoint hooks also pass. Complete serial/batch clip differences are
reported separately because FSQ motion-code rounding is discontinuous.

The initial proposal to gate every batch against a serial clip at 10 mm maximum
FK error was rejected by the native partial-window example. Its four-frame
history case changes 7 of 2,560 first-window FSQ codes by one code level despite
passing continuous numerical checks. This produces a 58.14 mm maximum joint
difference, 3.435 mm joint RMS and 7.163 mm maximum root difference. This is a
real reproducibility limitation, not an exact-parity claim. Native dense and
sparse long-path cases have zero position difference; the eight-actor example
differs by at most 2.032 mm. All browser batch clips have zero FK/root difference
from their serial baseline on the tested adapter, with rotations agreeing to
floating-point roundoff. Keep backend and batch size fixed for seeded comparisons.

Llama's ten reference cases cover causal/bidirectional masks, tokenization and
pooling. Native and browser minimum cosine similarity exceed 0.9999984 and
maximum RMSE remains below 0.003417. These numerical examples are not a
human-rated or dataset-level semantic quality benchmark.

## Timing and resource limits

See [measurements](timings.md), [native report](native.json),
[browser report](browser.json) and [provenance](provenance.json).
Timings include complete decoded, host-visible clips and exclude weight loading
and text encoding. Each case warms all paths, then alternates serial/batch order
over three synchronized rounds. Native uses the repository's optimized dev
profile; WASM uses its release profile. Cross-platform times are not a comparison
of equivalent build configurations. Native timings were repeated after an
initial run overlapped compilation; the initial timings are not used here.

The workstation is shared, with an RTX PRO 6000 Blackwell 96 GiB and driver
610.43.02. Browser tests use hardware WebGPU in Chromium 153, with no software
fallback. This is one adapter qualification, not a universal performance claim.
Batch browser peak linear memory is 278.75 MiB; the Llama run peaks at 479.5 MiB.
These figures exclude GPU and browser-managed storage memory. Whole-run event
loop gaps include loading, cold shader compilation and validation; the final batch
run has a 400 ms maximum gap, while the [earlier run](browser-initial.json)
reached 1.233 seconds. Cooperative scheduling does not eliminate
all cold-start stalls or guarantee a particular display frame rate.

## Integration and release checks

- Workspace tests and strict Clippy pass, including optional model tools.
- Native core-only Bevy tests and the WASM core-only library check pass.
- The explicit `studio` WASM release binary builds; CI builds that binary for Pages.
- GPU tests check actor ordering/cancellation and Bevy tensor handoff, normals,
  bounds and retention of bounds after a topology change.
- Browser studio smoke checks cover SOMA loading/posing, camera orbiting and
  image selection/crop controls and [world-space waypoints](studio-waypoints.png).
  See [SOMA pose](studio-wave.png) and
  [image input](studio-image-input.png). The displayed pose in the image-input
  screenshot is the prior SOMA pose, not a newly inferred GEM-X result.
- All three archive dry runs pass; archives contain no model weights.

## Reproduction

Prepare `embedding-0.json` through `embedding-4.json` using the published Llama
artifact and the five prompts in the existing text reference. Prepare inputs:

```sh
python tool/scripts/prepare_ardy_batch.py --embeddings EMBEDDING_DIR \
  --baseline-commit d26d7a4afa3b8657e85687e83539cbbdbbfca1f0 --out batch-input.json
```

In a separate checkout of that baseline, copy
`tool/scripts/ardy_batch_reference.rs` to
`crates/burn_ardy/examples/batch_reference.rs`, then capture native references:

```sh
cargo run -p burn_ardy --features tools,wgpu --example batch_reference -- \
  ARDY_BUNDLE batch-input.json batch-suite.json
```

For browser references, append `tool/scripts/ardy_batch_reference_web.rs` to the
baseline's `src/web_validation.rs`, build `web-validation` for WASM, run
wasm-bindgen, and invoke `capture_serial_webgpu(base, input_url, manifest_sha256)`.
Retain the result as the browser suite. Do not use native goldens as browser
bitwise baselines. On the candidate, run:

```sh
cargo run -p burn_ardy --features tools,wgpu --bin ardy-batch-validate -- \
  ARDY_BUNDLE batch-suite.json native.json
```

`tool/scripts/profile_webgpu.py` invokes `validate_batch_webgpu` with the same
three URL/digest arguments and the browser suite. It also profiles
`validate_text_webgpu` using the independent text reference. Fixture, executable
and module hashes are recorded in the provenance file; model artifacts remain
external to Git and crate packages.
