# Batch stability and browser startup qualification

This follow-up closes the measured batch-size reproducibility gap in the
[preceding qualification](../batch-2026-09-27/README.md) and separates production
startup from the synchronous diagnostic hooks. Model weights, manifests and CDN
paths are unchanged. Burn 0.21 and Bevy 0.19.1 still share WGPU 29.0.4 without
patches.

## Implementation

ARDY uses fixed dense matmul, attention and parallel reduction strategies on the
explicit, unfused WGPU backend. Shape-dependent autotuning had changed reduction
order enough to cross FSQ rounding boundaries. The replacement preserves batched
GPU dispatches. It does not run actors serially or quantize outputs to hide drift.
Other backends retain their own dispatch; the fixed WGPU path has a portable
kernel fallback for adapters without the preferred capabilities.

Browser generation yields tasks between transformer stages without adding GPU
fences or readbacks. Native builds compile out these browser suspension points.
Synchronous APIs remain available. Loading uses WebCrypto, authenticated shared
Burnpack views, bounded finite-value scans and task boundaries between cache
operations. Upload completion is awaited periodically during loading to prevent
large queues of weights from flushing during a later small write. It introduces
no per-layer inference synchronization.

Llama's tokenizer builds vocabulary and merge tables in groups of 2,048 entries.
Hugging Face normalization, pretokenization, added tokens and chat processing are
retained, as is the BPE rank/leftmost merge policy. An external-fixture test compares
exact token IDs on 1,028 Unicode, punctuation and long repeated inputs. Text
encoding also yields between projection, attention and feed-forward stages.

## Numerical gates

The batch suite covers every actor count from 1 through 8, histories of
0/4/40/160 frames, 40/44/80/120/240-frame clips, different and duplicate seeds,
wrapped headings, and dense/sparse curved paths. The added 3/5/6/7-actor cases
reuse corresponding independently captured serial references from the earlier
eight-actor fixture. They do not generate new goldens from the candidate.

Continuous denoising, ten-step DDIM sampling and identical-input decoding are
gated at 1e-6 maximum absolute difference. FSQ code changes must be exactly zero.
Complete serial/batch clips and actor permutations separately enforce maximum
FK error of 1 micrometre, RMS FK error of 0.1 micrometre, rotation error below
1e-6 radians and no contact disagreements. Legacy serial golden gates and the
original PyTorch checkpoint tolerances remain unchanged.

Both platforms pass all 11 cases with **zero changed FSQ codes and zero FK/root
position differences** between serial and batched generation. Actor permutations
match to the same precision; quaternion angular comparison has only floating-point
roundoff. The largest native difference from the older independent serial golden
is 1.500 mm; browser golden FK/root differences are zero. Cross-batch
consistency and compatibility with old kernels are distinct checks.

SOMA, GEM-X and Llama retain their independent checkpoint checks. GEM-X includes
image preprocessing and fitted SOMA reconstruction. Llama checks token IDs,
causal/bidirectional attention and instruction-excluding pooling. These examples
are numerical regressions, not a human-rated or dataset-level semantic benchmark.

## Startup and throughput

The final browser probes observed zero tasks longer than 50 ms in the production
ARDY/SOMA/GEM-X paths and the complete Llama reference suite. Loading uses cold
browser CacheStorage; ARDY fetches the public CDN and the other runs use the
identical staged bytes over loopback. These load times are not CDN speed estimates.

| Browser run | Load, s | First inference, s | Warm median, s | Long tasks | Max RAF gap, ms | Peak WASM heap, MiB |
| --- | --- | --- | --- | --- | --- | --- |
| ARDY / public CDN | 13.482 | 0.899 | 0.223 | 0 | 100.0 | 118.75 |
| SOMA | 0.798 | 0.117 | 0.005 | 0 | 50.0 | 124.50 |
| GEM-X | 67.961 | 1.769 | 0.516 | 0 | 100.0 | 220.69 |
| Llama / reference suite | 36.922 | 0.676 | 0.219 | 0 | 83.4 | 169.38 |

ARDY generates one 40-frame clip here. SOMA uses a prepared neutral identity and
correctives; identity preparation is a separate recorded phase. GEM-X uses the
reference image with its crop/camera conditions. Llama's first value is its first
causal reference prompt, including lazy vocabulary access and cold kernels; its
report also covers bidirectional encoding. WASM heap values exclude GPU and
browser-managed memory.

See [complete-clip throughput](timings.md), [native](native.json),
[browser](browser.json), [ARDY production](ardy-production.json),
[SOMA production](soma-production.json), [GEM-X production](gemx-production.json)
and [Llama](llama-browser.json). The synchronous batch diagnostic still records
a 112 ms task; it includes checkpoint hooks and large independent references.
The production API is measured separately.

Production profiles call the real load/generate, load/pose and image-inference
APIs. Their output reads synchronize GPU completion. They exclude WASM module
startup, large reference parsing and diagnostic reconstruction hooks. ARDY timing
also excludes text encoding; Llama is measured separately. A browser long task
means a main-thread task longer than 50 ms. Animation-frame gaps measure callback
scheduling, not rendered frame delivery or input-to-photon latency. Zero observed
long tasks is not a promise of 60 FPS or zero cold-compilation latency.

Diagnostic CPU profiles localized the former shared-loader stalls to repeated
Rust SHA-256, object/tensor staging copies, adjacent cache/hash work and a deferred
GPU upload flush. After the loading changes, the remaining 63 ms SOMA and 83 ms
GEM-X diagnostic tasks included large reference JSON parsing and PNG decoding
before reconstruction. Separate production profiles avoid those extra hooks.
Profiler-instrumented samples are diagnostic and are not used as throughput
measurements. Native and browser throughput use their respective optimized dev
and release profiles on a shared RTX PRO 6000 Blackwell / driver 610.43.02.
Browser tests use Chromium 153 hardware WebGPU, without a software fallback.
Qualification covers this adapter; cross-device bitwise equivalence is not implied.

## Studio and release

The Pages workflow explicitly builds `--no-default-features --features studio`
and pins wasm-bindgen-cli to Cargo.lock. It publishes `build.json` with the source
commit, `studio` feature, target and profile. The studio continues to expose text
and waypoint motion conditions, SOMA controls, image/crop inputs and PanOrbit
camera controls. GPU mesh handoff remains resident on the shared Bevy/Burn device.

Local release gates pass: 39 workspace tests, strict all-target Clippy with model
tools, scoped formatting, all native tools and the explicit studio WASM release
build. All three normally ignored hardware/external-fixture tests were then run
successfully. Browser transport fault checks pass, and all eight package dry runs
verify without bypassing compilation; [archive inspection](packages.json) finds
no weights.

Rendered studio smoke checks cover [world-space waypoints](studio-waypoints.png),
[SOMA loading and a wave pose](studio-soma-wave.png), camera orbiting, and
[image selection/crop input](studio-image-input.png). The body in the input
screenshot is the existing SOMA wave, not a newly inferred image pose.
Commands, input/module hashes and GPU state are in [provenance](provenance.json).

| Crate | Version |
| --- | --- |
| burn_human_motion | 0.1.1 |
| burn_human_inference | 0.1.4 |
| burn_ardy | 0.1.4 |
| burn_llama | 0.1.2 |
| burn_mhr | 0.1.2 |
| burn_soma | 0.1.3 |
| burn_gemx | 0.1.3 |
| bevy_burn_human | 0.6.1 |

## Reproduction

Use the external model bundles and independent fixture preparation described in
[the earlier batch qualification](../batch-2026-09-27/README.md#reproduction).
`prepare_ardy_batch.py` now includes all 11 cases. Run `ardy-batch-validate` for
native batch checks and `validate_batch_webgpu` through
`tool/scripts/profile_webgpu.py` for the browser. The same profiler can invoke
`profile_generation_webgpu`, `profile_soma_webgpu`, `profile_gem_webgpu` and
`validate_text_webgpu`; reports retain their exact arguments.

The tokenizer regression uses the pinned external tokenizer asset:

```sh
cargo test -p burn_llama --features tools --lib incremental_bpe -- --ignored
```

`tool/scripts/test_browser_transport.py` verifies the generated production JS,
including same-size corruption, warm hits without network and cache failures.
Hardware regression tests cover ARDY actor ordering/cancellation and Bevy's
allocation lease, normals, bounds and topology changes. See the
[performance guide](../../performance.md) for these commands.
