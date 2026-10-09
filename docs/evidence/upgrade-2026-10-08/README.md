# Bevy 0.20 / Burn 0.22 qualification

This release uses Bevy 0.20.0, Burn 0.22.0, CubeCL 0.11.0,
Cubek 0.3.0 and one WGPU 30.0.1 dependency, without dependency patches.
The studio uses Bevy Feathers, editable text, UI picking and the built-in
pan-orbit controller. Model loading, inference and rendering remain portable
Rust on native WGPU and browser WebGPU.

Qualification uses an NVIDIA RTX PRO 6000 Blackwell (96 GiB), driver 610.43.02,
Linux, Rust nightly 1.100.0 (2026-08-30) and Chromium 153.0.8010.12.
The repository and CI use `nightly-2026-08-31` (Rust compiler commit dated
2026-08-30). The initial floating-nightly CI failed in upstream `branches`
0.4.6 because a newer compiler removed `core::intrinsics::abort`; the pinned
toolchain matches the local qualification without dependency patches.
These results do not establish minimum-memory GPU or mobile compatibility.

## Numerical and model checks

- [Native ARDY checkpoint hooks](ardy-native.json): maximum denoiser error
  0.000238, decoder error 0.000203 and FK error 4.77e-7 m versus the independent
  PyTorch fixture.
- [Native](ardy-batch-native.json) and [browser](ardy-batch-browser.json) ARDY
  suites cover 11 cases, 1–8 actors, 0/4/40/160 history frames, 40–240-frame
  clips, actor reordering, curved paths and cancellation. Strict fixed-input
  FSQ and serial/batch pose gates pass. The existing single-actor baselines
  also pass. This is numerical qualification of the pinned fixtures, rather
  than a claim that arbitrary generated actions satisfy human quality review.
- [Native](llama-native.json) and [browser](llama-browser.json) Llama checks
  match tokenizer IDs, pooling and ten causal/bidirectional pooled outputs.
  Cosines exceed 0.999998; the current CDN's block-Q4 weights load directly.
- [Native](soma-native.json) and [browser](soma-browser.json) SOMA check four
  identities/poses, scales, correctives, skeleton transforms and batches
  1/8/32. Maximum vertex errors are 1.43e-6 m and 1.79e-6 m, respectively.
- [Native](gemx-native.json) and [browser](gem-browser.json) GEM-X validate
  the real image pipeline and independent fitted-body reconstruction.
  Maximum inferred-vertex error is 0.000391 m on both platforms.

## Runtime and transport

The Burn frontend now dispatches backend-free tensors through `Device`.
The GPU export bridge explicitly materializes strided/tiled tensors on device;
its hardware regression covers slices, strides, allocation lifetime, normals
and bounds. The resident material was migrated to WESL, including the new PBR
module paths and shader constants; a real shared-device SOMA render passes
with [zero compiler/console errors](resident-render.json). Contiguous buffers retain the existing allocation lease and offset.

Burn 0.22 introduces many static compiler registrations. Repeated LLD
command-export constructors made the first browser trial unusably slow. The
final entry points call a shared, guarded initializer once. The library does
not install an application start function. CI examines named WASM artifacts
before stripping and rejects repeated constructor wrappers. The studio uses a
uniquely named explicit startup wrapper, and its loading page waits for the
Bevy canvas before declaring startup complete.

[Public-CDN browser ARDY](ardy-production-public.json) generates a 40-frame clip
in 1.124 s on first use and 0.245–0.273 s warm, excluding text encoding, model
loading and module startup. Loading's p95 frame gap is 83.3 ms; generation's
maximum cold/warm gaps are 183.3/100 ms. This is a production-path measurement,
separate from the synchronous parity hooks in the batch qualification.

Warm SOMA shape/pose evaluation takes approximately 1.66/1.78/2.89 ms native
and 4.3/4.5/6.4 ms browser for batches 1/8/32. Warm browser Llama encoding
is 0.400–0.448 s. Warm browser GEM-X stage totals are 0.801–0.867 s.
The raw reports state their synchronization and scope. These measurements do
not measure an entire rendered frame or guarantee 60 FPS during inference.

[All nine public CDN components](cdn-native.json) pass authenticated cold
loading, zero-part-download warm reads and repair of exactly one damaged
cached part. All tensor inventories, legacy-Q4 decoding and paged vocabulary
objects are covered. [Browser transport fault checks](browser-transport.json)
cover CacheStorage hits, damaged parts, unavailable storage, bounded responses
and corrupted network bytes. The public bundles and their seals are unchanged.

First-use shader compilation and native autotuning still cost time: native
SOMA's first identity preparation takes about 9.7 s, and native GEM-X's cold
inference includes substantial kernel benchmarking. The browser GEM-X fixture
loads seven components in about 79 s and its first inference is slower than
warm inference. These are hardware-specific observations, not eliminated cold
starts. Models are downloaded on demand and use the existing bounded cache.

## Studio and build checks

The qualification includes native tests, strict workspace/tool Clippy,
compiled README example, native startup, model WebGPU checks, and a release
WASM studio below GitHub's 100 MiB file limit. A dynamic-form regression
checks that structural changes preserve text focus, editor identity and pending
keystrokes. The [studio interaction record](studio-interactions.json) has no
page or console errors. Manual inspection of captured frames confirms local
Llama prompt encoding, a six-second ARDY clip with requested/generated world
paths, Anny playback and orbit/pan/frame controls; SOMA wave, live scale and
hand filtering; and file-picker image input, crop controls and GEM-X pose
estimation on the same device as rendering.

- [ARDY prompt and trajectory](studio-motion-generated.png)
- [SOMA pose and joint filtering](studio-joint-filter-hand.png)
- [GEM-X image and rendered pose](studio-gem-frame.png)

The browser interaction sequence uses physical-duration mouse/key events.
These checks cover the tested desktop browser and Linux GPU; platform CI
covers native macOS/Windows compilation and contracts, not graphical or
minimum-memory hardware qualification.
