# Modular crate and CDN release qualification

This qualifies the family-organized `v1` upload tree and the new `burn_llama` /
`burn_gemx` crate names. The release catalog is
[cdn-2026-09-26.json](../../releases/cdn-2026-09-26.json); preparation and loading
commands are in [cdn.md](../../cdn.md).

## Artifact and transport checks

The upload tree contains four top-level families: `ardy`, `llama`, `soma-x`
and `gemx`. All nine component bundles were independently decoded and checked
for tensor inventory, shape, dtype, byte hash and finite values. All 2,603
published files passed an independent SHA-256 sweep. Re-running the Rust
packer from the pinned source exports reproduced the grouped tree's checksum
inventory and catalog exactly. The earlier flat staging tree was removed.

Only metadata changed relative to the prior numerical qualification: ARDY now
has manifest-bound license/attribution files, Llama adds the LLM2Vec license,
and the two vision exports add their third-party notices. Model tensor bytes
are unchanged. New metadata seals and the relative GEM-X suite file are pinned
in the owning model crates.

Native HTTP qualification covered every part, including all 636 Llama parts
and its lazy vocabulary pages. Each of the nine components passed an isolated
cold download, a warm load with **zero downloaded weight bytes**, and a corrupt
cache load downloading exactly the damaged part's bytes. See
[native-transport.json](native-transport.json). Compact metadata is fetched and
authenticated again; this is not an offline-loading claim.

Hardware Chrome WebGPU qualification used a separate-origin HTTP mirror to
exercise CORS and ran real numerical fixtures on every reload:

| Pipeline | Cold part fetches | Warm part fetches | Corrupt-cache part fetches |
| --- | ---: | ---: | ---: |
| ARDY | 59 | 0 | 1 |
| Llama / fixture vocabulary pages | 399 | 0 | 1 |
| SOMA | 80 | 0 | 1 |
| GEM-X / seven components | 1,866 | 0 | 1 |

Browser counts measure part fetches observed by Playwright; the browser's HTTP
cache can also satisfy a repair fetch. CacheStorage entries are authenticated
before use. The validation harness's nested JSON promise was fixed before
these recorded passes. The system Chrome did not expose WebGPU; the recorded
passes used Chromium headless shell **153.0.8010.12**, with a verified NVIDIA
Blackwell hardware adapter and no software fallback.

## Numerical and runtime checks

Native WGPU and browser WebGPU both passed the unchanged real-checkpoint
reference fixtures for all four models. Full GEM-X image-to-mesh maximum
vertex coordinate error is **0.391 mm** on both backends. The new reports
retain per-stage errors, synchronization scope and timing arrays; they are
not synthetic shape-only tests. Prior independent-oracle provenance and
semantic motion diagnostics remain in
[portable-2026-09-26](../portable-2026-09-26/README.md).

Full GEM-X warm inference measured **0.50–0.56 s** in the browser and
**0.71–0.74 s** natively, including output readback and excluding model load.
Browser Llama warm prompts measured **0.17–0.25 s**. These are observations on
the shared RTX PRO 6000 Blackwell workstation (96 GiB VRAM), not isolated
regression benchmarks or mobile-device qualification. GEM-X's measured WASM
linear-memory peak was **1,368 MiB**, excluding GPU memory and CacheStorage.

The Bevy viewer was rebuilt for WASM and loaded its default pinned public
SOMA URL through an explicit local mirror interception: 80 authenticated
parts, Ready status, evaluated mesh, rig controls and export UI. See
[viewer.json](viewer.json) and [screenshot](viewer-soma-cdn.png). This tests the
default catalog wiring without claiming that the public CDN is deployed.

Workspace validation passed **29 unit/integration tests**, Clippy for all
model tools with warnings denied, all four WASM validators, the Bevy WASM
build, and **eight verified publication dry-runs**. Package inspection found
no weights, build outputs or cache directories in the crate archives. The
resolved graph still has a single **wgpu 29.0.4**, with no dependency patches.
See [local-checks.json](local-checks.json) and
[package-inspection.json](package-inspection.json).

## Deployment boundary

The public manifest URLs returned **403 before upload**; see
[preflight](public-preflight-before-upload.json). Public native/browser
download and cache qualification must be run after the user uploads the
prepared tree. All HTTP/inference results above use the verified local mirror.
