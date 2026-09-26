# Public CDN and studio qualification

This run targets **https://aberration.technology/model**, after the grouped
bundle upload completed on 2026-09-26. Model requests used the public origin;
there was no local mirror or HTTP interception. The viewer and validation
pages were served separately on loopback. The release inventory remains
[cdn-2026-09-26.json](../../releases/cdn-2026-09-26.json).

## Upload, native transport and browser caches

All nine component bundles passed full native download, Burnpack decoding,
tensor inventory, shape, dtype, hash and finiteness checks. This covers
**14,089,662,896 weight bytes and 2,561 per-bundle unique parts**, including
every Llama vocabulary page. All **42 manifests, metadata files and the suite**
matched the staged files byte for byte. CORS headers allow the viewer origin.
The CDN supplies a seven-day immutable cache lifetime. License text MIME
types vary, but the authenticated bytes are correct.

Each component used an isolated cold native cache, then a warm load with zero
downloaded weight bytes, then intentional corruption. Every damaged entry
was repaired by downloading exactly that part. See
[native-transport.json](native-transport.json) and [metadata.json](metadata.json).
Compact metadata is fetched again on warm loads; this is not an offline test.

Real hardware Chrome WebGPU ran numerical inference on each cold, warm and
corrupt-cache load:

| Pipeline | Cold part fetches | Warm | Corrupt-cache repair |
| --- | ---: | ---: | ---: |
| ARDY | 59 | 0 | 1 |
| Llama, fixture vocabulary pages | 399 | 0 | 1 |
| SOMA | 80 | 0 | 1 |
| GEM-X, seven components | 1,866 | 0 | 1 |

The four `browser-*.json` reports retain numerical results, memory, timings,
adapter identity and network counts. No recorded model request failed.
Counts are observed part fetches; Chrome's HTTP cache may satisfy a repair
fetch. The independent native sweep covers vocabulary pages not required by
the browser's prompt fixtures. The 8 GiB cache budget can evict older parts
when switching between models; the whole catalog exceeds that budget.

## Numerical and performance scope

Native WGPU and browser WebGPU pass the real-checkpoint ARDY, Llama, SOMA and
GEM-X reference fixtures. Maximum GEM-X output vertex-coordinate error is
**0.391 mm** on both backends. Native SOMA's four pose/identity cases stay
below **1.5 micrometres**. Llama's minimum reference embedding cosine across
the tested prompts/modes exceeds **0.9999984** on both backends.

The reports retain the earlier independent reference provenance and exact
per-stage tolerances. ARDY's tensor parity fixture uses random embeddings;
it is not a semantic motion-quality score. Prompt-driven viewer generation
is an interaction smoke test. See the broader semantic diagnostics in
[portable qualification](../portable-2026-09-26/README.md).

Warm GEM-X image-to-mesh inference including output readback measured
**0.705–0.741 s native** and **0.517–0.538 s browser**. Browser GEM-X WASM linear
memory peaked at **873.75 MiB**, excluding GPU memory and CacheStorage. These
are observations on a shared RTX PRO 6000 Blackwell workstation with 96 GiB
VRAM and driver 610.43.02, not isolated performance regressions or minimum
hardware requirements. Chrome headless shell was 153.0.8010.12 with a NVIDIA
Blackwell adapter and no software fallback.

## Fixes discovered against the public origin

Native GEM-X previously concatenated relative suite paths containing `../`.
The public object store rejected these keys, while browser fetch and the
local test server normalized them. `burn_gemx` 0.1.1 now resolves suite
components as URLs, with regression coverage for shared SOMA dependencies,
root-relative paths, query strings and local filesystem paths. The native
public image-to-mesh validator passes after this fix. No bundle bytes change.

`burn_human_inference` 0.1.2 reuses a native HTTP connection pool for successive
shards. A local keep-alive regression test verifies two reads over one accepted
connection; integrity, size bounds and cache authentication remain in force.

The [studio guide](../../studio.md) documents the revised controls and the
distinction between ARDY's Anny/Core rig and GEM-X's SOMA output.

## Interactive viewer checks

Native Linux/Vulkan and the rebuilt browser/WebGPU viewer used the public
catalog on Bevy's shared GPU device. Native checks used a private X11 display
with a 1280 × 720 application window. Browser checks included 1500 × 1050 and
800 × 600 viewports. These are exercised workflows, not automated semantic
or visual-quality scores; see [viewer.json](viewer.json).

- Motion: one-click ARDY/Llama loading, editable prompt, six-second generation,
  Turn preset, requested/generated path display, playback and scrubbing. The
  native [120-frame export](native-motion.json) imported and played in the
  rebuilt [browser viewer](browser-native-clip.png).
- World controls: picking and dragging waypoint markers, ordered timing,
  camera orbit, right-button pan while editing, Escape to finish, and frame
  controls. See [waypoint editing](browser-waypoint-editing.png).
- SOMA: real body loading, live T pose/Wave edits, highlighted rig, native
  JSON save dialog, inferred MHR controls and switching to a canonical SOMA
  identity. See [rig controls](browser-soma-controls.png).
- Image pose: native and browser PNG file selection, crop drag, GEM-X
  inference, 2D keypoints, fitted SOMA body and JSON export. See
  [native](native-image-pose.png), [browser](browser-image-pose.png) and
  the [exported pose](browser-image-pose.json).
- Stale conditions: changing the crop removes old keypoint overlays and
  disables image-result export until re-estimation. See
  [changed condition](browser-stale-image-condition.png).
- Small windows: scrolling reaches the lower controls; missing WebGPU gives
  an actionable startup message. See [small viewport](browser-small-viewport.png)
  and [startup check](browser-startup.json).
- Anny: shape sliders update the GPU-skinned body; its editor stays below the
  compact mode/camera panel, leaving the subject visible at 1280 × 720. See
  [native controls](native-anny-controls.png).

Local validation passed **33 unit/integration tests**, workspace Clippy with
all model tools and warnings denied, the native viewer and release WASM
viewer builds, and three verified publication dry-runs. Archives contain no
model weights or build/cache artifacts. Changed Rust files are formatted;
whole-workspace formatting still reports pre-existing differences elsewhere.
The graph has one **wgpu 29.0.4**, without patches. See
[local-checks.json](local-checks.json) and
[package-inspection.json](package-inspection.json).
