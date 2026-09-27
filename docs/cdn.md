# Published model crates and CDN bundles

| Crate | Release | Responsibility |
| --- | --- | --- |
| `burn_ardy` | 0.1.3 | Batched ARDY Core27 motion, trajectory conditioning, checkpoint loading |
| `burn_llama` | 0.1.1 | Llama 3 8B / LLM2Vec text conditioning, paged vocabulary |
| `burn_soma` | 0.1.2 | SOMA-X identity, rig, correctives, skinning and optional MHR transfer |
| `burn_gemx` | 0.1.2 | GEM-X image inference and composition of seven component releases |
| `burn_mhr` | 0.1.1 | MHR evaluation and its independent checkpoint |
| `burn_human_inference` | 0.1.3 | Model-neutral transport, bounded cache, Burnpack/tensor verification |

`burn_llama` replaces `burn_ardy_text` 0.1.0; `burn_gemx` replaces `burn_gem`
0.1.0. Their implementations live in the newly named crates. The previously
published packages remain available. `burn_human` 0.5.1 supplies Anny;
`bevy_burn_human` 0.6.0 integrates the model crates with its default `studio`
feature. All use published Burn 0.21 / Bevy 0.19 dependencies
and the same wgpu 29 version, without patches.

## Loading

Each model crate owns a `pretrained` module with immutable bundle names and
compiled-in manifest SHA-256 identities. `burn_human_inference` has no model
catalog or CDN policy. The default root is `https://aberration.technology/model`.

```rust,ignore
let ardy = burn_ardy::Ardy::<B>::load_pretrained(&device, |done, total| {}).await?;
let llama = burn_llama::TextEncoder::<B>::load_pretrained(&device, |done, total| {}).await?;
let soma = burn_soma::Soma::<B>::load_pretrained(&device, |done, total| {}).await?;
let gem = burn_gemx::pipeline::Pipeline::<B>::load_pretrained(
    &device, |stage, done, total| {},
).await?;
```

All four also expose `load_pretrained_from(root, device, progress)`. A root
can be a native directory containing the bundle directories or an HTTP CDN
mirror; changing it retains the same trust anchors. GEM-X resolves its four
perception components and the `burn_mhr` / `burn_soma` catalogs together.
Individual models expose `load_artifact` for explicitly pinned custom exports.
GEM-X also accepts a `PipelineArtifacts` graph or a relative suite JSON; the
canonical public suite URL is checked against its compiled-in file hash.

The viewer starts with the pinned public URLs and manifest identities. Custom
bundle paths require the matching SHA-256. Native and browser loading happens
only when requested by the user; fetching a crate does not download weights.

`burn_llama` currently provides the ARDY LLM2Vec encoder, not chat generation.
GEM-X currently provides fast single-person, single-frame regression. See
[motion.md](motion.md) for model scope, controls and measured numerical quality.

## Transport and cache

The upload layout follows `burn_image`'s immutable part-only deployment contract:

```text
aberration.technology/model/
  ardy/core-rp-20fps-h40/v1/{manifest.json, metadata/, parts/}
  llama/ardy-llm2vec-8b/v1/{manifest.json, metadata/, parts/}
  soma-x/v1/
    body/{manifest.json, metadata/, parts/}
    mhr/{manifest.json, metadata/, parts/}
    transfer/{manifest.json, metadata/, parts/}
  gemx/v1/
    vitpose/{manifest.json, metadata/, parts/}
    sam-vision/{manifest.json, metadata/, parts/}
    sam-decoder/{manifest.json, metadata/, parts/}
    denoiser/{manifest.json, metadata/, parts/}
    suite.json
```

Every manifest seals model/source revisions, configuration, tensor inventory,
dtype, quantization, logical objects, physical parts, metadata sizes and hashes.
The release tool decodes every Burnpack and rejects missing, unknown, duplicate,
wrong-shaped, corrupt or non-finite tensors. Only manifest-declared metadata
and content-addressed physical parts enter the upload tree.

Logical objects are bounded to 64 MiB. Physical parts target 20 MiB and must
remain below 25,000,000 bytes. The current loader fetches and authenticates
whole bounded parts; unlike `burn_image`'s 4 MiB range cache, it does not require
HTTP Range support. It never assembles the complete model in WASM memory.

Native HTTP reads reuse a shared connection pool across shards. Loading uses `$XDG_CACHE_HOME/burn-human`, then
`$LOCALAPPDATA/burn-human`, then `$HOME/.cache/burn-human`. Browser loading uses
CacheStorage `burn-human-motion-parts-v1`. Both use an 8 GiB FIFO budget,
authenticate cache hits, repair corruption and tolerate unavailable storage.
Manifest and compact metadata requests are repeated on warm loads. Cache
eviction may cause downloads when switching between large models; all models
together exceed the cache budget. GPU residency is separate from these bounds.

## Prepare and upload

Run the pinned exporters described in [motion.md](motion.md), then from the
repository root:

```sh
cargo run -p burn_human_tool --bin human-cdn-prepare -- \
  tool/cdn-release.json .artifacts/cdn-upload-human-v1
cd .artifacts/cdn-upload-human-v1
sha256sum --check SHA256SUMS
```

The output directory must be new. The plan pins the earlier validated source
exports and supplements their missing license notices before resealing. Tensor
bytes do not change. Source paths and supplemental notices are listed in
`tool/cdn-release.json`; source exports must remain immutable when their parts
are hardlinked into staging. Copying to a different filesystem is also supported.

The prepared release contains **14,089,662,896 weight bytes across 2,561
per-bundle unique parts**, plus compact metadata. It includes all vocabulary
pages and the complete GEM-X dependency closure; SOMA is shared. See the
[compact release inventory](releases/cdn-2026-09-26.json).

Upload the contents of `aberration.technology/model/` to the matching CDN root,
preserving directory names. Family/version directories are immutable; use `v2`
for a future changed release instead of overwriting `v1`. Hashes and source
revisions remain in manifests, not directory names. Upload only to empty
component prefixes. Publish
each `manifest.json` after its parts/metadata and the suite JSON after all its
dependencies. Serve JSON as `application/json`, parts as
`application/octet-stream`, with `Access-Control-Allow-Origin: *` and
`Cache-Control: public, max-age=31536000, immutable`. GET must return the exact
bytes without transformations, authentication or HTML error wrappers.
Keep `UPLOAD.md`, `release.json` and `SHA256SUMS` outside the model prefixes.

The public suite will be:
`https://aberration.technology/model/gemx/v1/suite.json`.
The public deployment passed complete native shard verification and real
WebGPU inference with cold, warm and corrupted-cache loads. See
[public CDN and studio qualification](evidence/studio-2026-09-26/README.md).

## Validate native and browser downloads

To serve the staged model root with CDN headers on loopback:

```sh
python3 tool/scripts/serve_cdn.py --port 8887 --directory \
  .artifacts/cdn-upload-human-v1/aberration.technology/model
```

The native qualification downloads and decodes every component, then verifies
zero weight bytes on a warm load and exact single-part corruption repair.
Components use isolated cold caches so FIFO eviction cannot hide a miss:

```sh
cargo run -p burn_human_tool --bin human-cdn-check -- \
  docs/releases/cdn-2026-09-26.json http://127.0.0.1:8887 \
  .cache/cdn-native-check .cache/cdn-native-check.json
```

Replace the HTTP root with `https://aberration.technology/model` after upload;
use a fresh cache directory and output report for each qualification run.

Build the four WASM validators with their `web-validation` features and the
matching `wasm-bindgen` CLI, following [motion.md](motion.md). Serve this
checkout separately on loopback port 8866. `tool/scripts/ardy_browser_validate.py`
runs cold, warm and corrupt-cache passes in hardware Chrome WebGPU; use the
pages in `tool/web`, the released bundle URL and its manifest digest, and the
pinned reference fixtures from the exporters. For Llama add `--paged-text`;
for GEM-X add `--suite` and an `image` query parameter. This validates real
inference and numerical parity as well as network/cache behavior. The local
model server uses a different origin to exercise CORS. The same commands can
target the uploaded CDN without rebuilding WASM.

The reference GPU has 96 GiB VRAM. Bounded transport does not establish fit on
small integrated GPUs or mobile devices. See the linked numerical/performance
evidence for qualified hardware and model limitations.
