# Portable human inference and rigging

All runtime inference is Rust/Burn: Llama text encoding, ARDY motion generation,
SOMA body evaluation, and GEM-X image-to-pose prediction. Native WGPU and browser
WebGPU use the same model APIs. Python, PyTorch and ONNX Runtime are used only
for offline conversion and independent numerical reference tests. No text or
pose inference server is required.

| Crate | Responsibility |
|---|---|
| `burn_human_motion` | Motion/rig/condition contracts, SOMA animation interchange, sealed artifact manifests |
| `burn_human_inference` | Bounded Burnpack packing, authenticated transport/cache, tensor inventory and precision handling |
| `burn_ardy_text` | Llama 3 / LLM2Vec tokenizer, paged vocabulary, 32-layer text encoder and pooling |
| `burn_ardy` | Core27 tokenizer, DDIM/CFG, autoregressive history and trajectory conditioning |
| `burn_human::motion` | Bind-aware Core27 to Anny retargeting, phenotype proportions and coordinate conversion |
| `burn_soma` | Native PCA identity, skeleton fitting, procedural twists, correctives and batched GPU skinning; optional MHR adapter |
| `burn_mhr` | MHR identity/expression, scale-aware FK, correctives and GPU skinning |
| `burn_gem` | ViTPose, SAM image/body features, GEM prediction, camera decoding and fitted SOMA reconstruction |
| `bevy_burn_human::motion` | Shared graphics device, async loading/inference, motion/waypoint, body and image controls |

ARDY generates its trained Core27 representation and drives Anny. SOMA is a
separate full body backend with 77 public pose joints, 110 internal skin joints,
and the 18,056-vertex mid-LOD mesh. ARDY is not trained for SOMA; the crate boundaries
allow a future SOMA motion model without relabelling Core27 output.

## Build the ARDY bundle

Use the repository's nightly Rust toolchain. Python reference export needs an
upstream ARDY environment (PyTorch, NumPy, safetensors, hydra-core, pydantic,
vector-quantize-pytorch and upstream dependencies).

```sh
git clone https://github.com/nv-tlabs/ardy .cache/upstream/ardy
git -C .cache/upstream/ardy checkout 693f74d13b3d04a0a22ce127ee79c929dd89756b
hf download nvidia/ARDY-Core-RP-20FPS-Horizon40 \
  --revision abe6c43beb28c867c950acb824b9c4ef3d63fb76 \
  --local-dir .cache/ardy-models/ARDY-Core-RP-20FPS-Horizon40
PYTHONPATH=.cache/upstream/ardy python tool/scripts/ardy_reference.py \
  --source .cache/upstream/ardy --checkpoints .cache/ardy-models \
  --out .cache/ardy-reference
cargo run -p burn_ardy --features tools --bin ardy-pack -- \
  .cache/ardy-models/ARDY-Core-RP-20FPS-Horizon40 \
  .cache/ardy-reference/config.json \
  693f74d13b3d04a0a22ce127ee79c929dd89756b .cache/ardy-bundle
```

The destination must be new. The converter checks the checkpoint LFS digests,
exported configuration digest, source revision and complete 428-tensor
inventory. It maps source files, copies one logical stage at a time, sorts
tensor names, and writes deterministic Burnpack objects split into physical
parts. Keep the original model license. No model weights are committed here.

## Burnpack, CDN and cache contract

The model-neutral loader follows the bounded part-only strategy reviewed in
`burn_image`. It uses the published Burn 0.21 / Bevy 0.19 dependency graph, with
one compatible wgpu version and no dependency patches.

```text
bundle/
  manifest.json
  metadata/<tokenizer, rig, configuration and license files>
  parts/<sha256>.bin
```

Logical Burnpack objects are at most 64 MiB. Physical parts target 20 MiB and
must fit the 25,000,000-byte CDN ceiling. The loader authenticates each part,
object, tensor inventory, shape, dtype and tensor digest before use. Schema 2
also authenticates metadata assets and supports F16, Q4F32 and integer tables;
schema 1 F32 seals remain compatible. A caller can pin `content_sha256`.

Loading verifies/uploads one object at a time; it never assembles a full model
in the WASM heap. Llama's 128,256-row vocabulary is paged, with at most sixteen
512-row pages (64 MiB) in the host cache. GPU weight residency remains separate
from the host staging bound. The current F32 GEM suite stores approximately
7.37 GiB of weights; it requires a desktop-class GPU with sufficient memory.
This is not qualification for mobile devices or small integrated GPUs.

Native HTTP sources use the shared `$XDG_CACHE_HOME/burn-human` cache (falling
back to `$HOME/.cache/burn-human`). Browser sources use CacheStorage. Both have
an 8 GiB FIFO part budget. Corrupt cached parts are discarded and repaired;
unavailable storage falls back to verified source reads. Loading all model
families can evict older parts; it does not unload active GPU models.

Deploy at immutable versioned directories. Serve parts as
`application/octet-stream` with correct lengths and
`Cache-Control: public, max-age=31536000, immutable`; serve JSON as
`application/json`. Set CORS for a separate viewer origin. HTTPS is required
outside loopback for WebGPU/CacheStorage. HTTP range requests are unnecessary.
Retain the bundled license/attribution files. Weights are not included in Cargo
packages or this repository.

## Local Llama text conditioning

**Built with Meta Llama 3.** The public TREE Industries export supplies the
merged base plus LLM2Vec adapters. The converter preserves its INT4 weights;
Rust tokenization, attention, MLPs and pooling run in Burn. Native/browser
qualification uses WGPU/WebGPU, F32 activations and a bounded per-matrix GPU
dequantization before matmul. Packed weights remain on-device between calls.
This is not a fused INT4 matmul claim: Burn 0.21's packed transpose/mixed matmul
failed the independent `quant-parity` probe, so the portable implementation
uses exact dequantization before transposing, without patching Burn.

```sh
hf download TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX \
  --revision 7aa52a05d54c2fd9177366aeb3f88e9e7f3c5766 \
  --local-dir .cache/ardy-onnx
python tool/scripts/export_ardy_text.py .cache/ardy-onnx .cache/ardy-text-raw
cargo run -p burn_human_inference --features tools --bin human-model-pack -- \
  .cache/ardy-text-raw .cache/ardy-text-bundle
cargo run -p burn_ardy_text --features tools --bin text-encode -- \
  .cache/ardy-text-bundle embedding.json 'walk forward calmly'
```

The default attention is bidirectional, matching ARDY's LLM2Vec contract.
`AttentionMode::CausalExport` exists for comparison with the public ONNX graph,
whose actual mask is causal despite its descriptive model card. Tests compare
both modes against ONNX Runtime, changing only the reference mask for the
bidirectional case. Prompts use the upstream chat header, left padding and mean
pooling excluding the five header tokens and including EOT. Empty/control-token
prompts and inputs exceeding 64 tokens including the header are rejected.

For a complete native text-to-motion run, provide a JSON object mapping labels
to the motion requests below. Models are loaded once and embeddings, clips,
requests and measured times are written to a new output directory:

```sh
cargo run -p burn_ardy_text --features tools --bin ardy-text-run -- \
  .cache/ardy-text-bundle .cache/ardy-bundle requests.json generated
```

## Generate and view

```json
{
  "prompt": "walk forward calmly",
  "seed": 42,
  "frames": 120,
  "history_frames": 40,
  "diffusion_steps": 10,
  "text_guidance": 2.0,
  "trajectory_guidance": 2.0,
  "dense_trajectory": true,
  "waypoints": [
    {"frame":0,"position":[0,0,0],"heading":null,"constrain_height":false},
    {"frame":40,"position":[0,0,1],"heading":null,"constrain_height":false},
    {"frame":80,"position":[0.7,0,2],"heading":null,"constrain_height":false},
    {"frame":119,"position":[1.6,0,2.7],"heading":null,"constrain_height":false}
  ]
}
```

```sh
cargo run -p burn_ardy --features tools,wgpu --bin ardy-run -- \
  .cache/ardy-bundle request.json embedding.json clip.json
cargo run -p bevy_burn_human
```

In **Motion studio**, load the bundle URL/directory, load the local Llama text bundle, and generate. The prompt is encoded automatically when changed. You can also import
a previously computed embedding. Frames and history must be multiples of four.
World trajectory supports ground clicks, dragging a selected waypoint, frame
timing, heading and optional root height. Gold marks the requested path, blue
the generated root path, and green the source rig. Frame trajectory fits the
camera to the motion; ordinary camera orbit is disabled while placing points.
Playback has pause, scrub, speed, optional loop, per-joint rotation offsets and
JSON export. Disable motion animation to return to manual Anny posing.
The viewer can fit root height to Anny's leg proportions while preserving the
X/Z trajectory. Disable this option when exact source root height is required.
This is proportion compensation, not foot-locking or contact IK.

The browser has local file buttons for embeddings, clips and images. Native
inputs accept file paths or URLs. JSON clip export on native uses a new file
and reports an error instead of overwriting an existing one.

The demo's Burn device wraps Bevy's existing adapter, device and queue. Native
jobs run on a worker thread; browser jobs use async readbacks between windows.
Cancellation is checked between 40-frame windows. Autoregression retains at
most 160 history frames and a 200-frame conditioning window. `sample_window`
also accepts batched requests with shared window lengths; CFG uses three
branches per request. Fused attention uses Burn's backend attention primitive.

### SOMA interchange

```sh
python tool/scripts/soma_animation.py --animation motion.npz \
  --rig matching-soma-rig.json --fps 30 --out soma-animation.json
```

The rig must contain the animation's fitted local offsets and absolute bind
orientations in metres, in exact SOMA joint order. Identity/scales are already
baked into that rig. The converter accepts rotvec or SO(3) matrix poses and
metres/centimetres/millimetres, strips only an identity virtual Root, retains
identity metadata, and requires `joint_orient` for T-pose-relative animation.
Changing identity per frame needs separate rigs/clips. Import the result in
Motion studio. This animation-interchange path requires explicit fitted rig data; native SOMA body evaluation is described below.

## SOMA body evaluation

`Soma::prepare_identity` evaluates 128 native PCA coefficients and fits the
public skeleton once. `pose_batch` evaluates 1–256 poses with procedural twists,
full hand articulation, optional learned correctives and GPU linear blend
skinning. Sixty named bone-length ratios and a global scale control proportions.
Small skeleton fitting/FK calculations run on the host; the large shape,
corrective and mesh operations run in Burn. Reuse `PreparedIdentity` across
frames. Output uses metres, +Y up and +Z forward.

The optional `burn_soma/mhr` adapter evaluates MHR's 45 shape coefficients and
68 scales, transfers the mesh to SOMA topology with tetrahedral interpolation
and a compiled facial Laplacian solve, then fits the SOMA rig. MHR's full model
also supports 72 expression coefficients and the articulated parameters needed
by SAM's iterative decoder. MHR's internal units are centimetres; the SOMA
adapter converts them to metres.

```sh
git clone https://github.com/NVlabs/SOMA-X .cache/upstream/SOMA-X
git -C .cache/upstream/SOMA-X checkout cc1f3967755f8e36d187d2e26114633dbd651cd5
hf download nvidia/SOMA-X --revision 104578ed58857f6faa7592fb83d0a2dad43c36fa \
  --local-dir .cache/soma-assets
python tool/scripts/export_soma.py --upstream .cache/upstream/SOMA-X \
  --assets .cache/soma-assets --out .cache/soma-raw
cargo run -p burn_human_inference --features tools --bin human-model-pack -- \
  .cache/soma-raw .cache/soma-bundle
python tool/scripts/export_mhr.py .cache/soma-assets .cache/mhr-raw
python tool/scripts/export_mhr_soma_transfer.py --upstream .cache/upstream/SOMA-X \
  --assets .cache/soma-assets --out .cache/mhr-soma-transfer-raw
cargo run -p burn_human_inference --features tools --bin human-model-pack -- \
  .cache/mhr-raw .cache/mhr-bundle
cargo run -p burn_human_inference --features tools --bin human-model-pack -- \
  .cache/mhr-soma-transfer-raw .cache/mhr-soma-transfer-bundle
```

Install upstream conversion dependencies in an isolated environment; SOMA's USD,
Torch/Warp and MHR dependencies are offline tools, not Rust runtime dependencies.
Exports include independent fixtures for several identities, scale changes,
poses, hand joints and correctives. The Bevy **SOMA controls** tab loads the
bundle and exposes identity components, named bone proportions, global scale,
joint axis-angle controls, correctives, mesh/skeleton display and pose/identity
JSON export. Identity preparation is cached until its controls change.

## GEM-X image-to-pose inference

`Pipeline::estimate` takes a decoded RGB image, a square person crop and camera
intrinsics. It runs all seven components locally:

1. DINOv3 ViTPose with flip testing predicts the 77 SOMA heatmaps.
2. A separate DINOv3 backbone and the complete six-layer SAM body-token decoder
   produce image features, with intermediate MHR predictions and 2D/3D keypoint
   feedback. A backbone-only feature approximation is not used.
3. The 12-layer GEM network predicts identity, scale, pose and camera.
4. MHR identity transfer plus identity-fitted SOMA skinning produces the mesh.

This is the released fast regression path (zero latent, timestep 999), with
full image features. `GemDenoiser::denoise` also accepts explicit latents and
timesteps for future samplers. The image pipeline is single-person, single-frame;
it does not implement video tracking, multi-person detection or the optional
50-step diffusion refinement. The denoiser supports batched temporal inputs
independently. These are distinct from ARDY's autoregressive motion generation.

The predicted mesh is in camera coordinates: +X right, +Y down, +Z depth, metres.
The viewer rotates it to Y-up, centers it, and grounds it for display; exported
pose/camera values retain the original camera convention. GEM uses the fitted
bind convention and disables SOMA correctives, matching its reference wrapper.
Use native SOMA identities for canonical binding with correctives.

```sh
git clone --recursive https://github.com/NVlabs/GEM-X .cache/upstream/GEM-X
git -C .cache/upstream/GEM-X checkout 32992550dba114c62243fb55e361311972dce8f9
git -C .cache/upstream/GEM-X submodule update --init --recursive
hf download nvidia/GEM-X --revision 5ccf5ca3746c3620aa4016114f069a5f6ae399cd \
  --local-dir .cache/gem-assets
python tool/scripts/export_gem_vision.py --kind vitpose --assets .cache/gem-assets \
  --upstream .cache/upstream/GEM-X --out .cache/vitpose-raw --reference .cache/vitpose-reference.json
python tool/scripts/export_gem_vision.py --kind sam-body --assets .cache/gem-assets \
  --upstream .cache/upstream/GEM-X --mhr .cache/soma-assets/MHR/mhr_model_lod1.pt \
  --out .cache/sam-vision-raw --reference .cache/sam-vision-reference.json
python tool/scripts/export_gem_sam.py --assets .cache/gem-assets \
  --upstream .cache/upstream/GEM-X --mhr .cache/soma-assets/MHR/mhr_model_lod1.pt \
  --vision-reference .cache/sam-vision-reference.json --out .cache/sam-decoder-raw
python tool/scripts/export_gem_denoiser.py --assets .cache/gem-assets \
  --upstream .cache/upstream/GEM-X --out .cache/gem-denoiser-raw
```

Pack each raw directory with `human-model-pack`, as above. Construct a suite
whose relative paths work unchanged under a static CDN directory:

```sh
python tool/scripts/gem_suite.py --out .cache/gem-suite.json \
  --vitpose .cache/vitpose-bundle --sam-vision .cache/sam-vision-bundle \
  --sam-decoder .cache/sam-decoder-bundle --denoiser .cache/gem-denoiser-bundle \
  --mhr .cache/mhr-bundle --soma .cache/soma-bundle --transfer .cache/mhr-soma-transfer-bundle
cargo run -p burn_gem --features tools --bin gem-pose -- \
  .cache/gem-suite.json person.png crop-camera.json pose-estimate.json
```

`crop-camera.json` has `crop: {center: [cx,cy], size: pixels}` and
`camera: {focal: [fx,fy], center: [px,py]}` fields. In **Image pose**, load the
suite, select a PNG/JPEG, drag to center the person crop, adjust its size and
optionally the focal length, then select **Estimate pose**. Detected keypoints
appear on the image and the fitted SOMA body appears in the scene. The SOMA tab
can edit the inferred MHR identity and joint rotations. Pose/identity export is
a JSON document; it is separate from the explicit-rig animation interchange.

## Testing and measured evidence

Run model-free contracts and Anny retargeting tests with:

```sh
cargo test -p burn_human -p burn_human_motion -p burn_human_inference \
  -p burn_ardy -p burn_ardy_text -p burn_soma -p burn_mhr -p burn_gem \
  --features burn_ardy/transport,burn_soma/mhr --lib --tests
python tool/scripts/test_soma_animation.py
cargo check -p bevy_burn_human --target wasm32-unknown-unknown --no-default-features
```

Actual checkpoints have additional native and WebGPU tests; missing assets fail
rather than producing a skipped quality result. `text-validate` compares ten
prompt/mask cases against an independent ONNX oracle. `soma-validate`,
`mhr-validate` and `mhr-soma-validate` compare real identities and articulated
meshes against official Torch implementations. GEM has separate vision, SAM
iterative decoder and denoiser hooks, plus complete RGB image-to-mesh parity
against `gem_reference.py`. The fitted-bind regression is also covered by a
small CI test because joint-only checks cannot detect incorrect twist skinning.

```sh
cargo run -p burn_soma --features tools --bin soma-validate -- \
  .cache/soma-bundle .cache/soma-raw/reference.json soma-native.json
cargo run -p burn_gem --features tools --bin gem-validate -- \
  .cache/gem-suite.json person.png gem-reference.json gem-native.json
cargo build -p burn_gem --lib --target wasm32-unknown-unknown --features web-validation
wasm-bindgen --target web --out-dir .cache/gem-web \
  target/wasm32-unknown-unknown/debug/burn_gem.wasm
python -m http.server 8080 --bind 127.0.0.1
```

Use the wasm-bindgen CLI version matching Cargo.lock. From another terminal:

```sh
python tool/scripts/ardy_browser_validate.py --suite \
  --url 'http://127.0.0.1:8080/tool/web/gem-validation.html?bundle=http://127.0.0.1:8080/.cache/gem-suite.json&reference=http://127.0.0.1:8080/gem-reference.json&image=http://127.0.0.1:8080/person.png' \
  --out gem-browser.json
python tool/scripts/test_browser_transport.py --wasm-out .cache/gem-web \
  --base http://127.0.0.1:8080 --out transport.json
```

The same runner supports `tool/web/text-validation.html` (`--paged-text`),
`tool/web/soma-validation.html` and `tool/web/ardy-validation.html`. Build the
corresponding crate with `web-validation` and put bindgen outputs in
`.cache/ardy-text-web`, `.cache/soma-web`, or `tool/web/out` respectively. URLs
supply the bundle, reference fixture and manifest digest. The runner rejects
software adapters and verifies cold loads, zero-part-download warm loads, and
repair of exactly one deliberately corrupted part. Tests cover storage failures
and corrupt/oversized network responses separately.

[Portable inference evidence](evidence/portable-2026-09-26/) records checkpoint
pins, native/browser numerical errors, timings, cache requests and WASM heap
sizes. [Initial ARDY evidence](evidence/ardy-2026-09-26/) retains the earlier
checkpoint and Anny retargeting results; its host-ONNX measurements are historical.
The new text-to-motion diagnostics use Burn's bidirectional encoder. Prompt,
waypoint, contact and continuity measurements are diagnostic examples, not a
human-rated semantic or dataset-level FID benchmark. Throughput reports identify
whether model loading, decoder, mesh evaluation and output readback are included;
none of these numbers claim Bevy rendering frame rates.

CI runs contracts, native builds/tests and WASM compilation. Full checkpoint
qualification additionally requires the pinned downloaded artifacts and a GPU.
