# GPU inference and rendering

ARDY, GEM-X and SOMA use Burn 0.21 on native WGPU and browser WebGPU. Native
WGPU enables kernel autotuning, including backend-selected attention. WebGPU
keeps the portable kernel configuration: the tested WebGPU autotuning trial
failed SOMA shape/pose parity.
First use of a new shape can compile and benchmark kernels; warm timings must
be reported separately. These crates do **not** enable Burn graph fusion:
the tested native Burn 0.21 fusion configuration also failed SOMA numerical parity.
Downstream applications should qualify any additional backend features.

## Keep outputs on the device

`burn_gemx::pipeline::Pipeline::estimate_resident` returns a
`ResidentPoseEstimate<B>`. Its vertices and prepared identity remain on the
device. Use `into_host().await` only when exporting or processing vertices on
the CPU. The existing `estimate` API still returns host vertices for callers
that need them. Small pose parameters/keypoints remain host data because the
procedural rig and public output formats require them.

The Bevy plugin initializes Burn with Bevy's device and queue. Its explicit
`burn_human_inference::gpu::WgpuBackend` remains the unfused Cube backend even
if another dependency enables Burn's fusion alias. After SOMA skinning:

1. Export a tensor allocation lease, flushing queued compute without waiting.
2. Compute area-weighted normals and pack resident vertices in one GPU pass.
3. Read six floats of bounds asynchronously for grounding, culling and framing.
4. Bind the new buffer to a material; the topology and mesh attributes stay fixed.

The actor opts out of automatic mesh bounds with `NoAutoAabb`, since its static
mesh positions are placeholders. GPU-evaluated bounds still drive frustum
culling, including when switching from the SOMA controls to an inferred body.

Per-pose rendering does not read the full vertex array, compute normals on the
CPU, or upload positions/normals through `Assets<Mesh>`. A fresh output buffer
avoids overwriting data still used by extracted render frames. Retained frames
bound residency; this is not an unbounded buffer history. Topology uploads once
per loaded model. The material supports the forward, prepass and deferred
vertex interfaces.

`gpu::export` borrows the underlying allocation, including its offset and
logical byte length. Keep the returned lease until external GPU work completes;
cloning a raw WGPU buffer alone does not protect Burn's suballocation. A
non-contiguous tensor is copied **on device** before export. The render bridge
requires the same device/queue, f32 data and the explicit backend above.

## Pipeline changes

- **SOMA:** sparse correctives scatter across vertex coordinates instead of
  serially scanning each group. Each dispatch uses unique indices; overlapping
  groups remain ordered. Identity preparation is cached while posing, and bone
  ratios reuse the PCA rest shape and fitted bind. Identity changes still run
  the fitted-rig preparation and its necessary host reads.
- **GEM-X:** use backend attention dispatch, cache rotary tables, keep SAM
  keypoint projection/sampling on device, combine pose/camera reads, and omit
  the final SAM diagnostic head when only the body token is needed. Diagnostic
  `SamBody::forward` remains available for checkpoint comparisons. The viewer
  reuses the inferred prepared identity for subsequent rig edits.
- **ARDY:** retain checkpoint statistics and positional tables on device.
  Autoregressive generation retains at most the requested 160 history frames
  on device, avoiding the decoded-history re-upload. A host copy remains for
  the portable clip and window centering; decoded output is read once/window.

Native viewer jobs run on one persistent worker. Browser `generate` and
GEM-X inference yield real browser tasks between DDIM steps and vision blocks,
respectively. `sample_window_async` and `Vision::forward_async` expose the same
scheduling for other applications. These yields do not wait for GPU completion.
Synchronous variants remain available for throughput/validation callers.
Kernel compilation, identity fitting and necessary output reads can still add
latency; this is not a guarantee of stall-free execution on every adapter.

## Qualification

See [measured results and limitations](evidence/performance-2026-09-26/README.md).
`tool/scripts/profile_pipelines.py` runs the three synchronized checkpoint
validators sequentially and records commands, binary hashes and GPU state.
`tool/scripts/profile_webgpu.py` runs the corresponding browser exports and
records warm stage timing, memory and event-loop gaps. Preserve separate
baseline/candidate binaries, use the same fixtures and run GPU workloads
sequentially. Never infer inference throughput from enqueue time alone.

Hardware surface regression test:

```sh
cargo test -p bevy_burn_human --lib gpu_handoff_preserves -- --ignored --nocapture
```

This checks sliced/strided tensors, bounds, winding, an isolated vertex,
millimetre triangles, and Bevy bounds retention after a topology change.
Model fixtures/weights remain external to crate packages.
The performance update changes no CDN manifests or weights.
