# Inference and resident-surface performance

Baseline: `cb5c564c8042135466b461534e0e84902cbc8af2`. Tests use unchanged,
authenticated v1 CDN bundles and the independent reference fixtures described
in [portable qualification](../portable-2026-09-26/README.md). The performance
harness reads the staged catalog to exclude network variability; viewer smoke
tests use the public origin. No checkpoint or CDN re-upload is required.

## Native, synchronized warm inference

Shared RTX PRO 6000 Blackwell, 96 GiB, NVIDIA 610.43.02. Both revisions use the
same native development profile. Two interleaved before/after rounds supply six
ARDY/GEM-X and ten SOMA samples per row. Values below pool those warm samples.
The desktop and unrelated processes remained active, and some CPU compilation
overlapped; these are workstation observations, not isolated-device guarantees.

| Workload | Before median | After median | Speedup |
| --- | ---: | ---: | ---: |
| ARDY, 40-frame window, 10 DDIM steps, CFG batch 3 | 390.6 ms | 213.9 ms | 1.83× |
| GEM-X, image through host mesh | 619.6 ms | 325.8 ms | 1.90× |
| SOMA, one pose with shape/correctives | 16.92 ms | 1.26 ms | 13.5× |
| SOMA, 32 poses with shape/correctives | 28.45 ms | 3.96 ms | 7.19× |

All timings synchronize by materializing output. ARDY excludes its decoder and
Llama text encoder; SOMA excludes identity preparation; GEM-X includes pose
reconstruction and vertex readback. Thus the SOMA throughput row is not a viewer
FPS measurement. See [raw samples](native-summary.json), [protocol](protocol.json),
and the individual validator/command reports in this directory.

Native SOMA's four reference cases retain a maximum vertex error below
1.5 micrometres. GEM-X remains below 0.4 mm against the full reference mesh.
SAM's resident feedback/final-head omission differs from its diagnostic path
by at most 1.431e-6 at the output token; see [SAM parity](sam-parity.json).
ARDY's fixed-input hooks retain their original tolerances, including the
cooperative sampler.

## Autoregressive histories

Fixed prompt, embedding, seed and waypoint fixtures exercise 0, 40 and 160
history frames over 80, 120 and 240 output frames. All outputs validate and
remain finite. Compare rotations geometrically: quaternion sign changes are
not motion errors. The 160-frame history run differs from the baseline by
2.61 mm RMS and 13.46 mm maximum FK joint distance over 12 seconds; waypoint
RMSE is 24.63 mm before and 24.62 mm after. The ordinary 40-frame history run
has 0.363 mm RMS joint difference and approximately 30.3 mm waypoint RMSE.
See [diagnostics](history-quality.json) and the request/provenance records.
These are numerical/trajectory examples, not a dataset-level semantic score.

The zero-history boundary case remains discontinuous across windows, as in the
baseline; it verifies the bounded API case and is not a continuity claim.
Cold first-use timings in the history logs include autotuning and are slower
than the baseline for some shapes. Warm throughput gains do not remove cold
compilation/tuning costs.

## Rejected configurations

Native Burn graph fusion failed SOMA `shape_pose` parity with approximately
29.8 mm vertex error. WebGPU autotuning produced invalid matmul shader warnings
and also failed SOMA `shape_pose`. Both failure logs are retained here. The
released defaults use native autotuning without graph fusion, and portable
WebGPU kernels without autotuning/fusion. No upstream patches are applied.

## Surface path

The hardware handoff test passed for contiguous, sliced and transposed tensors,
bounds, normals, isolated vertices and millimetre triangles. Native Vulkan
renders the resident surface and live Wave controls correctly. The GPU handoff
replaces the full 216,672-byte vertex readback and CPU mesh-normal rebuild with
an asynchronous 24-byte bounds readback and a GPU normals/packing pass. Host
procedural rig inputs and identity fitting still have their own small uploads
and necessary reads. This is not a claim that every pipeline stage has no
synchronization or allocation.

The final browser check loaded SOMA, then the public GEM-X suite, selected the
reference image, and rendered its inferred body through the resident buffer
path. It caught and fixed Bevy replacing evaluated bounds with the static
placeholder mesh's bounds after a topology change. The actor now uses
`NoAutoAabb` while retaining frustum culling; the hardware test exercises that
actual Bevy update sequence. See [the inferred body](browser-gemx-resident.png).

The [performance guide](../../performance.md) documents API use, allocator
leases, platform policy, and reproduction tools.

## Browser model measurements

Chrome headless shell 153.0.8010.12, NVIDIA hardware WebGPU, no software adapter.
Both revisions are release WASM builds. One before/after run supplies three warm
ARDY/GEM-X samples and five SOMA samples; this is a smaller screen than native.

| Workload | Before median | After median |
| --- | ---: | ---: |
| ARDY synchronized 40-frame window | 192.6 ms | 208.2 ms |
| GEM-X image through host mesh | 558.6 ms | 531.7 ms |
| SOMA one pose with shape/correctives | 11.8 ms | 3.7 ms |

No browser ARDY throughput improvement is established by these samples. Its
cooperative interactive scheduler is checked separately from the synchronous
throughput loop. GEM-X improves modestly and SOMA improves substantially.
[Browser reports](browser-summary.json) retain all samples, adapter identity,
module hashes and event-loop observations.

Peak WASM linear memory was 239→279 MiB for the ARDY validator and
943→1,181 MiB for GEM-X; SOMA stayed at 131 MiB. These include load, cold kernel
compilation, parity hooks and warm runs, and exclude GPU memory/CacheStorage.
The update is **not** a demonstrated reduction in total browser memory.
Model loading/compilation still produces long tasks (roughly 0.2–0.3 seconds in
these runs); whole-run RAF percentiles are not warm viewer latency or a claim
of stall-free startup.

## Viewer responsiveness

The rebuilt native Vulkan and browser WebGPU viewers both render the resident
SOMA surface, including live Wave/rotation edits. During eight warmed browser
rotation-slider drags at 1500×1050, 487 RAF intervals had a median of 16.7 ms,
p95 of 16.8 ms and maximum of 33.4 ms, with no observed >50 ms main-thread long
task. See [frame observations](viewer-frames.json) and [rendered live edits](browser-live-edits.png).
RAF intervals describe browser scheduling opportunities; they do not measure
input-to-pixel latency or prove every intermediate requested pose was presented.
The viewer coalesces edits through its existing single-flight submission policy.

The shared native backend configuration was also checked against all ten Llama
prompt/mode fixtures: minimum embedding cosine remained above 0.9999984. See
[text parity](text-parity.json). The Llama crate itself has no source/version
change in this release.
