# burn_ardy

ARDY Core RP 20 FPS / Horizon40 inference in Rust/Burn 0.21, with native WGPU and browser WebGPU support.

The model emits Core27 motion and accepts text embeddings, history and world-space waypoints. Enable `transport` when disabling default features.

`Ardy::generate_batch` generates 1–8 independent actors together. Requests must
share frame count, history length, DDIM steps and guidance, while prompts, seeds
and waypoints may differ. Outputs retain request order. The same implementation
serves `generate` for one actor. GPU history is bounded to 160 frames per actor;
temporary conditioning covers at most 200 frames regardless of clip duration.
Progress reports completed frames per actor and permits cancellation between
windows. Text embeddings can be cached and reused; text encoding is separate.

The unfused WGPU backend pins matmul and reduction strategies to keep motion
codes stable as actor count changes. Qualification gates both continuous values
and discrete codes, then compares complete serial/batch clips and actor order.
Use `burn_human_inference::gpu::WgpuBackend` for this path; other backends and
adapters require their own numerical qualification. Seeds are not a promise of
bitwise equivalence across different devices or software versions.

```toml
burn_ardy = { version = "0.1.4", features = ["wgpu"] }
burn_human_inference = { version = "0.1.4", features = ["wgpu"] }
```

The model owns its immutable CDN catalog and loading policy:

```rust,ignore
let model = burn_ardy::Ardy::<burn_human_inference::gpu::WgpuBackend>::load_pretrained(
    &device,
    |done, total| { /* progress */ },
).await?;
```

`load_pretrained_from(root, device, progress)` accepts an alternate HTTP root or native model directory while preserving every compiled-in manifest SHA-256. The default root is `https://aberration.technology/model`. Loading is explicit and may require several gigabytes; no model is downloaded just by adding the dependency.

Physical Burnpack parts target 20 MiB and are authenticated before use. A logical object is bounded to 64 MiB; the loader does not concatenate the whole model in host/WASM memory. Shared transport uses a bounded 8 GiB native disk cache or browser CacheStorage, verifies cache hits and repairs corrupt entries. The cache can evict other models' older parts. GPU residency is separate from host/storage bounds.

Model weights are distributed separately under their upstream terms; consult the manifest-bound license metadata. The public CDN release has passed native and browser download/cache qualification; see the repository evidence for device and test coverage.

See [CDN preparation and validation](https://github.com/mosure/burn_human/blob/main/docs/cdn.md), [inference guide](https://github.com/mosure/burn_human/blob/main/docs/motion.md), and [numerical/performance evidence](https://github.com/mosure/burn_human/tree/main/docs/evidence/portable-2026-09-26).

Performance and GPU residency are described in the [performance guide](https://github.com/mosure/burn_human/blob/main/docs/performance.md). ARDY's sensitive WGPU kernels use fixed strategies on native and WebGPU. Burn graph fusion is not enabled by these crates. Browser loading uses WebCrypto for authenticated hashes and yields between tensor uploads; generation yields between transformer stages without extra device readbacks. The `denoise_async`, `encode_async`, `decode_async` and `sample_window_async` APIs expose cooperative scheduling; synchronous counterparts remain available.
