# burn_gemx

GEM-X inference in Rust/Burn 0.21, with native WGPU and browser WebGPU support.

Fast single-person, single-frame image pose inference. The crate owns DINOv3/ViTPose, SAM body features and GEM regression, and composes `burn_mhr` and `burn_soma` for reconstruction. It loads all seven required artifact bundles.

```toml
burn_gemx = { version = "0.1.1", features = ["wgpu"] }
```

The model owns its immutable CDN catalog and loading policy:

```rust,ignore
let model = burn_gemx::pipeline::Pipeline::<burn::backend::Wgpu>::load_pretrained(
    &device,
    |stage, done, total| { /* progress */ },
).await?;
```

`load_pretrained_from(root, device, progress)` accepts an alternate HTTP root or native model directory while preserving every compiled-in manifest SHA-256. The default root is `https://aberration.technology/model`. Loading is explicit and may require several gigabytes; no model is downloaded just by adding the dependency.

Physical Burnpack parts target 20 MiB and are authenticated before use. A logical object is bounded to 64 MiB; the loader does not concatenate the whole model in host/WASM memory. Shared transport uses a bounded 8 GiB native disk cache or browser CacheStorage, verifies cache hits and repairs corrupt entries. The cache can evict other models' older parts. GPU residency is separate from host/storage bounds.

Model weights are distributed separately under their upstream terms; consult the manifest-bound license metadata. The CDN release is prepared for upload; deployment and public download qualification are tracked separately in the repository.

See [CDN preparation and validation](https://github.com/mosure/burn_human/blob/main/docs/cdn.md), [inference guide](https://github.com/mosure/burn_human/blob/main/docs/motion.md), and [numerical/performance evidence](https://github.com/mosure/burn_human/tree/main/docs/evidence/portable-2026-09-26).

This crate replaces `burn_gem` 0.1.0.
