# burn_human_inference

Bounded, authenticated Burnpack loading and precision utilities for native and WebAssembly inference.

Part of [burn_human](https://github.com/mosure/burn_human). See the
[portable inference guide](https://github.com/mosure/burn_human/blob/main/docs/motion.md)
for model conversion, pinned checkpoints, native/WebGPU usage, CDN layout and
independent numerical/performance validation.

Model checkpoints are downloaded separately; no weights or Python runtime are
included in the Cargo package. The model crates share portable Burn operations
and the same bounded loader on native WGPU and browser WebGPU.

The `wgpu` feature also exposes `gpu::export` for same-device tensor buffers.
Its allocation lease must remain alive until external GPU commands finish;
see the [GPU handoff and performance guide](https://github.com/mosure/burn_human/blob/main/docs/performance.md).
`cooperative::yield_to_browser` yields a browser task without a device wait.
Native WGPU enables autotuning; WebGPU retains the validated portable kernels.
