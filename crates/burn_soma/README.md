# burn_soma

SOMA-X parametric identities, procedural skeletons, correctives and batched GPU skinning.

Part of [burn_human](https://github.com/mosure/burn_human). See the
[portable inference guide](https://github.com/mosure/burn_human/blob/main/docs/motion.md)
for model conversion, pinned checkpoints, native/WebGPU usage, CDN layout and
independent numerical/performance validation.

Model checkpoints are downloaded separately; no weights or Python runtime are
included in the Cargo package. The model crates share portable Burn operations
and the same bounded loader on native WGPU and browser WebGPU.
