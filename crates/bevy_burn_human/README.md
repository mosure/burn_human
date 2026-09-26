# bevy_burn_human

Bevy rendering, rig controls and playback for Anny and SOMA humans, with local
Burn inference for Llama text encoding, ARDY motion and GEM-X image poses.
The models share Bevy's WGPU device and run on native platforms and browser
WebGPU. Weights load on demand from verified, bounded Burnpack parts.

`BurnHumanPlugin` provides the parametric Anny renderer.
`motion::HumanMotionPlugin` adds the motion studio, world-space waypoints,
SOMA controls and image-pose UI; it requires Bevy Egui and the render plugins.
The repository's demo shows the complete plugin setup and required Anny assets.

See the [model and viewer guide](https://github.com/mosure/burn_human/blob/main/docs/motion.md)
for conversion, loading, native/WASM use and numerical/performance evidence.
Model weights are separate and retain their upstream terms; see
[third-party licenses](https://github.com/mosure/burn_human/blob/main/THIRD_PARTY.md).
