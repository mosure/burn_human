# burn_human 🔥🧍

[![test](https://github.com/mosure/burn_human/workflows/test/badge.svg)](https://github.com/Mosure/burn_human/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_human.svg)](https://crates.io/crates/burn_human)


parametric 3d human ([anny](https://arxiv.org/abs/2511.03589)) model, [view the demo](https://mosure.github.io/burn_human).


![Alt text](./docs/example.gif)


## features

- [x] parametric 3dmm forward
- [x] bevy plugin
- [x] morphs, skeleton, skinning
- [x] ARDY motion generation and local Llama text conditioning
- [x] SOMA identity, rig, correctives and GPU skinning
- [x] GEM-X image-to-pose inference
- [x] native / WebGPU studio with text, image and trajectory controls
- [x] single-image fitting (GEM-X)


## compatibility

This release uses **Bevy 0.20, Burn 0.22 and WGPU 30**, from crates.io without
dependency patches. Use Rust 1.97.1 or newer for the Bevy integration (the model
crates require Rust 1.95). The repository validates with nightly Rust.

Burn tensors and model types no longer take a backend type parameter. Select
execution with `burn::tensor::Device::wgpu(...)` or `Device::flex()`; native
CPU inference uses Burn Flex. Browser initialization is asynchronous through
`Device::wgpu_options().init_async().await`. Model loading retains the existing
authenticated CDN catalog, including its Burn 0.21 block-Q4 packs.

The studio uses Bevy’s Feathers UI and built-in pan-orbit camera controller.

The [upgrade qualification](docs/evidence/upgrade-2026-10-08/README.md) records
native/WebGPU numerical checks, batch consistency, CDN cache recovery and
measured cold/warm behavior.

## usage

```rust
use bevy::{prelude::*, ecs::schedule::common_conditions::run_once};
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

fn main() {
    bevy_burn_human::initialize_browser_runtime();
    App::new()
        .add_plugins((DefaultPlugins, BurnHumanPlugin::default()))
        .add_systems(Startup, setup_view)
        .add_systems(Update, setup_body.run_if(resource_exists::<BurnHumanAssets>.and_then(run_once)))
        .run();
}

fn setup_view(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.5, 4.0).looking_at(Vec3::Y, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 2.0).looking_at(Vec3::Y, Vec3::Y),
    ));

}

fn setup_body(
    mut commands: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    assets: Res<BurnHumanAssets>,
) {
    let n = assets.body.metadata().metadata.phenotype_labels.len();
    commands.spawn((
        BurnHumanInput {
            phenotype_inputs: Some(vec![0.5; n]),
            ..default()
        },
        MeshMaterial3d(mats.add(StandardMaterial::default())),
    ));
}
```


## generate reference data

```bash
python tool/scripts/export_reference.py --output assets/model/fullbody_default.safetensors --seed 1234
```

The exporter now writes a lighter neutral bundle (`fullbody_default.safetensors` + `fullbody_default.meta.json`) with the heavy blendshape tensors quantized to float16 and unused reference cases removed. Output is deterministic (seeded) and cached assets live under `.cache/anny/`.

Bevy loads the model through the asset server (native and wasm). Keep the files under `assets/model/` so the asset path `model/fullbody_default.meta.json` resolves at runtime.


## run the demo
```bash
cargo run --locked -p bevy_burn_human
```


## license
mit or apache-2.0 (anny stays under its original license). Model crates and converted weights retain their [upstream licenses](THIRD_PARTY.md).

## Motion generation

Portable Llama text encoding, ARDY motion generation, Anny retargeting, SOMA body evaluation, and GEM-X image-to-pose inference run locally in Rust/Burn on native WGPU and browser WebGPU. See [Motion inference and rigging](docs/motion.md) for model packing, world-space waypoint controls, SOMA controls, image input, and numerical/performance evidence.

The published model crates are `burn_ardy`, `burn_llama`, `burn_soma` and `burn_gemx`. Each owns its pinned aberration CDN catalog and cached loading API. See [CDN bundles and loading](docs/cdn.md); the older `burn_ardy_text` and `burn_gem` 0.1.0 packages remain available under their original names.

The [studio guide](docs/studio.md) covers text prompts, SOMA controls, world-space
waypoints, PanOrbit navigation and GEM-X image input on native and WebGPU.
