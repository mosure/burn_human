# burn_human 🔥🧍

[![test](https://github.com/mosure/burn_human/workflows/test/badge.svg)](https://github.com/Mosure/burn_human/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_human.svg)](https://crates.io/crates/burn_human)


parametric 3d human ([anny](https://arxiv.org/abs/2511.03589)) model, [view the demo](https://mosure.github.io/burn_human).


![Alt text](./docs/example.gif)


## features

- [x] parametric 3dmm forward
- [x] bevy plugin
- [x] morphs, skeleton, skinning
- [ ] fitting


## usage

```rust
use bevy::prelude::*;
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, BurnHumanPlugin::default()))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    assets: Res<BurnHumanAssets>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.5, 4.0).looking_at(Vec3::Y, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 2.0).looking_at(Vec3::Y, Vec3::Y),
    ));

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
cargo run -p bevy_burn_human
```


## license
mit or apache-2.0 (anny stays under its original license)
