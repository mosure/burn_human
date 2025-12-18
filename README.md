# burn_human 🔥🧍

[![test](https://github.com/mosure/burn_human/workflows/test/badge.svg)](https://github.com/Mosure/burn_human/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/burn_human.svg)](https://crates.io/crates/burn_human)


parametric 3d human ([anny](https://arxiv.org/abs/2511.03589)) model, [view the demo](https://mosure.github.io/burn_human).


![Alt text](./docs/example.gif)


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
python tool/scripts/export_reference.py --output tests/reference/fullbody_default.safetensors --seed 1234
```

exporter runs against the vendored `anny/` codebase and writes `tests/reference/fullbody_default.safetensors` plus `tests/reference/fullbody_default.meta.json` consumed by the library/tests/demo. The output is deterministic (seeded) and cached assets live under `.cache/anny/`.


## run the demo
```bash
cargo run -p bevy_burn_human
```


## license
mit or apache-2.0 (anny stays under its original license)
