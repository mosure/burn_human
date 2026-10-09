use bevy::{ecs::schedule::common_conditions::run_once, prelude::*};
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

fn main() {
    bevy_burn_human::initialize_browser_runtime();
    App::new()
        .add_plugins((DefaultPlugins, BurnHumanPlugin::default()))
        .add_systems(Startup, setup_view)
        .add_systems(
            Update,
            setup_body.run_if(resource_exists::<BurnHumanAssets>.and_then(run_once)),
        )
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
