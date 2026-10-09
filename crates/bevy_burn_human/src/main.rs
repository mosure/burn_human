use bevy::app::AppExit;
use bevy::camera_controller::pan_orbit_camera::prelude::{
    DefaultPanOrbitCameraPlugins, PanOrbitCamera,
};
use bevy::input::{ButtonInput, keyboard::KeyCode};
use bevy::prelude::*;
use bevy::prelude::{MessageReader, MessageWriter};
use bevy_burn_human::motion::{HumanMotionPlugin, MotionActor, MotionPlayback, MotionUi};
use bevy_burn_human::studio_ui::{ControlEvent, Forms, Row, StudioInput, Value};
use bevy_burn_human::{
    BurnHumanAssets, BurnHumanInput, BurnHumanMeshMode, BurnHumanPlugin, BurnHumanRenderMode,
};
use noise::{NoiseFn, OpenSimplex};

#[derive(Component)]
struct HumanTag;

#[derive(Resource, Clone)]
struct DemoState {
    phenotype_labels: Vec<String>,
    phenotype_values: Vec<f64>,
    phenotype_noise_baseline: Vec<f64>,
    use_reference_case: bool,
    selected_case: usize,
    selected_bone: usize,
    bone_euler_deg: Vec<[f32; 3]>,
    bone_noise_baseline: Vec<[f32; 3]>,
    noise_enabled: bool,
    noise_amp: f32,
    phenotype_noise_amp: f32,
    upper_leg_noise_amp: f32,
    lower_leg_noise_amp: f32,
    upper_arm_noise_amp: f32,
    lower_arm_noise_amp: f32,
    wrist_noise_amp: f32,
    hand_noise_amp: f32,
    spine_noise_amp: f32,
    other_pose_noise_amp: f32,
    time_scale: f32,
}

#[derive(Resource)]
struct NoiseRig {
    noise: OpenSimplex,
}

#[derive(Resource, Default)]
struct SceneSpawned(bool);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start_human_studio() {
    main();
}

pub fn main() {
    burn_human_inference::initialize_browser_runtime();
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // Let Bevy's asset server fetch the files at runtime (native + wasm).
    run_app(BurnHumanPlugin::default());
}

// Support `cargo run -p bevy_burn_human` and direct execution from the repo.
// Packaged applications may keep assets next to the executable or set BEVY_ASSET_ROOT.
fn studio_assets() -> bevy::asset::AssetPlugin {
    #[allow(unused_mut)]
    let mut plugin = bevy::asset::AssetPlugin {
        meta_check: bevy::asset::AssetMetaCheck::Never,
        ..default()
    };
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("BEVY_ASSET_ROOT").is_none()
        && std::path::Path::new("assets/model/fullbody_default.meta.json").is_file()
        && let Ok(path) = std::fs::canonicalize("assets")
    {
        plugin.file_path = path.to_string_lossy().into_owned();
    }
    plugin
}

fn run_app(burn_plugin: BurnHumanPlugin) {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.08)))
        .insert_resource(SceneSpawned::default())
        .add_plugins(DefaultPlugins.set(studio_assets()).set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_burn_human".to_string(),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(DefaultPanOrbitCameraPlugins)
        .add_plugins(burn_plugin)
        .add_plugins(HumanMotionPlugin)
        .add_systems(Startup, setup_stage)
        .add_systems(PreUpdate, handle_close_requests)
        .add_systems(
            PreUpdate,
            (
                apply_random_pose_on_key
                    .run_if(resource_exists::<DemoState>)
                    .run_if(manual_pose),
                drive_noise
                    .run_if(manual_pose)
                    .run_if(|state: Option<Res<DemoState>>| {
                        state.is_some_and(|state| state.noise_enabled || state.is_changed())
                    })
                    .run_if(resource_exists::<NoiseRig>)
                    .run_if(resource_exists::<DemoState>),
            )
                .run_if(resource_exists::<BurnHumanAssets>),
        )
        .add_systems(
            Update,
            setup_scene_once.run_if(resource_exists::<BurnHumanAssets>),
        )
        .add_systems(
            Update,
            ui_controls
                .run_if(resource_exists::<BurnHumanAssets>)
                .run_if(resource_exists::<DemoState>),
        )
        .add_systems(
            Update,
            setup_noise.run_if(resource_exists::<BurnHumanAssets>),
        )
        .run();
}

fn setup_scene_once(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    assets: Res<BurnHumanAssets>,
    mut spawned: ResMut<SceneSpawned>,
) {
    if spawned.0 {
        return;
    }
    let phenotype_labels = assets.body.metadata().metadata.phenotype_labels.clone();
    let phenotype_values = vec![0.5; phenotype_labels.len()];
    let selected_case = assets
        .body
        .metadata()
        .cases
        .iter()
        .position(|c| c.pose_parameters.shape[0] == 1)
        .unwrap_or(0usize);
    let bone_count = assets.body.metadata().metadata.bone_labels.len();

    commands.insert_resource(DemoState {
        phenotype_labels,
        phenotype_values: phenotype_values.clone(),
        phenotype_noise_baseline: phenotype_values.clone(),
        use_reference_case: false,
        selected_case,
        selected_bone: 0,
        bone_euler_deg: vec![[0.0; 3]; bone_count],
        bone_noise_baseline: vec![[0.0; 3]; bone_count],
        noise_enabled: false,
        noise_amp: 0.35,
        phenotype_noise_amp: 1.0,
        upper_leg_noise_amp: 12.0,
        lower_leg_noise_amp: 10.0,
        upper_arm_noise_amp: 10.0,
        lower_arm_noise_amp: 8.0,
        wrist_noise_amp: 6.0,
        hand_noise_amp: 6.0,
        spine_noise_amp: 6.0,
        other_pose_noise_amp: 3.0,
        time_scale: 1.0,
    });
    commands.insert_resource(NoiseRig {
        noise: OpenSimplex::new(42),
    });

    commands.spawn((
        BurnHumanInput {
            case_name: assets
                .body
                .metadata()
                .metadata
                .case_names
                .get(selected_case)
                .cloned(),
            phenotype_inputs: Some(phenotype_values),
            ..Default::default()
        },
        BurnHumanRenderMode(BurnHumanMeshMode::SkinnedMesh),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.7, 0.68),
            metallic: 0.0,
            reflectance: 0.5,
            perceptual_roughness: 0.55,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_translation(Vec3::Y)
            .with_scale(Vec3::splat(1.15)),
        Visibility::default(),
        Name::new("burn_human"),
        HumanTag,
        MotionActor,
    ));

    spawned.0 = true;
}

fn setup_stage(mut commands: Commands) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.85, 0.85, 0.9),
        brightness: 1.05,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 3_000.0,
            shadow_maps_enabled: false,
            color: Color::srgb(0.97, 0.97, 1.0),
            ..default()
        },
        Transform::from_xyz(5.0, 7.0, 4.5).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 1_800.0,
            shadow_maps_enabled: false,
            color: Color::srgb(0.9, 0.94, 1.0),
            ..default()
        },
        Transform::from_xyz(-4.0, 5.5, -4.5).looking_at(Vec3::new(0.0, 1.1, 0.0), Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 620.0,
            shadow_maps_enabled: false,
            range: 18.0,
            color: Color::srgb(0.94, 0.95, 0.99),
            ..default()
        },
        Transform::from_xyz(-3.0, 3.2, 2.4),
    ));
    commands.spawn((
        PointLight {
            intensity: 520.0,
            range: 16.0,
            color: Color::srgb(0.7, 0.75, 0.9),
            ..default()
        },
        Transform::from_xyz(3.2, 2.4, -2.8),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.8, 1.6, 4.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        PanOrbitCamera {
            last_anchor_depth: -4.5,
            ..default()
        },
    ));
}

fn setup_noise(mut commands: Commands, has_noise: Option<Res<NoiseRig>>) {
    if has_noise.is_none() {
        commands.insert_resource(NoiseRig {
            noise: OpenSimplex::new(42),
        });
    }
}

fn ui_controls(
    mut commands: Commands,
    mut events: MessageReader<ControlEvent>,
    assets: Res<BurnHumanAssets>,
    mut state: ResMut<DemoState>,
    motion: Res<MotionUi>,
    mut forms: ResMut<Forms>,
    mut query: Query<(Entity, Option<&BurnHumanRenderMode>, &mut Transform), With<HumanTag>>,
) {
    if !motion.is_manual_tab() {
        forms.0.insert("anny", vec![]);
        return;
    }
    let Ok((entity, render_mode, mut transform)) = query.single_mut() else {
        return;
    };
    let mut mode = render_mode
        .as_ref()
        .map(|m| m.0)
        .unwrap_or(BurnHumanMeshMode::SkinnedMesh);
    for event in events.read() {
        if !event.id.starts_with("anny.") {
            continue;
        }
        let id = &event.id[5..];
        match (&event.value, id) {
            (Value::Choice(value), "mode") => {
                mode = if *value == 0 {
                    BurnHumanMeshMode::SkinnedMesh
                } else {
                    BurnHumanMeshMode::BakedMesh
                }
            }
            (Value::Choice(value), "bone") => state.selected_bone = *value,
            (Value::Choice(value), "case") => state.selected_case = *value,
            (Value::Toggle(value), "reference") => state.use_reference_case = *value,
            (Value::Toggle(value), "noise") => {
                if *value && !state.noise_enabled {
                    state.phenotype_noise_baseline = state.phenotype_values.clone();
                    state.bone_noise_baseline = state.bone_euler_deg.clone();
                }
                state.noise_enabled = *value;
            }
            (Value::Action, "reset_shape") => state.phenotype_values.fill(0.5),
            (Value::Action, "reset_bone") => {
                let i = state.selected_bone;
                if let Some(bone) = state.bone_euler_deg.get_mut(i) {
                    *bone = [0.; 3];
                }
            }
            (Value::Number(value), _) if value.is_finite() => match id {
                "noise_amp" => state.noise_amp = *value,
                "phenotype_noise_amp" => state.phenotype_noise_amp = *value,
                "upper_leg_noise_amp" => state.upper_leg_noise_amp = *value,
                "lower_leg_noise_amp" => state.lower_leg_noise_amp = *value,
                "upper_arm_noise_amp" => state.upper_arm_noise_amp = *value,
                "lower_arm_noise_amp" => state.lower_arm_noise_amp = *value,
                "wrist_noise_amp" => state.wrist_noise_amp = *value,
                "hand_noise_amp" => state.hand_noise_amp = *value,
                "spine_noise_amp" => state.spine_noise_amp = *value,
                "other_pose_noise_amp" => state.other_pose_noise_amp = *value,
                "time_scale" => state.time_scale = *value,
                _ if id.starts_with("shape_") => {
                    if let Ok(i) = id[6..].parse::<usize>()
                        && let Some(slot) = state.phenotype_values.get_mut(i)
                    {
                        *slot = *value as f64;
                    }
                }
                _ if id.starts_with("rotation_") => {
                    if let Ok(i) = id[9..].parse::<usize>()
                        && i < 3
                    {
                        let selected = state.selected_bone;
                        if let Some(bone) = state.bone_euler_deg.get_mut(selected) {
                            bone[i] = *value;
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    if render_mode
        .as_ref()
        .is_none_or(|previous| previous.0 != mode)
    {
        commands.entity(entity).insert(BurnHumanRenderMode(mode));
    }
    let posed = Transform {
        translation: Vec3::Y,
        scale: Vec3::splat(1.15),
        ..*transform
    };
    transform.set_if_neq(posed);

    let mut rows = vec![
        Row::Label("Anny body & pose · R: random pose".into()),
        Row::select(
            "anny.mode",
            "Render mode",
            vec!["Skinned (GPU)".into(), "Baked (CPU)".into()],
            usize::from(mode == BurnHumanMeshMode::BakedMesh),
        ),
        Row::toggle(
            "anny.reference",
            "Use bundled reference pose",
            state.use_reference_case,
        ),
    ];
    if state.use_reference_case {
        rows.push(Row::select(
            "anny.case",
            "Reference pose",
            assets.body.metadata().metadata.case_names.clone(),
            state.selected_case,
        ));
    } else {
        for (i, value) in state.phenotype_values.iter().enumerate() {
            rows.push(Row::slider(
                &format!("anny.shape_{i}"),
                &state.phenotype_labels[i],
                *value as f32,
                0.,
                1.,
                0.01,
            ));
        }
        rows.push(Row::button("anny.reset_shape", "Reset body shape", true));
    }
    rows.push(Row::toggle(
        "anny.noise",
        "Procedural motion",
        state.noise_enabled,
    ));
    for (id, title, value, min, max) in [
        ("noise_amp", "Global noise", state.noise_amp, 0., 2.),
        (
            "phenotype_noise_amp",
            "Body shape noise",
            state.phenotype_noise_amp,
            0.,
            4.,
        ),
        (
            "upper_leg_noise_amp",
            "Upper leg noise (degrees)",
            state.upper_leg_noise_amp,
            0.,
            50.,
        ),
        (
            "lower_leg_noise_amp",
            "Lower leg noise (degrees)",
            state.lower_leg_noise_amp,
            0.,
            50.,
        ),
        (
            "upper_arm_noise_amp",
            "Upper arm noise (degrees)",
            state.upper_arm_noise_amp,
            0.,
            50.,
        ),
        (
            "lower_arm_noise_amp",
            "Lower arm noise (degrees)",
            state.lower_arm_noise_amp,
            0.,
            50.,
        ),
        (
            "wrist_noise_amp",
            "Wrist noise (degrees)",
            state.wrist_noise_amp,
            0.,
            50.,
        ),
        (
            "hand_noise_amp",
            "Hand noise (degrees)",
            state.hand_noise_amp,
            0.,
            50.,
        ),
        (
            "spine_noise_amp",
            "Spine noise (degrees)",
            state.spine_noise_amp,
            0.,
            50.,
        ),
        (
            "other_pose_noise_amp",
            "Other pose noise (degrees)",
            state.other_pose_noise_amp,
            0.,
            50.,
        ),
        (
            "time_scale",
            "Procedural time scale",
            state.time_scale,
            0.25,
            3.,
        ),
    ] {
        rows.push(Row::slider(
            &format!("anny.{id}"),
            title,
            value,
            min,
            max,
            0.05,
        ));
    }
    rows.push(Row::select(
        "anny.bone",
        "Bone orientation",
        assets.body.metadata().metadata.bone_labels.clone(),
        state.selected_bone,
    ));
    if let Some(bone) = state.bone_euler_deg.get(state.selected_bone) {
        for (i, title) in [
            "Euler X (degrees)",
            "Euler Y (degrees)",
            "Euler Z (degrees)",
        ]
        .iter()
        .enumerate()
        {
            rows.push(Row::slider(
                &format!("anny.rotation_{i}"),
                title,
                bone[i],
                -90.,
                90.,
                1.,
            ));
        }
    }
    rows.push(Row::button("anny.reset_bone", "Reset bone", true));
    forms.0.insert("anny", rows);
}

fn manual_pose(playback: Res<MotionPlayback>, motion: Res<MotionUi>) -> bool {
    !playback.active && (motion.is_motion_tab() || motion.is_manual_tab())
}

fn apply_random_pose_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    ui_input: Res<StudioInput>,
    assets: Res<BurnHumanAssets>,
    mut state: ResMut<DemoState>,
    mut query: Query<&mut BurnHumanInput, With<HumanTag>>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    if ui_input.keyboard {
        return;
    }
    let mut rng = fastrand::Rng::new();
    let mut input = if let Ok(i) = query.single_mut() {
        i
    } else {
        return;
    };

    // Shuffle phenotype sliders around mid-range.
    let phen_len = assets.body.metadata().metadata.phenotype_labels.len();
    state.phenotype_values.clear();
    for _ in 0..phen_len {
        let jitter = rng.f64() * 0.5 - 0.25;
        state.phenotype_values.push((0.5 + jitter).clamp(0.0, 1.0));
    }

    // Shuffle bone Euler sliders (skip root bone to avoid unrealistic drift).
    for (idx, bone) in state.bone_euler_deg.iter_mut().enumerate() {
        if idx == 0 {
            *bone = [0.0; 3];
            continue;
        }
        bone[0] = rng.f32() * 40.0 - 20.0;
        bone[1] = rng.f32() * 50.0 - 25.0;
        bone[2] = rng.f32() * 40.0 - 20.0;
    }

    // Drive directly from sliders (no reference case).
    state.use_reference_case = false;
    state.phenotype_noise_baseline = state.phenotype_values.clone();
    state.bone_noise_baseline = state.bone_euler_deg.clone();
    input.case_name = None;
    input.phenotype_inputs = Some(state.phenotype_values.clone());
    input.blendshape_weights = None;
    input.blendshape_delta = None;
    input.pose_parameters = None;
    input.pose_parameters_delta = None;
    apply_pose_from_state(&mut input, &assets, &state);
}

fn handle_close_requests(
    mut reader: MessageReader<bevy::window::WindowCloseRequested>,
    mut exit: MessageWriter<AppExit>,
) {
    if reader.read().next().is_some() {
        exit.write(AppExit::Success);
    }
}

fn apply_pose_from_state(input: &mut BurnHumanInput, assets: &BurnHumanAssets, state: &DemoState) {
    let bones = assets.body.metadata().metadata.bone_labels.len();
    let (batch, mut pose) = if let Some(name) = input.case_name.as_ref() {
        if let Some(case) = assets
            .body
            .metadata()
            .cases
            .iter()
            .find(|c| &c.name == name)
        {
            (
                case.pose_parameters.shape[0],
                case.pose_parameters.data.clone(),
            )
        } else {
            (1, Vec::new())
        }
    } else {
        (1, Vec::new())
    };

    if pose.is_empty() {
        pose = vec![0.0f64; bones * 16];
        for bone in 0..bones {
            let base = bone * 16;
            pose[base] = 1.0;
            pose[base + 5] = 1.0;
            pose[base + 10] = 1.0;
            pose[base + 15] = 1.0;
        }
    }

    for b in 0..batch.max(1) {
        for (bone, rot_deg) in state.bone_euler_deg.iter().enumerate().take(bones) {
            let base = (b * bones + bone) * 16;
            if base + 15 >= pose.len() {
                break;
            }
            let base_rot = [
                [pose[base], pose[base + 1], pose[base + 2]],
                [pose[base + 4], pose[base + 5], pose[base + 6]],
                [pose[base + 8], pose[base + 9], pose[base + 10]],
            ];
            let delta = euler_deg_to_mat3(rot_deg[0], rot_deg[1], rot_deg[2]);
            let rot = mat3_mul(delta, base_rot);
            pose[base] = rot[0][0];
            pose[base + 1] = rot[0][1];
            pose[base + 2] = rot[0][2];
            pose[base + 4] = rot[1][0];
            pose[base + 5] = rot[1][1];
            pose[base + 6] = rot[1][2];
            pose[base + 8] = rot[2][0];
            pose[base + 9] = rot[2][1];
            pose[base + 10] = rot[2][2];
            pose[base + 15] = 1.0;
        }
    }
    input.pose_parameters = Some(pose);
    input.pose_parameters_delta = None;
}

fn euler_deg_to_mat3(rx_deg: f32, ry_deg: f32, rz_deg: f32) -> [[f64; 3]; 3] {
    let (rx, ry, rz) = (
        rx_deg.to_radians() as f64,
        ry_deg.to_radians() as f64,
        rz_deg.to_radians() as f64,
    );
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    // Z * Y * X rotation order
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

struct PoseNoiseScales {
    upper_leg: f32,
    lower_leg: f32,
    upper_arm: f32,
    lower_arm: f32,
    wrist: f32,
    hand: f32,
    spine: f32,
    other: f32,
}

impl From<&DemoState> for PoseNoiseScales {
    fn from(state: &DemoState) -> Self {
        Self {
            upper_leg: state.upper_leg_noise_amp,
            lower_leg: state.lower_leg_noise_amp,
            upper_arm: state.upper_arm_noise_amp,
            lower_arm: state.lower_arm_noise_amp,
            wrist: state.wrist_noise_amp,
            hand: state.hand_noise_amp,
            spine: state.spine_noise_amp,
            other: state.other_pose_noise_amp,
        }
    }
}

fn pose_noise_scale_for_bone(name: &str, scales: &PoseNoiseScales) -> f32 {
    let lower = name.to_ascii_lowercase();
    if lower.contains("upperleg") || lower.contains("upper_leg") || lower.contains("thigh") {
        scales.upper_leg
    } else if lower.contains("lowerleg")
        || lower.contains("lower_leg")
        || lower.contains("calf")
        || lower.contains("knee")
        || lower.contains("foot")
        || lower.contains("toe")
    {
        scales.lower_leg
    } else if lower.contains("upper_arm")
        || lower.contains("shoulder")
        || lower.contains("clavicle")
        || lower.contains("upperarm")
    {
        scales.upper_arm
    } else if lower.contains("lower_arm")
        || lower.contains("lowerarm")
        || lower.contains("elbow")
        || lower.contains("forearm")
    {
        scales.lower_arm
    } else if lower.contains("wrist") {
        scales.wrist
    } else if lower.contains("hand") || lower.contains("finger") || lower.contains("metacarpal") {
        scales.hand
    } else if lower.contains("spine") || lower.contains("neck") || lower.contains("chest") {
        scales.spine
    } else {
        scales.other
    }
}

fn pick_single_sample_case(state: &DemoState, assets: &BurnHumanAssets) -> Option<(usize, String)> {
    let names = &assets.body.metadata().metadata.case_names;
    names
        .get(state.selected_case)
        .cloned()
        .map(|n| (state.selected_case, n))
}

fn drive_noise(
    time: Res<Time>,
    assets: Res<BurnHumanAssets>,
    noise: Res<NoiseRig>,
    mut state: ResMut<DemoState>,
    mut query: Query<&mut BurnHumanInput, With<HumanTag>>,
) {
    let mut input = if let Ok(i) = query.single_mut() {
        i
    } else {
        return;
    };

    if !state.noise_enabled {
        input.case_name = if state.use_reference_case {
            pick_single_sample_case(&state, &assets).map(|(_, n)| n)
        } else {
            None
        };
        input.phenotype_inputs = if input.case_name.is_none() {
            Some(state.phenotype_values.clone())
        } else {
            None
        };
        apply_pose_from_state(&mut input, &assets, &state);
        return;
    }

    state.use_reference_case = false;
    let t = time.elapsed_secs_f64() * state.time_scale as f64;

    let phenotype_len = assets.body.metadata().metadata.phenotype_labels.len();
    state.phenotype_noise_baseline.resize(phenotype_len, 0.5);
    state.phenotype_values.resize(phenotype_len, 0.5);
    let noise_amp = state.noise_amp as f64;
    let phen_amp = state.phenotype_noise_amp as f64;
    let phen_noise_active = noise_amp > 0.0 && phen_amp > 0.0;
    if phen_noise_active {
        let phen_labels = assets.body.metadata().metadata.phenotype_labels.clone();
        let phen_baseline = state.phenotype_noise_baseline.clone();
        for (i, value) in state.phenotype_values.iter_mut().enumerate() {
            let label = phen_labels.get(i).map(|s| s.as_str()).unwrap_or("");
            let (freq, amp_hint) = match label {
                "gender" => (0.5, 0.25),
                "muscle" | "weight" => (0.7, 0.22),
                "height" => (0.6, 0.18),
                "proportions" => (0.8, 0.22),
                _ => (0.6, 0.2),
            };
            let n = noise.noise.get([t * freq, (200.0 + i as f64) * 0.11]);
            let amp = amp_hint * noise_amp * phen_amp;
            let base = *phen_baseline.get(i).unwrap_or(&0.5);
            *value = (base + n * amp).clamp(0.0, 1.0);
        }
    } else {
        state.phenotype_noise_baseline = state.phenotype_values.clone();
    }

    let bone_count = state.bone_euler_deg.len();
    state.bone_noise_baseline.resize(bone_count, [0.0; 3]);
    let bone_labels = assets.body.metadata().metadata.bone_labels.clone();
    let pose_scales = PoseNoiseScales::from(&*state);
    let noise_amp_f32 = state.noise_amp;
    for idx in 0..bone_count {
        let baseline_val = state
            .bone_noise_baseline
            .get(idx)
            .copied()
            .unwrap_or([0.0; 3]);
        if idx == 0 {
            state.bone_euler_deg[idx] = [0.0; 3];
            state.bone_noise_baseline[idx] = [0.0; 3];
            continue;
        }
        let group_amp = bone_labels
            .get(idx)
            .map(|name| pose_noise_scale_for_bone(name, &pose_scales))
            .unwrap_or(pose_scales.other);
        let amp = group_amp * noise_amp_f32;
        if amp <= f32::EPSILON {
            let current = state.bone_euler_deg.get(idx).copied().unwrap_or([0.0; 3]);
            state.bone_noise_baseline[idx] = current;
            continue;
        }
        let nx = noise.noise.get([t * 0.35, idx as f64 * 0.17]);
        let ny = noise.noise.get([t * 0.45 + 97.0, idx as f64 * 0.23]);
        let nz = noise.noise.get([t * 0.55 + 197.0, idx as f64 * 0.13]);
        let new_bone = [
            (baseline_val[0] + (nx as f32) * amp).clamp(-90.0, 90.0),
            (baseline_val[1] + (ny as f32) * amp).clamp(-90.0, 90.0),
            (baseline_val[2] + (nz as f32) * amp).clamp(-90.0, 90.0),
        ];
        state.bone_euler_deg[idx] = new_bone;
    }

    input.case_name = None;
    input.phenotype_inputs = Some(state.phenotype_values.clone());
    input.blendshape_weights = None;
    input.blendshape_delta = None;
    apply_pose_from_state(&mut input, &assets, &state);
}
