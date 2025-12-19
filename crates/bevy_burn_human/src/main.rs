use bevy::app::AppExit;
use bevy::input::{ButtonInput, keyboard::KeyCode};
use bevy::prelude::*;
use bevy::prelude::{MessageReader, MessageWriter};
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
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


pub fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // Let Bevy's asset server fetch the files at runtime (native + wasm).
    run_app(BurnHumanPlugin::default());
}

fn run_app(burn_plugin: BurnHumanPlugin) {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.08)))
        .insert_resource(SceneSpawned::default())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_burn_human".to_string(),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(burn_plugin)
        .add_systems(
            PreUpdate,
            (
                handle_close_requests,
                apply_random_pose_on_key.run_if(resource_exists::<DemoState>),
                gate_pan_orbit_during_egui,
                drive_noise
                    .run_if(resource_exists::<NoiseRig>)
                    .run_if(resource_exists::<DemoState>),
            )
                .run_if(resource_exists::<BurnHumanAssets>),
        )
        .add_systems(Update, setup_scene_once.run_if(resource_exists::<BurnHumanAssets>))
        .add_systems(
            EguiPrimaryContextPass,
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
        use_reference_case: true,
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

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.85, 0.85, 0.9),
        brightness: 1.05,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 3_000.0,
            shadows_enabled: false,
            color: Color::srgb(0.97, 0.97, 1.0),
            ..default()
        },
        Transform::from_xyz(5.0, 7.0, 4.5).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 1_800.0,
            shadows_enabled: false,
            color: Color::srgb(0.9, 0.94, 1.0),
            ..default()
        },
        Transform::from_xyz(-4.0, 5.5, -4.5).looking_at(Vec3::new(0.0, 1.1, 0.0), Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 620.0,
            shadows_enabled: false,
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
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.7, 0.68),
            metallic: 0.0,
            reflectance: 0.5,
            perceptual_roughness: 0.55,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(1.15)),
        Visibility::default(),
        Name::new("burn_human"),
        HumanTag,
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.8, 1.6, 4.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        PanOrbitCamera::default(),
    ));

    spawned.0 = true;
}

fn setup_noise(mut commands: Commands, has_noise: Option<Res<NoiseRig>>) {
    if has_noise.is_none() {
        commands.insert_resource(NoiseRig {
            noise: OpenSimplex::new(42),
        });
    }
}

fn ui_controls(
    mut contexts: EguiContexts,
    assets: Res<BurnHumanAssets>,
    mut state: ResMut<DemoState>,
    mut query: Query<&mut BurnHumanInput, With<HumanTag>>,
) {
    let mut input = if let Ok(i) = query.single_mut() {
        i
    } else {
        return;
    };

    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::Window::new("burn_human controls").show(ctx, |ui| {
        ui.label("Reference data exported from the bundled Python Anny model.");
        ui.separator();
        ui.checkbox(&mut state.use_reference_case, "Use reference case");
        if state.use_reference_case {
            if let Some((idx, name)) = pick_single_sample_case(&mut state, &assets) {
                state.selected_case = idx;
                ui.label(format!("Reference case: {name}"));
                input.case_name = Some(name);
            } else {
                ui.label("No single-sample reference case available");
                input.case_name = None;
            }
            input.phenotype_inputs = None;
            input.blendshape_weights = None;
            input.blendshape_delta = None;
        } else {
            input.case_name = None;
            ui.label("Phenotype sliders (drive blendshapes via mask).");
            for idx in 0..state.phenotype_values.len() {
                let label = state.phenotype_labels.get(idx).cloned().unwrap_or_default();
                if let Some(value) = state.phenotype_values.get_mut(idx) {
                    ui.add(egui::Slider::new(value, 0.0..=1.0).text(label));
                }
            }
            if ui.button("Reset phenotype").clicked() {
                for v in state.phenotype_values.iter_mut() {
                    *v = 0.5;
                }
            }
            input.phenotype_inputs = Some(state.phenotype_values.clone());
        }

        ui.separator();
        ui.label(format!(
            "Blendshapes: {} · Bones: {}",
            assets.body.metadata().static_data.blendshapes.shape[0],
            assets.body.metadata().metadata.bone_labels.len()
        ));
        ui.label(format!(
            "Reference cases available: {}",
            assets.body.metadata().metadata.case_names.len()
        ));
        ui.separator();
        let was_noise = state.noise_enabled;
        ui.checkbox(&mut state.noise_enabled, "Procedural motion");
        if state.noise_enabled && !was_noise {
            state.phenotype_noise_baseline = state.phenotype_values.clone();
            state.bone_noise_baseline = state.bone_euler_deg.clone();
        }
        ui.add(
            egui::Slider::new(&mut state.noise_amp, 0.0..=2.0)
                .text("global noise")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.phenotype_noise_amp, 0.0..=4.0)
                .text("phenotype noise")
                .logarithmic(false),
        );
        ui.separator();
        ui.label("Pose noise (deg, per group)");
        ui.add(
            egui::Slider::new(&mut state.upper_leg_noise_amp, 0.0..=50.0)
                .text("upper leg")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.lower_leg_noise_amp, 0.0..=50.0)
                .text("lower leg")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.upper_arm_noise_amp, 0.0..=50.0)
                .text("upper arm")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.lower_arm_noise_amp, 0.0..=50.0)
                .text("lower arm")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.wrist_noise_amp, 0.0..=50.0)
                .text("wrist")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.hand_noise_amp, 0.0..=50.0)
                .text("hand/fingers")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.spine_noise_amp, 0.0..=50.0)
                .text("spine")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.other_pose_noise_amp, 0.0..=50.0)
                .text("other pose")
                .logarithmic(false),
        );
        ui.add(
            egui::Slider::new(&mut state.time_scale, 0.25..=3.0)
                .text("time scale")
                .logarithmic(false),
        );
        ui.separator();
        ui.label("Bone orientation (degrees)");
        egui::ComboBox::from_id_salt("bone_select")
            .selected_text(
                assets
                    .body
                    .metadata()
                    .metadata
                    .bone_labels
                    .get(state.selected_bone)
                    .cloned()
                    .unwrap_or_else(|| "bone".to_string()),
            )
            .show_ui(ui, |ui| {
                for (idx, name) in assets
                    .body
                    .metadata()
                    .metadata
                    .bone_labels
                    .iter()
                    .enumerate()
                {
                    ui.selectable_value(&mut state.selected_bone, idx, name);
                }
            });
        let selected_bone = state.selected_bone;
        let mut euler = state
            .bone_euler_deg
            .get(selected_bone)
            .copied()
            .unwrap_or([0.0; 3]);
        ui.add(egui::Slider::new(&mut euler[0], -90.0..=90.0).text("X"));
        ui.add(egui::Slider::new(&mut euler[1], -90.0..=90.0).text("Y"));
        ui.add(egui::Slider::new(&mut euler[2], -90.0..=90.0).text("Z"));
        if ui.button("Reset bone").clicked() {
            euler = [0.0; 3];
        }
        if let Some(slot) = state.bone_euler_deg.get_mut(selected_bone) {
            *slot = euler;
        }
    });
}

fn gate_pan_orbit_during_egui(mut contexts: EguiContexts, mut query: Query<&mut PanOrbitCamera>) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let block =
        ctx.is_pointer_over_area() || ctx.wants_pointer_input() || ctx.wants_keyboard_input();
    for mut cam in query.iter_mut() {
        cam.enabled = !block;
    }
}

fn apply_random_pose_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    assets: Res<BurnHumanAssets>,
    mut state: ResMut<DemoState>,
    mut query: Query<&mut BurnHumanInput, With<HumanTag>>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
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

fn pick_single_sample_case(
    state: &mut DemoState,
    assets: &BurnHumanAssets,
) -> Option<(usize, String)> {
    let cases = &assets.body.metadata().cases;
    let names = &assets.body.metadata().metadata.case_names;
    if let Some((idx, _)) = cases
        .iter()
        .enumerate()
        .find(|(_, c)| c.pose_parameters.shape[0] == 1)
    {
        state.selected_case = idx;
        return names.get(idx).cloned().map(|n| (idx, n));
    }
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
            pick_single_sample_case(&mut state, &assets).map(|(_, n)| n)
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
    let phen_labels = assets.body.metadata().metadata.phenotype_labels.clone();
    let phen_baseline = state.phenotype_noise_baseline.clone();
    let noise_amp = state.noise_amp as f64;
    let phen_amp = state.phenotype_noise_amp as f64;
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

    let bone_count = state.bone_euler_deg.len();
    state.bone_noise_baseline.resize(bone_count, [0.0; 3]);
    let bone_baseline = state.bone_noise_baseline.clone();
    let noise_amp_f32 = state.noise_amp;
    let bone_labels = assets.body.metadata().metadata.bone_labels.clone();
    let pose_scales = PoseNoiseScales::from(&*state);
    for (idx, bone) in state.bone_euler_deg.iter_mut().enumerate() {
        if idx == 0 {
            *bone = [0.0; 3];
            continue;
        }
        let base = bone_baseline.get(idx).copied().unwrap_or([0.0; 3]);
        let nx = noise.noise.get([t * 0.35, idx as f64 * 0.17]);
        let ny = noise.noise.get([t * 0.45 + 97.0, idx as f64 * 0.23]);
        let nz = noise.noise.get([t * 0.55 + 197.0, idx as f64 * 0.13]);
        let group_amp = bone_labels
            .get(idx)
            .map(|name| pose_noise_scale_for_bone(name, &pose_scales))
            .unwrap_or(pose_scales.other);
        let amp = group_amp * noise_amp_f32;
        bone[0] = (base[0] + (nx as f32) * amp).clamp(-90.0, 90.0);
        bone[1] = (base[1] + (ny as f32) * amp).clamp(-90.0, 90.0);
        bone[2] = (base[2] + (nz as f32) * amp).clamp(-90.0, 90.0);
    }

    input.case_name = None;
    input.phenotype_inputs = Some(state.phenotype_values.clone());
    input.blendshape_weights = None;
    input.blendshape_delta = None;
    apply_pose_from_state(&mut input, &assets, &state);
}
