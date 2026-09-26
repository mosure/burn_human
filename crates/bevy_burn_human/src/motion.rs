//! Optional-on-demand motion inference and playback. No checkpoint is fetched
//! until the user loads a bundle. Burn shares Bevy's adapter/device/queue.
mod body;
mod body_ui;
mod body_view;
#[cfg(target_arch = "wasm32")]
mod browser_io;
mod runtime;
mod ui;

use crate::{BurnHumanAssets, BurnHumanInput};
use bevy::prelude::*;
use bevy_egui::EguiPrimaryContextPass;
use burn_human::motion::AnnyMotionBinding;
use burn_human_motion::{MotionClip, MotionRequest};
pub use runtime::{MotionRuntime, MotionStatus};
pub use ui::MotionUi;

pub struct HumanMotionPlugin;
impl Plugin for HumanMotionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MotionRuntime>()
            .init_resource::<MotionUi>()
            .init_resource::<MotionPlayback>()
            .init_resource::<body_view::BodyDisplay>()
            .add_systems(EguiPrimaryContextPass, ui::controls)
            .add_systems(
                PreUpdate,
                (receive_clip, playback)
                    .chain()
                    .after(bevy::time::TimeSystems),
            )
            .add_systems(
                Update,
                (
                    ui::place_waypoints,
                    draw_motion,
                    frame_trajectory,
                    body_view::update,
                    body_view::draw,
                ),
            );
    }
    fn finish(&self, app: &mut App) {
        runtime::initialize_device(app);
    }
}

/// Attach to an Anny entity to let MotionPlayback animate it.
#[derive(Component)]
pub struct MotionActor;

#[derive(Resource)]
pub struct MotionPlayback {
    pub clip: Option<MotionClip>,
    pub playing: bool,
    pub active: bool,
    pub looping: bool,
    pub time: f32,
    pub speed: f32,
    pub show_skeleton: bool,
    pub show_path: bool,
    pub joint: usize,
    pub joint_euler: [f32; 3],
    pub frame_view: bool,
    pub fit_root_height: bool,
    binding: Option<AnnyMotionBinding>,
    phenotype: Vec<f64>,
}
impl Default for MotionPlayback {
    fn default() -> Self {
        Self {
            clip: None,
            playing: false,
            active: false,
            looping: false,
            time: 0.0,
            speed: 1.0,
            show_skeleton: true,
            show_path: true,
            joint: 0,
            joint_euler: [0.0; 3],
            frame_view: false,
            fit_root_height: true,
            binding: None,
            phenotype: Vec::new(),
        }
    }
}

fn receive_clip(runtime: Res<MotionRuntime>, mut playback: ResMut<MotionPlayback>) {
    if let Some(clip) = runtime.0.lock().unwrap().clip.take() {
        playback.clip = Some(clip);
        playback.time = 0.0;
        playback.playing = true;
        playback.active = true;
        playback.binding = None;
        playback.joint = 0;
        playback.joint_euler = [0.0; 3];
        playback.frame_view = true;
    }
}

fn playback(
    time: Res<Time>,
    assets: Option<Res<BurnHumanAssets>>,
    mut player: ResMut<MotionPlayback>,
    mut actors: Query<(&mut BurnHumanInput, &mut Transform), With<MotionActor>>,
    runtime: Res<MotionRuntime>,
    body_display: Res<body_view::BodyDisplay>,
) {
    if !player.active || body_display.visible {
        return;
    }
    let Some(assets) = assets else {
        return;
    };
    let Some(clip) = player.clip.as_ref() else {
        return;
    };
    let duration = clip.duration();
    if player.playing {
        player.time += time.delta_secs() * player.speed;
        if player.time > duration {
            if player.looping && duration > 0.0 {
                player.time %= duration;
            } else {
                player.time = duration;
                player.playing = false;
            }
        }
    }
    for (mut input, mut transform) in &mut actors {
        let phenotype = input
            .phenotype_inputs
            .clone()
            .unwrap_or_else(|| vec![0.5; assets.body.metadata().metadata.phenotype_labels.len()]);
        if player.binding.is_none() || player.phenotype != phenotype {
            match AnnyMotionBinding::new(
                &assets.body,
                &phenotype,
                &player.clip.as_ref().unwrap().rig,
            ) {
                Ok(binding) => {
                    player.binding = Some(binding);
                    player.phenotype = phenotype.clone();
                }
                Err(e) => {
                    runtime.0.lock().unwrap().status = MotionStatus::Failed(e.to_string());
                    player.active = false;
                    return;
                }
            }
        }
        let clip = player.clip.as_ref().unwrap();
        let mut frame = clip.sample(player.time);
        if player.joint < frame.local_rotations.len() {
            frame.local_rotations[player.joint] *= Quat::from_euler(
                EulerRot::XYZ,
                player.joint_euler[0].to_radians(),
                player.joint_euler[1].to_radians(),
                player.joint_euler[2].to_radians(),
            );
        }
        if player.fit_root_height {
            frame.root_translation.y *= player.binding.as_ref().unwrap().root_height_scale;
        }
        match player
            .binding
            .as_ref()
            .unwrap()
            .pose_parameters(&clip.rig, &frame)
        {
            Ok(pose) => {
                if input.case_name.is_some()
                    || input.phenotype_inputs.as_ref() != Some(&phenotype)
                    || input.pose_parameters.as_ref() != Some(&pose)
                    || input.pose_parameters_delta.is_some()
                    || input.root_translation_delta.is_some()
                {
                    input.case_name = None;
                    input.phenotype_inputs = Some(phenotype);
                    input.pose_parameters = Some(pose);
                    input.pose_parameters_delta = None;
                    input.root_translation_delta = None;
                }
                if transform.scale != Vec3::ONE || transform.translation != Vec3::ZERO {
                    transform.scale = Vec3::ONE;
                    transform.translation = Vec3::ZERO;
                }
            }
            Err(e) => {
                runtime.0.lock().unwrap().status = MotionStatus::Failed(e.to_string());
                player.active = false;
            }
        }
    }
}

fn draw_motion(
    mut gizmos: Gizmos,
    player: Res<MotionPlayback>,
    ui: Res<MotionUi>,
    body_display: Res<body_view::BodyDisplay>,
) {
    if player.show_path {
        for i in -10..=10 {
            let d = i as f32;
            let c = Color::srgb(0.16, 0.2, 0.27);
            gizmos.line(Vec3::new(d, 0.0, -10.0), Vec3::new(d, 0.0, 10.0), c);
            gizmos.line(Vec3::new(-10.0, 0.0, d), Vec3::new(10.0, 0.0, d), c);
        }
        if body_display.visible {
            return;
        }
        let request: &MotionRequest = &ui.request;
        for pair in request.waypoints.windows(2) {
            gizmos.line(
                pair[0].position,
                pair[1].position,
                Color::srgb(1.0, 0.7, 0.1),
            );
        }
        for waypoint in &request.waypoints {
            let p = waypoint.position;
            gizmos.sphere(
                Isometry3d::from_translation(p),
                0.06,
                Color::srgb(1.0, 0.65, 0.1),
            );
            gizmos.line(
                p - Vec3::X * 0.15,
                p + Vec3::X * 0.15,
                Color::srgb(1.0, 0.65, 0.1),
            );
            gizmos.line(
                p - Vec3::Z * 0.15,
                p + Vec3::Z * 0.15,
                Color::srgb(1.0, 0.65, 0.1),
            );
            if let Some(h) = waypoint.heading {
                gizmos.arrow(
                    p,
                    p + Vec3::new(-h.sin(), 0.0, h.cos()) * 0.3,
                    Color::srgb(0.9, 0.9, 0.1),
                );
            }
        }
        if let Some(clip) = &player.clip {
            for pair in clip.frames.windows(2) {
                let mut a = pair[0].root_translation;
                let mut b = pair[1].root_translation;
                a.y = 0.015;
                b.y = 0.015;
                gizmos.line(a, b, Color::srgb(0.1, 0.8, 1.0));
            }
        }
    }
    if !body_display.visible
        && player.show_skeleton
        && player.active
        && let Some(clip) = &player.clip
    {
        let mut frame = clip.sample(player.time);
        if player.joint < frame.local_rotations.len() {
            frame.local_rotations[player.joint] *= Quat::from_euler(
                EulerRot::XYZ,
                player.joint_euler[0].to_radians(),
                player.joint_euler[1].to_radians(),
                player.joint_euler[2].to_radians(),
            );
        }
        if let Ok((positions, _)) = clip.rig.forward(&frame) {
            for (i, j) in clip.rig.joints.iter().enumerate() {
                if let Some(p) = j.parent {
                    gizmos.line(positions[p], positions[i], Color::srgb(0.2, 1.0, 0.5));
                }
            }
        }
    }
}

fn frame_trajectory(
    mut player: ResMut<MotionPlayback>,
    ui: Res<MotionUi>,
    mut cameras: Query<&mut bevy_panorbit_camera::PanOrbitCamera>,
) {
    if !player.frame_view {
        return;
    }
    player.frame_view = false;
    let mut low = Vec3::ZERO;
    let mut high = Vec3::Y * 1.9;
    for p in ui.request.waypoints.iter().map(|w| w.position).chain(
        player
            .clip
            .iter()
            .flat_map(|c| c.frames.iter().map(|f| f.root_translation)),
    ) {
        low = low.min(p);
        high = high.max(p + Vec3::Y * 0.9);
    }
    for mut camera in &mut cameras {
        camera.target_focus = (low + high) * 0.5;
        // Include perspective depth and room for the side panel, not only the
        // projected root path; otherwise an approaching actor can leave view.
        camera.target_radius = ((high - low).length() * 1.8).max(4.0);
        camera.force_update = true;
    }
}
