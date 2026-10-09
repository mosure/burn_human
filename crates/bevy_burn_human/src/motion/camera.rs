//! Input adapter for Bevy 0.20's built-in pan-orbit controller.
use super::{MotionPlayback, MotionUi, body_view::BodyDisplay};
use crate::studio_ui::StudioInput;
pub(super) use bevy::camera_controller::pan_orbit_camera::prelude::PanOrbitCamera;
use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    window::PrimaryWindow,
};
#[derive(Clone, Copy)]
pub(super) enum View {
    Home,
    Front,
    Side,
    Top,
}
#[derive(Resource, Default)]
pub(super) struct ViewControls {
    pub requested: Option<View>,
    active: Option<MouseButton>,
}
/// Reset controller momentum when moving the camera programmatically.
pub(super) fn frame(
    camera: &mut PanOrbitCamera,
    transform: &mut Transform,
    focus: Vec3,
    distance: f32,
) {
    camera.current_motion = Default::default();
    camera.last_anchor_depth = -(distance as f64);
    transform.translation = focus + transform.rotation * Vec3::Z * distance;
}
#[allow(clippy::too_many_arguments)]
pub(super) fn update(
    input: Res<StudioInput>,
    mut motion: ResMut<MotionUi>,
    mut view: ResMut<ViewControls>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut moves: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
    mut player: ResMut<MotionPlayback>,
    mut body: ResMut<BodyDisplay>,
    mut cameras: Query<(
        &mut PanOrbitCamera,
        &mut Transform,
        &GlobalTransform,
        &Camera,
        &Projection,
    )>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let delta: Vec2 = moves.read().map(|m| m.delta).sum();
    let zoom: f32 = scroll
        .read()
        .map(|m| {
            m.y * if m.unit == MouseScrollUnit::Line {
                150.
            } else {
                1.
            }
        })
        .sum();
    if !input.keyboard {
        if keys.just_pressed(KeyCode::Escape) {
            motion.place_waypoints = false;
        }
        if keys.just_pressed(KeyCode::Home) {
            view.requested = Some(View::Home);
        }
        if keys.just_pressed(KeyCode::KeyF) {
            if motion.is_body_tab() {
                body.frame_view = true;
            } else {
                player.frame_view = true;
            }
        }
    }
    let orbit = if motion.place_waypoints {
        MouseButton::Middle
    } else {
        MouseButton::Left
    };
    for (mut camera, mut transform, global, lens, _) in &mut cameras {
        if let Some(preset) = view.requested {
            let focus = if matches!(preset, View::Home) {
                Vec3::Y
            } else {
                transform.translation
                    + transform.forward().as_vec3() * camera.last_anchor_depth().abs() as f32
            };
            let (yaw, pitch) = match preset {
                View::Home => (0.6, 0.18),
                View::Front => (0., 0.),
                View::Side => (std::f32::consts::FRAC_PI_2, 0.),
                View::Top => (0., std::f32::consts::FRAC_PI_2 - 0.02),
            };
            transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, -pitch, 0.);
            let distance = if matches!(preset, View::Home) {
                4.5
            } else {
                camera.last_anchor_depth().abs() as f32
            };
            frame(&mut camera, &mut transform, focus, distance);
        }
        let blocked = input.pointer || input.keyboard || windows.single().is_ok_and(|w| !w.focused);
        if blocked {
            if view.active.take().is_some() {
                camera.end_move();
                camera.current_motion = Default::default();
            }
            continue;
        }
        if view.active.is_some_and(|b| buttons.just_released(b)) {
            camera.end_move();
            view.active = None;
        }
        let continuing_gesture = view.active.is_some();
        // The depth plane remains stable when there is no pickable geometry.
        // Ground intersection gives world-space anchoring during path discovery.
        let anchor = windows
            .single()
            .ok()
            .and_then(Window::cursor_position)
            .and_then(|p| lens.viewport_to_world(global, p).ok())
            .and_then(|ray| {
                let t = -ray.origin.y / ray.direction.y;
                (t.is_finite() && (0.1..100.).contains(&t)).then(|| {
                    global
                        .to_matrix()
                        .inverse()
                        .transform_point3(ray.origin + ray.direction * t)
                        .as_dvec3()
                })
            });
        if buttons.just_pressed(orbit) {
            camera.start_orbit(anchor);
            view.active = Some(orbit);
        } else if buttons.just_pressed(MouseButton::Right) {
            camera.start_pan(anchor);
            view.active = Some(MouseButton::Right);
        } else if zoom != 0. && !camera.is_actively_controlled() {
            camera.start_zoom(anchor);
        }
        // MouseMotion also includes hovering before the press. Applying that
        // delta to a newly started orbit makes a simple scene click jump.
        if continuing_gesture && view.active.is_some() {
            camera.send_screenspace_input(delta);
        }
        if zoom != 0. {
            camera.send_zoom_input(zoom);
        }
        // A scroll gesture has no button-release event.
        if view.active.is_none() && zoom == 0. && camera.current_motion.is_zooming_only() {
            camera.end_move();
        }
    }
    view.requested = None;
}
