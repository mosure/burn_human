//! Scene navigation shares the current egui input decision with waypoint editing.
use super::{MotionPlayback, MotionUi, body_view::BodyDisplay};
use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{EguiContexts, egui};
use bevy_panorbit_camera::PanOrbitCamera;

#[derive(Clone, Copy)]
enum View {
    Home,
    Front,
    Side,
    Top,
}

#[derive(Resource, Default)]
pub(super) struct ViewControls {
    requested: Option<View>,
}

pub(super) fn controls(ui: &mut egui::Ui, view: &mut ViewControls, placing: bool) {
    ui.horizontal(|ui| {
        for (label, preset) in [
            ("Home", View::Home),
            ("Front", View::Front),
            ("Side", View::Side),
            ("Top", View::Top),
        ] {
            if ui.small_button(label).clicked() {
                view.requested = Some(preset);
            }
        }
    });
    ui.small(if placing {
        "Left: add / drag point · Middle: orbit · Right: pan\nScroll: zoom · Esc: finish path · Delete: remove point"
    } else {
        "Left drag: orbit · Right drag: pan · Scroll: zoom\nF: frame subject / path · Home: reset view"
    });
}

#[allow(clippy::too_many_arguments)] // Bevy injects independent input and view resources.
pub(super) fn update(
    mut contexts: EguiContexts,
    mut motion: ResMut<MotionUi>,
    mut view: ResMut<ViewControls>,
    keys: Res<ButtonInput<KeyCode>>,
    mut player: ResMut<MotionPlayback>,
    mut body: ResMut<BodyDisplay>,
    mut cameras: Query<&mut PanOrbitCamera>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let typing = ctx.egui_wants_keyboard_input();
    if !typing {
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
    let block = ctx.is_pointer_over_egui()
        || ctx.egui_wants_pointer_input()
        || typing
        || windows.single().is_ok_and(|w| !w.focused);
    for mut camera in &mut cameras {
        camera.enabled = !block;
        camera.button_orbit = if motion.place_waypoints {
            MouseButton::Middle
        } else {
            MouseButton::Left
        };
        if let Some(view) = view.requested {
            (camera.target_yaw, camera.target_pitch) = match view {
                View::Home => (0.6, 0.18),
                View::Front => (0.0, 0.0),
                View::Side => (std::f32::consts::FRAC_PI_2, 0.0),
                View::Top => (0.0, std::f32::consts::FRAC_PI_2 - 0.02),
            };
            if matches!(view, View::Home) {
                camera.target_focus = Vec3::Y;
                camera.target_radius = 4.5;
            }
            camera.force_update = true;
        }
    }
    view.requested = None;
}
