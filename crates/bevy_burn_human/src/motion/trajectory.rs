//! Timed world-space trajectory editing, independent of model loading.
use super::{MotionPlayback, MotionUi};
use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{EguiContexts, egui};
use burn_human_motion::{MotionRequest, Waypoint};

pub(super) fn retime(request: &mut MotionRequest) {
    let count = request.waypoints.len();
    // Keep every point representable when a long path is shortened.
    request.frames = request
        .frames
        .max(count.next_multiple_of(4))
        .clamp(40, 12000);
    for (i, point) in request.waypoints.iter_mut().enumerate() {
        point.frame = if count > 1 {
            i * (request.frames - 1) / (count - 1)
        } else {
            0
        };
    }
}

fn add(state: &mut MotionUi, position: Vec3) {
    if state.request.waypoints.len() >= state.request.frames {
        return;
    }
    state.request.waypoints.push(Waypoint {
        frame: 0,
        position,
        heading: None,
        constrain_height: false,
    });
    retime(&mut state.request);
    state.selected_waypoint = Some(state.request.waypoints.len() - 1);
}

pub(super) fn controls(ui: &mut egui::Ui, state: &mut MotionUi, playback: &mut MotionPlayback) {
    egui::CollapsingHeader::new("Trajectory · optional")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                for (label, points) in [
                    ("Straight", &[[0.0, 0.0], [0.0, 3.0]][..]),
                    ("Turn", &[[0.0, 0.0], [0.0, 1.5], [1.5, 2.5]][..]),
                    (
                        "Loop",
                        &[[0.0, 0.0], [-1.0, 1.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]][..],
                    ),
                ] {
                    if ui.button(label).clicked() {
                        state.request.waypoints.clear();
                        for [x, z] in points {
                            add(state, Vec3::new(*x, 0.02, *z));
                        }
                        state.selected_waypoint = None;
                        playback.show_path = true;
                        playback.frame_view = true;
                    }
                }
            });
            ui.toggle_value(&mut state.place_waypoints, "Edit path in scene");
            if state.place_waypoints {
                ui.small("Click ground to add; drag a marker to move it. Esc finishes editing.");
            }
            ui.small(
                "Gold = requested path · Blue = generated path. Points are soft motion conditions.",
            );
            ui.checkbox(
                &mut state.request.dense_trajectory,
                "Follow the path between points",
            );
            let mut remove = None;
            let count = state.request.waypoints.len();
            for i in 0..count {
                let min = if i == 0 {
                    0
                } else {
                    state.request.waypoints[i - 1].frame + 1
                };
                let max = if i + 1 == count {
                    state.request.frames - 1
                } else {
                    state.request.waypoints[i + 1].frame - 1
                };
                let point = &mut state.request.waypoints[i];
                ui.push_id(i, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(
                                state.selected_waypoint == Some(i),
                                format!("Point {}", i + 1),
                            )
                            .clicked()
                        {
                            state.selected_waypoint = Some(i);
                        }
                        let mut seconds = point.frame as f32 / 20.0;
                        if ui
                            .add(
                                egui::DragValue::new(&mut seconds)
                                    .range(min as f32 / 20.0..=max as f32 / 20.0)
                                    .speed(0.05)
                                    .suffix(" s"),
                            )
                            .changed()
                        {
                            point.frame = ((seconds * 20.0).round() as usize).clamp(min, max);
                        }
                        if ui.small_button("Remove").clicked() {
                            remove = Some(i);
                        }
                    });
                    if state.selected_waypoint == Some(i) {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut point.position.x)
                                    .prefix("X ")
                                    .suffix(" m")
                                    .speed(0.05),
                            );
                            ui.add(
                                egui::DragValue::new(&mut point.position.z)
                                    .prefix("Z ")
                                    .suffix(" m")
                                    .speed(0.05),
                            );
                        });
                        let mut heading = point.heading.is_some();
                        if ui.checkbox(&mut heading, "Set facing direction").changed() {
                            point.heading = heading.then_some(0.0);
                        }
                        if let Some(h) = &mut point.heading {
                            let mut degrees = h.to_degrees();
                            ui.add(
                                egui::Slider::new(&mut degrees, -180.0..=180.0).text("Heading °"),
                            );
                            *h = degrees.to_radians();
                        }
                        ui.checkbox(&mut point.constrain_height, "Set root height");
                        if point.constrain_height {
                            ui.add(
                                egui::DragValue::new(&mut point.position.y)
                                    .prefix("Y ")
                                    .suffix(" m")
                                    .speed(0.02),
                            );
                        }
                    }
                });
            }
            if let Some(i) = remove {
                state.request.waypoints.remove(i);
                state.selected_waypoint = None;
            }
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(count < state.request.frames, egui::Button::new("Add point"))
                    .clicked()
                {
                    add(state, Vec3::new(0.0, 0.02, 0.0));
                }
                if ui.button("Clear path").clicked() {
                    state.request.waypoints.clear();
                    state.selected_waypoint = None;
                }
                if ui.button("Frame path").clicked() {
                    playback.frame_view = true;
                }
            });
            ui.checkbox(&mut playback.show_path, "Show paths and ground grid");
        });
}

pub(super) fn place_waypoints(
    mut state: ResMut<MotionUi>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut contexts: EguiContexts,
) {
    if !state.is_motion_tab() || !state.place_waypoints {
        state.dragging_waypoint = false;
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    if ctx.egui_wants_keyboard_input() {
        return;
    }
    if keys.just_pressed(KeyCode::Delete)
        && let Some(index) = state.selected_waypoint.take()
    {
        state.request.waypoints.remove(index);
        state.dragging_waypoint = false;
    }
    if !buttons.pressed(MouseButton::Left) {
        state.dragging_waypoint = false;
        return;
    }
    if ctx.is_pointer_over_egui() || ctx.egui_wants_pointer_input() {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    if buttons.just_pressed(MouseButton::Left) {
        let hit = state
            .request
            .waypoints
            .iter()
            .enumerate()
            .filter_map(|(i, w)| {
                camera
                    .world_to_viewport(transform, w.position)
                    .ok()
                    .map(|p| (i, p.distance(cursor)))
            })
            .filter(|(_, d)| *d <= 18.0)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        state.selected_waypoint = hit.map(|(i, _)| i);
        state.dragging_waypoint = hit.is_some();
    }
    let Ok(ray) = camera.viewport_to_world(transform, cursor) else {
        return;
    };
    if ray.direction.y.abs() < 1e-6 {
        return;
    }
    let distance = (0.02 - ray.origin.y) / ray.direction.y;
    if !(0.0..1000.0).contains(&distance) {
        return;
    }
    let point = ray.origin + ray.direction * distance;
    if state.dragging_waypoint {
        if let Some(i) = state.selected_waypoint
            && let Some(w) = state.request.waypoints.get_mut(i)
        {
            w.position.x = point.x;
            w.position.z = point.z;
        }
    } else if buttons.just_pressed(MouseButton::Left) {
        add(&mut state, point);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adding_and_shortening_paths_preserves_order_and_valid_timing() {
        let mut state = MotionUi::default();
        for i in 0..48 {
            add(&mut state, Vec3::new(i as f32, 0.02, 0.0));
        }
        state.request.frames = 40;
        retime(&mut state.request);
        assert_eq!(state.request.frames, 48);
        state.request.validate().unwrap();
        assert_eq!(state.request.waypoints.last().unwrap().frame, 47);
        assert_eq!(state.request.waypoints[20].position.x, 20.0);
    }
}
