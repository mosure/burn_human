//! Timed world-space trajectory editing, independent of model loading.
use super::MotionUi;
use crate::studio_ui::StudioInput;
use bevy::{prelude::*, window::PrimaryWindow};
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

pub(super) fn add(state: &mut MotionUi, position: Vec3) {
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

pub(super) fn place_waypoints(
    mut state: ResMut<MotionUi>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    input: Res<StudioInput>,
) {
    if !state.is_motion_tab() || !state.place_waypoints {
        state.dragging_waypoint = false;
        return;
    }
    if input.keyboard {
        return;
    }
    if keys.just_pressed(KeyCode::Delete)
        && let Some(index) = state.selected_waypoint.take()
        && index < state.request.waypoints.len()
    {
        state.request.waypoints.remove(index);
        retime(&mut state.request);
        state.dragging_waypoint = false;
    }
    if !buttons.pressed(MouseButton::Left) {
        state.dragging_waypoint = false;
        return;
    }
    if input.pointer {
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
