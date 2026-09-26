use super::{MotionActor, MotionRuntime, body::Surface};
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
#[derive(Resource)]
pub(super) struct BodyDisplay {
    pub visible: bool,
    pub skeleton: bool,
    pub frame_view: bool,
    mesh: Option<Handle<Mesh>>,
    surface: Option<Surface>,
    transform: Transform,
    center: Vec3,
    radius: f32,
}
impl Default for BodyDisplay {
    fn default() -> Self {
        Self {
            visible: false,
            skeleton: true,
            frame_view: false,
            mesh: None,
            surface: None,
            transform: Transform::default(),
            center: Vec3::Y * 0.9,
            radius: 1.3,
        }
    }
}
#[derive(Component)]
pub(super) struct SomaActor;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub(super) struct RigGizmos;

pub(super) fn configure_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<RigGizmos>();
    config.depth_bias = -1.0;
    config.line.width = 1.5;
}

#[allow(clippy::too_many_arguments)] // Bevy injects each system resource/query.
pub(super) fn update(
    mut commands: Commands,
    runtime: Res<MotionRuntime>,
    ui: Res<super::MotionUi>,
    mut display: ResMut<BodyDisplay>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut actors: Query<(&mut Visibility, &mut Transform), With<SomaActor>>,
    mut anny: Query<&mut Visibility, (With<MotionActor>, Without<SomaActor>)>,
    mut cameras: Query<(&mut bevy_panorbit_camera::PanOrbitCamera, &Projection)>,
) {
    let pending = runtime.0.lock().unwrap().body.surface.take();
    if let Some(surface) = pending {
        let mesh = body_mesh(&surface.vertices, &surface.faces);
        let rotation = if surface.camera_space {
            Quat::from_rotation_x(std::f32::consts::PI)
        } else {
            Quat::IDENTITY
        };
        let mut low = Vec3::splat(f32::INFINITY);
        let mut high = Vec3::splat(f32::NEG_INFINITY);
        for p in &surface.vertices {
            let p = rotation * Vec3::from_array(*p);
            low = low.min(p);
            high = high.max(p);
        }
        let center = (low + high) * 0.5;
        display.transform = Transform::from_rotation(rotation)
            .with_translation(Vec3::new(-center.x, -low.y, -center.z));
        display.center = Vec3::Y * (high.y - low.y) * 0.5;
        display.radius = ((high - low).length() * 0.5).max(0.1);
        if let Some(handle) = &display.mesh {
            if let Some(mut current) = meshes.get_mut(handle) {
                *current = mesh;
            }
        } else {
            let handle = meshes.add(mesh);
            commands.spawn((
                SomaActor,
                Mesh3d(handle.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.55, 0.74, 0.9),
                    perceptual_roughness: 0.65,
                    cull_mode: None,
                    ..default()
                })),
                display.transform,
                Visibility::Visible,
                Name::new("SOMA body"),
            ));
            display.mesh = Some(handle);
        }
        display.frame_view |= display.surface.is_none() && ui.is_body_tab();
        display.surface = Some(surface);
    }
    let active = display.visible && display.surface.is_some();
    for (mut visibility, mut transform) in &mut actors {
        *visibility = if active {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *transform = display.transform;
    }
    for mut visibility in &mut anny {
        *visibility = if ui.is_body_tab() {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
    if display.frame_view {
        display.frame_view = false;
        for (mut camera, projection) in &mut cameras {
            camera.target_focus = display.center;
            // Fit the evaluated body after scale, identity or pose changes.
            let radius = display.radius * 1.12;
            camera.target_radius = match projection {
                Projection::Perspective(p) => {
                    let vertical = p.fov * 0.5;
                    let horizontal = (vertical.tan() * p.aspect_ratio).atan();
                    radius / vertical.min(horizontal).max(0.01).sin()
                }
                Projection::Orthographic(p) => {
                    // PanOrbit uses radius as the orthographic projection scale.
                    2.0 * radius * p.scale / p.area.size().min_element().max(0.01)
                }
                _ => camera.target_radius,
            };
            camera.force_update = true;
        }
    }
}

fn body_mesh(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices.to_vec());
    mesh.insert_indices(Indices::U32(faces.iter().flatten().copied().collect()));
    // Metre-scale face/hand triangles fall below the absolute corner-angle
    // epsilon in compute_smooth_normals, producing black zero-normal patches.
    mesh.compute_area_weighted_normals();
    mesh
}

pub(super) fn draw(
    mut gizmos: Gizmos<RigGizmos>,
    display: Res<BodyDisplay>,
    ui: Res<super::MotionUi>,
) {
    if !display.visible || !display.skeleton {
        return;
    }
    let Some(surface) = &display.surface else {
        return;
    };
    if ui.is_body_tab()
        && let Some(joint) = surface.joints.get(ui.body.joint + 1)
    {
        gizmos.sphere(
            Isometry3d::from_translation(
                display.transform.transform_point(Vec3::from_array(*joint)),
            ),
            0.015,
            Color::WHITE,
        );
    }
    for (i, &parent) in surface.parents.iter().enumerate().skip(1) {
        if parent == 0 {
            continue;
        }
        let a = display
            .transform
            .transform_point(Vec3::from_array(surface.joints[parent]));
        let b = display
            .transform
            .transform_point(Vec3::from_array(surface.joints[i]));
        gizmos.line(a, b, Color::srgb(1.0, 0.7, 0.1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn millimetre_triangles_have_finite_unit_normals() {
        let mesh = body_mesh(
            &[[0.0; 3], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]],
            &[[0, 1, 2]],
        );
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("missing normals")
        };
        assert!(
            normals
                .iter()
                .all(|n| (Vec3::from_array(*n) - Vec3::Z).length() < 1e-6)
        );
    }
}
