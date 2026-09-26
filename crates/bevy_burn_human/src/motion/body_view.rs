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
        }
    }
}
#[derive(Component)]
pub(super) struct SomaActor;

#[allow(clippy::too_many_arguments)] // Bevy injects each system resource/query.
pub(super) fn update(
    mut commands: Commands,
    runtime: Res<MotionRuntime>,
    mut display: ResMut<BodyDisplay>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut actors: Query<(&mut Visibility, &mut Transform), With<SomaActor>>,
    mut anny: Query<&mut Visibility, (With<MotionActor>, Without<SomaActor>)>,
    mut cameras: Query<&mut bevy_panorbit_camera::PanOrbitCamera>,
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
        display.frame_view = display.surface.is_none();
        display.surface = Some(surface);
        display.visible = true;
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
        *visibility = if active {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
    if display.frame_view {
        display.frame_view = false;
        for mut camera in &mut cameras {
            camera.target_focus = Vec3::Y * 0.9;
            camera.target_radius = 4.0;
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

pub(super) fn draw(mut gizmos: Gizmos, display: Res<BodyDisplay>) {
    if !display.visible || !display.skeleton {
        return;
    }
    let Some(surface) = &display.surface else {
        return;
    };
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
