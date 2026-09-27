use super::{
    MotionActor, MotionRuntime,
    body::Surface,
    surface::{self, BodyMaterial, SurfaceMaterial},
};
use bevy::{
    camera::{primitives::Aabb, visibility::NoAutoAabb},
    prelude::*,
};
#[derive(Resource)]
pub(super) struct BodyDisplay {
    pub visible: bool,
    pub skeleton: bool,
    pub frame_view: bool,
    mesh: Option<Handle<Mesh>>,
    material: Option<Handle<BodyMaterial>>,
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
            material: None,
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
    mut materials: ResMut<Assets<BodyMaterial>>,
    mut actors: Query<(&mut Visibility, &mut Transform, &mut Aabb, &mut Mesh3d), With<SomaActor>>,
    mut anny: Query<&mut Visibility, (With<MotionActor>, Without<SomaActor>)>,
    mut cameras: Query<(&mut bevy_panorbit_camera::PanOrbitCamera, &Projection)>,
) {
    let pending = runtime.0.lock().unwrap().body.surface.take();
    if let Some(surface) = pending {
        let rotation = if surface.camera_space {
            Quat::from_rotation_x(std::f32::consts::PI)
        } else {
            Quat::IDENTITY
        };
        let a = rotation * surface.gpu.low;
        let b = rotation * surface.gpu.high;
        let low = a.min(b);
        let high = a.max(b);
        let bounds = Aabb::from_min_max(surface.gpu.low, surface.gpu.high);
        let center = (low + high) * 0.5;
        display.transform = Transform::from_rotation(rotation)
            .with_translation(Vec3::new(-center.x, -low.y, -center.z));
        display.center = Vec3::Y * (high.y - low.y) * 0.5;
        display.radius = ((high - low).length() * 0.5).max(0.1);
        if let Some(handle) = &display.material {
            if let Some(mut material) = materials.get_mut(handle) {
                material.extension.vertices = surface.gpu.vertices.clone();
            }
            let topology_changed = display.surface.as_ref().is_none_or(|previous| {
                !std::sync::Arc::ptr_eq(&previous.gpu.topology, &surface.gpu.topology)
            });
            if topology_changed {
                let handle = meshes.add(surface::mesh(&surface.gpu.topology));
                for (_, _, _, mut mesh) in &mut actors {
                    mesh.0 = handle.clone();
                }
                // Topology changes only when a different model is loaded.
                display.mesh = Some(handle);
            }
            for (_, _, mut current_bounds, _) in &mut actors {
                *current_bounds = bounds;
            }
        } else {
            let handle = meshes.add(surface::mesh(&surface.gpu.topology));
            let material = materials.add(BodyMaterial {
                base: StandardMaterial {
                    base_color: Color::srgb(0.55, 0.74, 0.9),
                    perceptual_roughness: 0.65,
                    cull_mode: None,
                    ..default()
                },
                extension: SurfaceMaterial {
                    vertices: surface.gpu.vertices.clone(),
                },
            });
            commands.spawn((
                SomaActor,
                Mesh3d(handle.clone()),
                MeshMaterial3d(material.clone()),
                display.transform,
                bounds,
                // Static mesh positions are placeholders. Keep the evaluated
                // GPU bounds when Mesh3d changes, while retaining frustum culling.
                NoAutoAabb,
                Visibility::Visible,
                Name::new("SOMA body"),
            ));
            display.mesh = Some(handle);
            display.material = Some(material);
        }
        display.frame_view |= display.surface.is_none() && ui.is_body_tab();
        display.surface = Some(surface);
    }
    let active = display.visible && display.surface.is_some();
    for (mut visibility, mut transform, _, _) in &mut actors {
        visibility.set_if_neq(if active {
            Visibility::Visible
        } else {
            Visibility::Hidden
        });
        transform.set_if_neq(display.transform);
    }
    for mut visibility in &mut anny {
        visibility.set_if_neq(if ui.is_body_tab() {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        });
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

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn check_changed_topology_keeps_gpu_bounds(
    gpu: super::surface::GpuSurface,
    replacement: std::sync::Arc<super::surface::Topology>,
) {
    use bevy::camera::visibility::calculate_bounds;
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
        .init_asset::<Mesh>()
        .init_asset::<BodyMaterial>()
        .init_resource::<MotionRuntime>()
        .init_resource::<super::MotionUi>()
        .init_resource::<BodyDisplay>()
        .add_systems(Update, (update, calculate_bounds).chain());
    let vertices = gpu.vertices.clone();
    let first_bounds = Aabb::from_min_max(gpu.low, gpu.high);
    let make_surface = |gpu| Surface {
        gpu,
        joints: vec![],
        parents: vec![],
        camera_space: true,
    };
    app.world()
        .resource::<MotionRuntime>()
        .0
        .lock()
        .unwrap()
        .body
        .surface = Some(make_surface(gpu));
    app.update();
    let actor = app
        .world_mut()
        .query_filtered::<Entity, With<SomaActor>>()
        .single(app.world())
        .unwrap();
    assert_eq!(*app.world().get::<Aabb>(actor).unwrap(), first_bounds);
    // A later model changes the Mesh3d handle. Bevy must not overwrite these
    // evaluated bounds with the static mesh's all-zero placeholder positions.
    let low = Vec3::new(2.0, 3.0, 4.0);
    let high = Vec3::new(3.0, 4.0, 5.0);
    app.world()
        .resource::<MotionRuntime>()
        .0
        .lock()
        .unwrap()
        .body
        .surface = Some(make_surface(super::surface::GpuSurface {
        vertices,
        topology: replacement,
        low,
        high,
    }));
    app.update();
    assert_eq!(
        *app.world().get::<Aabb>(actor).unwrap(),
        Aabb::from_min_max(low, high)
    );
}
