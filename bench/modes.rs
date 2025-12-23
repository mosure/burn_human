use bevy::asset::AssetPlugin;
use bevy::mesh::skinning::SkinnedMeshInverseBindposes;
use bevy::prelude::*;
use bevy_burn_human::{
    BurnHumanDefaults, BurnHumanInput, BurnHumanMeshMode, BurnHumanPlugin, BurnHumanSource,
};
use burn_human::data::reference::TensorData;
use burn_human::model::{blendshape, kinematics, skinning};
use burn_human::{AnnyBody, AnnyInput};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

struct BenchData {
    rest_vertices: TensorData<f64>,
    rest_bone_poses: TensorData<f64>,
    vertex_bone_indices: TensorData<i64>,
    vertex_bone_weights: TensorData<f64>,
    faces_quads: TensorData<i64>,
    bone_parents: Vec<i64>,
    pose_parameters: TensorData<f64>,
}

fn load_body() -> AnnyBody {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tensor = base.join("assets").join("model").join("fullbody_default.safetensors");
    let meta = base.join("assets").join("model").join("fullbody_default.meta.json");
    AnnyBody::from_reference_paths(tensor, meta).expect("load reference")
}

fn identity_pose_parameters(bones: usize) -> TensorData<f64> {
    let mut data = vec![0.0; bones * 16];
    for bone in 0..bones {
        let base = bone * 16;
        data[base] = 1.0;
        data[base + 5] = 1.0;
        data[base + 10] = 1.0;
        data[base + 15] = 1.0;
    }
    TensorData {
        shape: vec![1, bones, 4, 4],
        data,
    }
}

fn identity_pose_vec(bones: usize) -> Vec<f64> {
    identity_pose_parameters(bones).data
}

fn build_bench_data(body: &AnnyBody) -> BenchData {
    let bundle = body.metadata();
    let phenotype_len = bundle.metadata.phenotype_labels.len();
    let phenotype_inputs = TensorData {
        shape: vec![1, phenotype_len],
        data: vec![0.5; phenotype_len],
    };
    let blendshape_weights = body
        .phenotype_evaluator()
        .weights(&phenotype_inputs)
        .expect("phenotype weights");

    let rest_vertices = blendshape::apply_blendshapes(
        &bundle.static_data.template_vertices,
        &bundle.static_data.blendshapes,
        &blendshape_weights,
    )
    .expect("rest vertices");
    let rest_bone_heads = blendshape::apply_bone_blendshapes(
        &bundle.static_data.template_bone_heads,
        &bundle.static_data.bone_heads_blendshapes,
        &blendshape_weights,
    )
    .expect("rest bone heads");
    let rest_bone_tails = blendshape::apply_bone_blendshapes(
        &bundle.static_data.template_bone_tails,
        &bundle.static_data.bone_tails_blendshapes,
        &blendshape_weights,
    )
    .expect("rest bone tails");
    let rest_bone_poses = kinematics::rest_bone_poses_from_heads_tails(
        &rest_bone_heads,
        &rest_bone_tails,
        &bundle.static_data.bone_rolls_rotmat,
    )
    .expect("rest bone poses");

    let bones = bundle.metadata.bone_labels.len();
    let pose_parameters = identity_pose_parameters(bones);

    BenchData {
        rest_vertices,
        rest_bone_poses,
        vertex_bone_indices: bundle.static_data.vertex_bone_indices.clone(),
        vertex_bone_weights: bundle.static_data.vertex_bone_weights.clone(),
        faces_quads: bundle.static_data.faces_quads.clone(),
        bone_parents: bundle.metadata.bone_parents.clone(),
        pose_parameters,
    }
}

fn setup_bevy_app(
    body: Arc<AnnyBody>,
    pose: &[f64],
    mode: BurnHumanMeshMode,
    humans: usize,
) -> (App, Vec<Entity>) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(AssetPlugin::default());
    app.insert_resource(Assets::<Mesh>::default());
    app.insert_resource(Assets::<SkinnedMeshInverseBindposes>::default());
    app.add_plugins(BurnHumanPlugin {
        source: BurnHumanSource::Preloaded(body),
        defaults: BurnHumanDefaults { render_mode: mode },
    });

    let mut entities = Vec::with_capacity(humans);
    for _ in 0..humans {
        let id = app
            .world_mut()
            .spawn(BurnHumanInput {
                pose_parameters: Some(pose.to_vec()),
                ..Default::default()
            })
            .id();
        entities.push(id);
    }

    // Warm up so the first timed update doesn't include initial mesh/rig creation.
    app.update();

    (app, entities)
}

fn update_bevy_inputs(app: &mut App, entities: &[Entity], phase: f64) {
    let dx = phase.sin() * 0.05;
    let dy = phase.cos() * 0.03;
    let world = app.world_mut();
    for &entity in entities {
        if let Some(mut input) = world.get_mut::<BurnHumanInput>(entity) {
            if let Some(pose) = input.pose_parameters.as_mut() {
                if pose.len() >= 12 {
                    pose[3] = dx;
                    pose[7] = dy;
                    pose[11] = 0.0;
                }
            }
        }
    }
}

fn animate_root(pose_parameters: &mut TensorData<f64>, phase: f64) {
    let dx = phase.sin() * 0.05;
    let dy = phase.cos() * 0.03;
    if pose_parameters.data.len() >= 12 {
        pose_parameters.data[3] = dx;
        pose_parameters.data[7] = dy;
        pose_parameters.data[11] = 0.0;
    }
}

fn baked_update(data: &BenchData, humans: usize) {
    for _ in 0..humans {
        let (bone_poses, _) = kinematics::forward_root_relative_world(
            &data.rest_bone_poses,
            &data.pose_parameters,
            &data.bone_parents,
        )
        .expect("bone poses");
        let posed_vertices = skinning::linear_blend_skinning(
            &data.rest_vertices,
            &bone_poses,
            &data.rest_bone_poses,
            &data.vertex_bone_indices,
            &data.vertex_bone_weights,
        )
        .expect("skinning");
        let positions = tensor_to_vec3(&posed_vertices);
        let indices = triangulate_quads(&data.faces_quads);
        let normals = compute_normals_from_quads(&positions, &data.faces_quads);
        black_box(indices);
        black_box(normals);
    }
}

fn skinned_update(data: &BenchData, humans: usize) {
    for _ in 0..humans {
        let (bone_poses, _) = kinematics::forward_root_relative_world(
            &data.rest_bone_poses,
            &data.pose_parameters,
            &data.bone_parents,
        )
        .expect("bone poses");
        let mats = bone_poses_to_f32(&bone_poses);
        black_box(mats);
    }
}

fn tensor_to_vec3(data: &TensorData<f64>) -> Vec<[f64; 3]> {
    match data.shape.as_slice() {
        [n, 3] => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        [b, n, 3] if *b >= 1 => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        other => panic!("expected [N,3] or [B,N,3] tensor, got shape {:?}", other),
    }
}

fn triangulate_quads(quads: &TensorData<i64>) -> Vec<u32> {
    assert_eq!(quads.shape.len(), 2, "faces tensor should be [F,4]");
    assert_eq!(quads.shape[1], 4, "faces tensor should be [F,4]");
    let mut indices = Vec::with_capacity(quads.shape[0] * 6);
    for face in quads.data.chunks_exact(4) {
        let (a, b, c, d) = (
            face[0] as u32,
            face[1] as u32,
            face[2] as u32,
            face[3] as u32,
        );
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }
    indices
}

fn compute_normals_from_quads(
    positions: &[[f64; 3]],
    quads: &TensorData<i64>,
) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f64; 3]; positions.len()];
    for face in quads.data.chunks_exact(4) {
        let a = face[0] as usize;
        let b = face[1] as usize;
        let c = face[2] as usize;
        let d = face[3] as usize;
        let pa = positions[a];
        let pb = positions[b];
        let pc = positions[c];
        let pd = positions[d];
        let n0 = cross3(sub3(pb, pa), sub3(pc, pa));
        let n1 = cross3(sub3(pc, pa), sub3(pd, pa));
        let normal = normalize3(add3(n0, n1));
        for idx in [a, b, c, d] {
            normals[idx] = add3(normals[idx], normal);
        }
    }
    normals
        .into_iter()
        .map(|n| normalize3(n))
        .map(|n| [n[0] as f32, n[1] as f32, n[2] as f32])
        .collect()
}

fn bone_poses_to_f32(bone_poses: &TensorData<f64>) -> Vec<[f32; 16]> {
    let bones = match bone_poses.shape.as_slice() {
        [_, bones, 4, 4] => *bones,
        [bones, 4, 4] => *bones,
        other => panic!("expected [B,J,4,4], got {:?}", other),
    };
    let mut out = Vec::with_capacity(bones);
    for j in 0..bones {
        let idx = j * 16;
        let mut mat = [0.0f32; 16];
        for k in 0..16 {
            mat[k] = bone_poses.data[idx + k] as f32;
        }
        out.push(mat);
    }
    out
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

fn bench_modes(c: &mut Criterion) {
    let body = load_body();
    let cases: Vec<String> = body.case_names().map(|s| s.to_string()).collect();
    if cases.is_empty() {
        return;
    }
    let _ = body
        .forward(AnnyInput::case(cases[0].as_str()))
        .expect("warmup forward");

    let mut group = c.benchmark_group("bevy_mode_costs");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    let counts = [16usize, 64];
    for &humans in &counts {
        group.throughput(Throughput::Elements(humans as u64));
        let mut baked = build_bench_data(&body);
        group.bench_with_input(BenchmarkId::new("baked", humans), &humans, |b, &humans| {
            let mut phase = 0.0f64;
            b.iter(|| {
                phase += 0.1;
                animate_root(&mut baked.pose_parameters, phase);
                baked_update(&baked, humans);
            });
        });

        let mut skinned = build_bench_data(&body);
        group.bench_with_input(BenchmarkId::new("skinned", humans), &humans, |b, &humans| {
            let mut phase = 0.0f64;
            b.iter(|| {
                phase += 0.1;
                animate_root(&mut skinned.pose_parameters, phase);
                skinned_update(&skinned, humans);
            });
        });
    }
    group.finish();
}

fn bench_modes_e2e(c: &mut Criterion) {
    let body = Arc::new(load_body());
    let bone_count = body.metadata().metadata.bone_labels.len();
    let pose = identity_pose_vec(bone_count);

    let mut group = c.benchmark_group("bevy_mode_e2e");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    let counts = [16usize, 64, 256];
    for &humans in &counts {
        group.throughput(Throughput::Elements(humans as u64));

        let (mut baked_app, baked_entities) =
            setup_bevy_app(body.clone(), &pose, BurnHumanMeshMode::BakedMesh, humans);
        group.bench_with_input(BenchmarkId::new("baked_e2e", humans), &humans, |b, &_humans| {
            let mut phase = 0.0f64;
            b.iter(|| {
                phase += 0.1;
                update_bevy_inputs(&mut baked_app, &baked_entities, phase);
                baked_app.update();
            });
        });

        let (mut skinned_app, skinned_entities) =
            setup_bevy_app(body.clone(), &pose, BurnHumanMeshMode::SkinnedMesh, humans);
        group.bench_with_input(
            BenchmarkId::new("skinned_e2e", humans),
            &humans,
            |b, &_humans| {
                let mut phase = 0.0f64;
                b.iter(|| {
                    phase += 0.1;
                    update_bevy_inputs(&mut skinned_app, &skinned_entities, phase);
                    skinned_app.update();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_modes, bench_modes_e2e);
criterion_main!(benches);
