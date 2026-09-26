//! Real Anny bind matrices: catch row/column, root-relative, axis, and twist errors.
use burn_human::{AnnyBody, AnnyInput, motion::AnnyMotionBinding};
use burn_human_motion::RigDefinition;
use glam::{Mat4, Quat, Vec3};

#[test]
fn core_motion_matches_anny_forward_kinematics() -> anyhow::Result<()> {
    let body = AnnyBody::from_reference_paths(
        "assets/model/fullbody_default.safetensors",
        "assets/model/fullbody_default.meta.json",
    )?;
    let source: RigDefinition = serde_json::from_str(include_str!("data/core27-rig.json"))?;
    // Exercise two phenotypes: offsets and bind frames must be rebuilt together.
    for value in [0.35, 0.65] {
        let phenotype = vec![value; body.metadata().metadata.phenotype_labels.len()];
        let binding = AnnyMotionBinding::new(&body, &phenotype, &source)?;
        let (source_rest, _) = source.forward(&source.rest_frame())?;
        let mut standing = source.rest_frame();
        standing.root_translation.y =
            -source_rest[source.index("LeftToeBase").unwrap()].y * binding.root_height_scale;
        let fitted = binding
            .mapping
            .retarget(&source, &binding.target, &standing)?;
        let (fitted_joints, _) = binding.target.forward(&fitted)?;
        assert!(
            fitted_joints[binding.target.index("toe3-1.L").unwrap()]
                .z
                .abs()
                < 1e-5,
            "proportion-fitted standing toe height"
        );
        let mut pose = source.rest_frame();
        pose.root_translation = Vec3::new(1.3, 0.94, -2.1);
        pose.local_rotations[0] = Quat::from_rotation_y(0.8) * Quat::from_rotation_x(0.2);
        pose.local_rotations[source.index("LeftArm").unwrap()] = Quat::from_rotation_z(0.65);
        pose.local_rotations[source.index("RightLeg").unwrap()] = Quat::from_rotation_x(-0.4);
        let mapped = binding.mapping.retarget(&source, &binding.target, &pose)?;
        let (positions, rotations) = binding.target.forward(&mapped)?;
        let (source_positions, _) = source.forward(&pose)?;
        for (a, b, s, t) in [
            ("upperarm01.L", "lowerarm01.L", "LeftArm", "LeftForeArm"),
            ("lowerarm01.L", "wrist.L", "LeftForeArm", "LeftHand"),
            ("upperleg01.R", "lowerleg01.R", "RightUpLeg", "RightLeg"),
            ("lowerleg01.R", "foot.R", "RightLeg", "RightFoot"),
        ] {
            let actual = (positions[binding.target.index(b).unwrap()]
                - positions[binding.target.index(a).unwrap()])
            .normalize();
            let expected = (binding.mapping.basis
                * (source_positions[source.index(t).unwrap()]
                    - source_positions[source.index(s).unwrap()]))
            .normalize();
            assert!(
                actual.dot(expected) > 0.9999,
                "rest-pose calibration {a}: cosine {}",
                actual.dot(expected)
            );
        }
        let parameters = binding.pose_parameters(&source, &pose)?;
        let actual = body.forward(AnnyInput {
            phenotype_inputs: Some(&phenotype),
            pose_parameters: Some(&parameters),
            ..Default::default()
        })?;
        for (i, values) in actual
            .bone_poses
            .data
            .as_chunks::<16>()
            .0
            .iter()
            .enumerate()
        {
            let matrix =
                Mat4::from_cols_array(&std::array::from_fn(|j| values[(j % 4) * 4 + j / 4] as f32));
            let (_, rotation, position) = matrix.to_scale_rotation_translation();
            assert!(
                (position - positions[i]).length() < 2e-5,
                "position {}: {position:?} != {:?}",
                binding.target.joints[i].name,
                positions[i]
            );
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                assert!(
                    (rotation * axis - rotations[i] * axis).length() < 2e-5,
                    "rotation {}",
                    binding.target.joints[i].name
                );
            }
        }
        let root = Vec3::new(
            parameters[3] as f32,
            parameters[7] as f32,
            parameters[11] as f32,
        );
        assert!((root - Vec3::new(1.3, 2.1, 0.94)).length() < 1e-6);
        assert!(actual.posed_vertices.data.iter().all(|v| v.is_finite()));
    }
    Ok(())
}
