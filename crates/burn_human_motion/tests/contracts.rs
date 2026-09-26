use burn_human_motion::{artifacts::*, rig::rotation_from_6d, *};
use glam::{Quat, Vec3};

fn rig() -> RigDefinition {
    RigDefinition {
        id: "test".into(),
        joints: vec![
            RigJoint {
                name: "root".into(),
                parent: None,
                offset: Vec3::ZERO,
                bind_rotation: Quat::IDENTITY,
            },
            RigJoint {
                name: "child".into(),
                parent: Some(0),
                offset: Vec3::Y,
                bind_rotation: Quat::IDENTITY,
            },
        ],
    }
}

#[test]
fn hierarchy_and_rotation_validation_correctness() {
    let mut r = rig();
    r.validate().unwrap();
    r.joints[1].parent = Some(1);
    assert!(r.validate().is_err());
    r.joints[1].parent = Some(0);
    r.joints[1].name = "root".into();
    assert!(r.validate().is_err());
    assert!(rotation_from_6d([0.0; 6]).is_err());
    assert!(rotation_from_6d([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]).is_err());
    let q = rotation_from_6d([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]).unwrap();
    assert!((q * Vec3::X - Vec3::Y).length() < 1e-6);
}

#[test]
fn bind_aware_retarget_correctness() {
    let source = rig();
    let mut target = rig();
    target.joints[0].bind_rotation = Quat::from_rotation_x(0.7);
    target.joints[1].bind_rotation = Quat::from_rotation_z(-0.4);
    let mapping = RigMapping {
        source_for_target: vec![Some(0), None],
        root_scale: 2.0,
        basis: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
    };
    let mut frame = source.rest_frame();
    frame.root_translation = Vec3::Y;
    frame.local_rotations[0] = Quat::from_rotation_y(0.5);
    let mapped = mapping.retarget(&source, &target, &frame).unwrap();
    assert!((mapped.root_translation - Vec3::Z * 2.0).length() < 1e-6);
    assert!(mapped.local_rotations[1].angle_between(target.joints[1].bind_rotation) < 1e-5);
    let (_, global) = target.forward(&mapped).unwrap();
    let expected = mapping.basis
        * frame.local_rotations[0]
        * mapping.basis.inverse()
        * target.joints[0].bind_rotation;
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        assert!((global[0] * axis - expected * axis).length() < 1e-6);
    }
}

#[test]
fn soma_joint_orient_reference() {
    let mut r = rig();
    r.id = "soma-77".into();
    let a = Quat::from_rotation_z(0.7);
    let b = Quat::from_rotation_x(0.4);
    let relative = Vec3::Y * 0.5;
    let animation = soma::SomaAnimation {
        rig: r,
        fps: 30.0,
        identity_model_type: "soma".into(),
        identity_coeffs: vec![0.0; 128],
        scale_params: vec![],
        global_scale: None,
        poses: vec![vec![Vec3::ZERO, relative]],
        translations: vec![Vec3::new(0.0, 100.0, 0.0)],
        absolute_pose: false,
        joint_orient: Some(vec![a, b]),
        unit: "cm".into(),
        source_revision: "fixture".into(),
    };
    let clip = animation.into_clip().unwrap();
    let expected = a.inverse() * Quat::from_scaled_axis(relative) * b;
    assert!(clip.frames[0].local_rotations[1].angle_between(expected) < 1e-5);
    assert_eq!(clip.frames[0].root_translation, Vec3::Y);
    assert!((clip.rig.joints[1].offset.y - 0.01).abs() < 1e-7);
}

fn manifest() -> Manifest {
    let bytes = vec![0u8; 4];
    let part = Part {
        sha256: sha256(&bytes),
        size: bytes.len(),
    };
    let mut m = Manifest {
        schema_version: 1,
        model: "fixture".into(),
        model_revision: "a".repeat(40),
        source_revision: "b".repeat(40),
        converter: "test".into(),
        license: "test".into(),
        config: serde_json::json!({}),
        objects: vec![Object {
            stage: "one".into(),
            sha256: part.sha256.clone(),
            size: 4,
            parts: vec![part.clone()],
            tensors: vec![TensorSpec {
                name: "one.weight".into(),
                shape: vec![1],
                dtype: "f32".into(),
                sha256: part.sha256,
            }],
        }],
        assets: Vec::new(),
        content_sha256: String::new(),
    };
    m.seal().unwrap();
    m
}

#[test]
fn sealed_inventory_integrity_correctness() {
    let good = manifest();
    good.validate().unwrap();
    let mut precise = good.clone();
    precise.config = serde_json::json!({"scale":0.0000018230751162020558,"negative_zero":-0.0});
    precise.seal().unwrap();
    Manifest::from_bytes(
        &serde_json::to_vec_pretty(&precise).unwrap(),
        Some(&precise.content_sha256),
    )
    .unwrap();
    let mut m = good.clone();
    m.config = serde_json::json!({"changed":true});
    assert!(m.validate().is_err());
    let mut m = good.clone();
    m.objects[0].parts[0].sha256 = "../unsafe".into();
    assert!(m.seal().is_err());
    let mut m = good.clone();
    m.objects[0].size = MAX_OBJECT_BYTES + 1;
    assert!(m.seal().is_err());
    let mut m = good.clone();
    let duplicate = m.objects[0].tensors[0].clone();
    m.objects[0].tensors.push(duplicate);
    assert!(m.seal().is_err());
    let mut m = good.clone();
    m.objects[0].tensors[0].shape = vec![usize::MAX, 4];
    assert!(m.seal().is_err());
    assert!(good.objects[0].parts[0].verify(&[0, 0, 0, 1]).is_err());
    assert!(good.objects[0].parts[0].verify(&[0, 0, 0]).is_err());
    assert!(
        Manifest::from_bytes(&serde_json::to_vec(&good).unwrap(), Some(&"0".repeat(64))).is_err()
    );
}

#[test]
fn request_and_camera_validation_correctness() {
    let mut request = MotionRequest::default();
    request.validate().unwrap();
    request.diffusion_steps = 0;
    assert!(request.validate().is_err());
    request.diffusion_steps = 10;
    request.frames = 41;
    assert!(request.validate().is_err());
    request.frames = 40;
    request.waypoints = vec![
        Waypoint {
            frame: 4,
            position: Vec3::ZERO,
            heading: None,
            constrain_height: false
        };
        2
    ];
    assert!(request.validate().is_err());
    let image = ImageCondition {
        width: 2,
        height: 1,
        rgba: vec![0; 8],
        crop_xywh: Some([u32::MAX, 0, 2, 1]),
        focal_length_px: None,
    };
    assert!(image.validate().is_err());
}

#[test]
fn metadata_and_packed_precision_are_authenticated() {
    let payload = b"{\"vocab\":42}";
    let asset = Asset {
        path: "metadata/tokenizer.json".into(),
        size: payload.len(),
        sha256: sha256(payload),
    };
    let mut m = manifest();
    m.schema_version = 2;
    m.assets.push(asset.clone());
    m.seal().unwrap();
    Manifest::from_bytes(&serde_json::to_vec(&m).unwrap(), Some(&m.content_sha256)).unwrap();
    asset.verify(payload).unwrap();
    assert!(asset.verify(b"{\"vocab\":43}").is_err());
    let mut bad = m.clone();
    bad.assets[0].path = "metadata/../weights".into();
    assert!(bad.seal().is_err());
    let mut bad = m.clone();
    bad.assets.push(asset);
    assert!(bad.seal().is_err());
    let mut bad = m.clone();
    bad.schema_version = 1;
    assert!(bad.seal().is_err());
    let mut spec = m.objects[0].tensors[0].clone();
    spec.shape = vec![2, 32];
    spec.dtype = "q4f32".into();
    assert_eq!(spec.byte_len(), Some(40));
    spec.shape = vec![2, 31];
    assert_eq!(spec.byte_len(), None);
    spec.shape = vec![2, 32];
    spec.dtype = "f16".into();
    assert_eq!(spec.byte_len(), Some(128));
    spec.dtype = "unknown".into();
    assert_eq!(spec.byte_len(), None);
    let legacy = manifest();
    let json = serde_json::to_value(&legacy).unwrap();
    assert!(json.get("assets").is_none(), "Do not change schema-1 seals");
}
