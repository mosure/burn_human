use burn_human::data::reference::load_reference_bundle;
use burn_human::{AnnyBody, AnnyInput, AnnyReference};
use std::error::Error;

// Reference artifact is stored in mixed f16/f32; allow a looser tolerance.
const TOLERANCE: f64 = 1e-3;

fn load_model() -> Result<AnnyReference, Box<dyn Error>> {
    Ok(AnnyReference::from_paths(
        "assets/model/fullbody_default.safetensors",
        "assets/model/fullbody_default.meta.json",
    )?)
}

#[test]
fn reference_cases_roundtrip() -> Result<(), Box<dyn Error>> {
    let model = load_model()?;
    let bundle = load_reference_bundle(
        "assets/model/fullbody_default.safetensors",
        "assets/model/fullbody_default.meta.json",
    )?;

    for name in model.case_names() {
        let expected = bundle
            .cases
            .iter()
            .find(|c| c.name == name)
            .expect("missing case in bundle");
        let out = model.forward_case(name)?;

        assert_eq!(out.rest_vertices.shape, expected.rest_vertices.shape);
        assert_eq!(out.posed_vertices.shape, expected.posed_vertices.shape);
        assert_eq!(out.rest_bone_poses.shape, expected.rest_bone_poses.shape);
        assert_eq!(out.bone_poses.shape, expected.bone_poses.shape);
        assert_eq!(out.bone_heads.shape, expected.bone_heads.shape);
        assert_eq!(out.bone_tails.shape, expected.bone_tails.shape);

        let rest_err = max_abs_diff(&out.rest_vertices.data, &expected.rest_vertices.data);
        let posed_err = max_abs_diff(&out.posed_vertices.data, &expected.posed_vertices.data);
        let rest_bone_err = max_abs_diff(&out.rest_bone_poses.data, &expected.rest_bone_poses.data);
        let bone_err = max_abs_diff(&out.bone_poses.data, &expected.bone_poses.data);
        let head_err = max_abs_diff(&out.bone_heads.data, &expected.bone_heads.data);
        let tail_err = max_abs_diff(&out.bone_tails.data, &expected.bone_tails.data);

        assert!(
            rest_err < TOLERANCE,
            "rest vertices err {} for case {}",
            rest_err,
            name
        );
        assert!(
            posed_err < TOLERANCE,
            "posed vertices err {} for case {}",
            posed_err,
            name
        );
        assert!(
            rest_bone_err < TOLERANCE,
            "rest bone pose err {} for case {}",
            rest_bone_err,
            name
        );
        assert!(
            bone_err < TOLERANCE,
            "bone pose err {} for case {}",
            bone_err,
            name
        );
        assert!(
            head_err < TOLERANCE,
            "bone heads err {} for case {}",
            head_err,
            name
        );
        assert!(
            tail_err < TOLERANCE,
            "bone tails err {} for case {}",
            tail_err,
            name
        );
    }

    Ok(())
}

#[test]
fn reference_metadata_present() -> Result<(), Box<dyn Error>> {
    let model = load_model()?;
    let meta = &model.metadata().metadata;
    assert!(!meta.case_names.is_empty());
    assert!(!meta.bone_labels.is_empty());
    assert!(!meta.phenotype_labels.is_empty());
    assert!(!meta.rig.is_empty());
    assert!(!meta.topology.is_empty());
    assert!(!meta.pose_parameterization.is_empty());
    assert!(!meta.bone_parents.is_empty());
    assert!(!meta.macrodetail_keys.is_empty());
    assert!(!meta.phenotype_variations.is_empty());
    assert!(!meta.phenotype_anchors.is_empty());
    Ok(())
}

#[test]
fn anny_body_forward_returns_reference_outputs() -> Result<(), Box<dyn Error>> {
    let body = AnnyBody::from_reference_paths(
        "assets/model/fullbody_default.safetensors",
        "assets/model/fullbody_default.meta.json",
    )?;
    let bundle = load_reference_bundle(
        "assets/model/fullbody_default.safetensors",
        "assets/model/fullbody_default.meta.json",
    )?;

    for case in &bundle.cases {
        let out = body.forward(AnnyInput {
            case_name: Some(&case.name),
            ..Default::default()
        })?;
        // Compare rest vertices and posed vertices.
        assert_eq!(out.posed_vertices.shape, case.posed_vertices.shape);
        assert_eq!(out.rest_vertices.shape, case.rest_vertices.shape);
        assert_eq!(out.bone_poses.shape, case.bone_poses.shape);
        assert_eq!(out.rest_bone_poses.shape, case.rest_bone_poses.shape);

        let max_abs = max_abs_diff(&out.posed_vertices.data, &case.posed_vertices.data);
        assert!(
            max_abs < TOLERANCE,
            "posed vertices mismatch for case {} (max abs diff {})",
            case.name,
            max_abs
        );
        let max_rest = max_abs_diff(&out.rest_vertices.data, &case.rest_vertices.data);
        assert!(
            max_rest < TOLERANCE,
            "rest vertices mismatch for case {} (max abs diff {})",
            case.name,
            max_rest
        );

        let max_bone = max_abs_diff(&out.bone_poses.data, &case.bone_poses.data);
        assert!(
            max_bone < TOLERANCE,
            "bone poses mismatch for case {} (max abs diff {})",
            case.name,
            max_bone
        );
        assert_eq!(out.posed_vertices.shape, case.posed_vertices.shape);
    }
    Ok(())
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max {
            max = d;
        }
    }
    max
}
