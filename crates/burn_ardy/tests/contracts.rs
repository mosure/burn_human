use burn_ardy::{
    config::{ArdyConfig, Stats, expected_tensors},
    representation::trajectory_conditions,
    sampler::{NormalNoise, schedule},
};
use burn_human_motion::{MotionRequest, RigDefinition, Waypoint};
use glam::Vec3;

fn config() -> ArdyConfig {
    ArdyConfig {
        architecture: "ardy-core-rp-v1".into(),
        fps: 20,
        horizon: 40,
        frames_per_token: 4,
        motion_stats: Stats {
            mean: vec![0.0; 334],
            std: vec![1.0; 334],
        },
        latent_stats: Stats {
            mean: vec![0.0; 128],
            std: vec![1.0; 128],
        },
        skeleton: RigDefinition {
            id: "unused".into(),
            joints: vec![],
        },
    }
}

#[test]
fn ddim_schedule_reference() {
    let full = schedule(10).unwrap();
    assert_eq!(full[0].0, 0);
    assert!((full[0].1 - 0.97209275).abs() < 1e-7);
    assert_eq!(full[0].2, 1.0);
    assert!(full[9].1 < 0.000025);
    assert_eq!(
        schedule(4).unwrap().iter().map(|x| x.0).collect::<Vec<_>>(),
        vec![0, 3, 6, 9]
    );
    assert!(schedule(0).is_err());
    assert!(schedule(11).is_err());
}
#[test]
fn trajectory_wrap_and_sparse_mask_correctness() {
    let mut request = MotionRequest {
        frames: 40,
        ..Default::default()
    };
    request.waypoints = vec![
        Waypoint {
            frame: 4,
            position: Vec3::ZERO,
            heading: Some(179f32.to_radians()),
            constrain_height: false,
        },
        Waypoint {
            frame: 20,
            position: Vec3::new(2.0, 1.0, 4.0),
            heading: Some(-179f32.to_radians()),
            constrain_height: false,
        },
    ];
    let cfg = config();
    let (obs, mask) = trajectory_conditions(&cfg, &request).unwrap();
    let denorm = cfg.motion_stats.scale(0);
    assert!((obs[12 * 330] * denorm - 1.0).abs() < 1e-6);
    assert!((obs[12 * 330 + 2] * denorm - 2.0).abs() < 1e-6);
    assert!(obs[12 * 330 + 3] < -0.999);
    assert_eq!(mask[12 * 330 + 1], 0.0);
    assert_eq!(mask[3 * 330], 0.0);
    assert_eq!(mask[21 * 330], 0.0);
    request.dense_trajectory = false;
    let (_, mask) = trajectory_conditions(&cfg, &request).unwrap();
    assert_eq!(mask[12 * 330], 0.0);
    assert_eq!(mask[4 * 330], 1.0);
}
#[test]
fn full_checkpoint_inventory_correctness() {
    let inventory = expected_tensors();
    assert_eq!(inventory.len(), 428);
    assert_eq!(
        inventory["global_root_hybrid_constraints_proj.weight"],
        vec![1024, 2768]
    );
    assert_eq!(inventory["decoder.output_proj.weight"], vec![1316, 512]);
}
#[test]
fn reproducible_noise_distribution_correctness() {
    let mut a = NormalNoise::new(42);
    let mut b = NormalNoise::new(42);
    let samples: Vec<_> = (0..10000)
        .map(|_| {
            let x = a.sample();
            assert_eq!(x, b.sample());
            x
        })
        .collect();
    let mean = samples.iter().sum::<f32>() / 10000.0;
    let second = samples.iter().map(|v| v * v).sum::<f32>() / 10000.0;
    assert!(mean.abs() < 0.04 && (second - 1.0).abs() < 0.06);
}
