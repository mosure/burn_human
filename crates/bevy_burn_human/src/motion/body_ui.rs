use super::{
    MotionRuntime,
    body::{self, BodyIdentity},
};
use burn_gemx::camera::{Camera, Crop};
use burn_soma::{IdentityParameters, SomaPose};

pub(super) struct BodyUi {
    pub soma_bundle: String,
    pub soma_digest: String,
    pub gem_suite: String,
    pub identity: BodyIdentity,
    pub pose: SomaPose,
    pub joint: usize,
    pub component: usize,
    pub scale: usize,
    pub revision: u64,
    pub crop: Crop,
    pub camera: Camera,
    pub image_size: [u32; 2],
    pub export_path: String,
    pub live: bool,
    submitted: String,
    pub joint_filter: String,
}
impl Default for BodyUi {
    fn default() -> Self {
        Self {
            soma_bundle: burn_soma::pretrained::DEFAULT
                .at_root(burn_soma::pretrained::CDN_ROOT)
                .base,
            soma_digest: burn_soma::pretrained::DEFAULT.sha256.into(),
            gem_suite: burn_gemx::pretrained::SUITE_URL.into(),
            identity: BodyIdentity::Native(IdentityParameters::default()),
            pose: SomaPose::default(),
            joint: 0,
            component: 0,
            scale: 0,
            revision: 0,
            crop: Crop {
                center: [0.0; 2],
                size: 512.0,
            },
            camera: Camera {
                focal: [512.0; 2],
                center: [256.0; 2],
            },
            image_size: [0; 2],
            export_path: "pose.json".into(),
            live: true,
            submitted: String::new(),
            joint_filter: String::new(),
        }
    }
}

pub(super) fn sync(state: &mut BodyUi, runtime: &MotionRuntime) {
    let shared = runtime.0.lock().unwrap();
    if shared.body.revision != state.revision {
        state.revision = shared.body.revision;
        if let Some(pose) = &shared.body.estimate {
            state.identity = BodyIdentity::Image(pose.identity.clone());
            state.pose = pose.pose.clone();
        } else {
            state.identity = BodyIdentity::Native(IdentityParameters::default());
            state.pose = SomaPose::default();
        }
        state.joint = 0;
        state.component = 0;
        state.scale = 0;
        state.submitted = controls_key(state);
    }
    if let Some(image) = &shared.image {
        let dims = [image.width, image.height];
        if dims != state.image_size {
            state.image_size = dims;
            state.crop = Crop {
                center: [dims[0] as f32 / 2.0, dims[1] as f32 / 2.0],
                size: dims[1] as f32,
            };
            let focal = (dims[0] as f32).hypot(dims[1] as f32);
            state.camera = Camera {
                focal: [focal; 2],
                center: state.crop.center,
            };
        }
    }
}
fn controls_key(state: &BodyUi) -> String {
    serde_json::to_string(&(&state.identity, &state.pose)).unwrap_or_default()
}

pub(super) fn pose_preset(preset: &str, names: &[String], correctives: bool) -> SomaPose {
    let mut pose = SomaPose {
        apply_correctives: correctives,
        ..Default::default()
    };
    let rotations: &[(&str, f32)] = match preset {
        "Relaxed" => &[("LeftArm", -65.0), ("RightArm", 65.0)],
        "Wave" => &[
            ("LeftArm", -65.0),
            ("RightArm", -25.0),
            ("RightForeArm", -65.0),
        ],
        _ => &[],
    };
    for (name, degrees) in rotations {
        if let Some(i) = names.iter().position(|n| n == name) {
            pose.rotations[i][2] = degrees.to_radians();
        }
    }
    pose
}

pub(super) fn apply_if_changed(state: &mut BodyUi, runtime: &MotionRuntime, force: bool) {
    let shared = runtime.0.lock().unwrap();
    let ready = !shared.status.busy() && !shared.body.joint_names.is_empty();
    drop(shared);
    let key = controls_key(state);
    if ready && (force || state.live && state.submitted != key) {
        state.submitted = key;
        body::apply(runtime, state.identity.clone(), state.pose.clone());
    }
}
