//! Studio bindings and workflows. Widget construction lives in `studio_ui`.
use super::{
    MotionPlayback, MotionRuntime, MotionStatus,
    body::{self, BodyIdentity},
    body_view::BodyDisplay,
    camera::{View, ViewControls},
    runtime::{self, InputKind},
};
use crate::studio_ui::{ControlEvent, Forms, Row, Value};
use bevy::{
    image::Image,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use burn_human_motion::MotionRequest;
#[derive(Resource)]
pub struct MotionUi {
    pub request: MotionRequest,
    pub place_waypoints: bool,
    bundle: String,
    digest: String,
    text_bundle: String,
    text_digest: String,
    embedding: String,
    clip: String,
    pub(super) image: String,
    export_path: String,
    image_preview: Option<Handle<Image>>,
    pub(super) image_revision: u64,
    pub(super) body: super::body_ui::BodyUi,
    tab: usize,
    advanced: bool,
    generation: bool,
    pub(super) selected_waypoint: Option<usize>,
    pub(super) dragging_waypoint: bool,
}
impl Default for MotionUi {
    fn default() -> Self {
        Self {
            request: MotionRequest::default(),
            place_waypoints: false,
            bundle: burn_ardy::pretrained::DEFAULT
                .at_root(burn_ardy::pretrained::CDN_ROOT)
                .base,
            digest: burn_ardy::pretrained::DEFAULT.sha256.into(),
            text_bundle: burn_llama::pretrained::DEFAULT
                .at_root(burn_llama::pretrained::CDN_ROOT)
                .base,
            text_digest: burn_llama::pretrained::DEFAULT.sha256.into(),
            embedding: String::new(),
            clip: String::new(),
            image: String::new(),
            export_path: "motion.json".into(),
            image_preview: None,
            image_revision: 0,
            body: Default::default(),
            tab: 0,
            advanced: false,
            generation: false,
            selected_waypoint: None,
            dragging_waypoint: false,
        }
    }
}
impl MotionUi {
    pub fn is_motion_tab(&self) -> bool {
        self.tab == 0
    }
    pub fn is_manual_tab(&self) -> bool {
        self.tab == 3
    }
    pub(super) fn is_body_tab(&self) -> bool {
        matches!(self.tab, 1 | 2)
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn controls(
    mut events: MessageReader<ControlEvent>,
    mut state: ResMut<MotionUi>,
    runtime: Res<MotionRuntime>,
    mut player: ResMut<MotionPlayback>,
    mut display: ResMut<BodyDisplay>,
    mut view: ResMut<ViewControls>,
    mut forms: ResMut<Forms>,
    mut images: ResMut<Assets<Image>>,
) {
    super::body_ui::sync(&mut state.body, &runtime);
    for event in events.read() {
        if !event.id.starts_with("motion.") {
            continue;
        }
        let id = &event.id[7..];
        match &event.value {
            Value::Text(value) => match id {
                "prompt" => state.request.prompt = value.clone(),
                "bundle" => state.bundle = value.clone(),
                "digest" => state.digest = value.clone(),
                "text_bundle" => state.text_bundle = value.clone(),
                "text_digest" => state.text_digest = value.clone(),
                "embedding" => state.embedding = value.clone(),
                "clip" => state.clip = value.clone(),
                "image" => state.image = value.clone(),
                "soma_bundle" => state.body.soma_bundle = value.clone(),
                "soma_digest" => state.body.soma_digest = value.clone(),
                "gem_suite" => state.body.gem_suite = value.clone(),
                "joint_filter" => {
                    state.body.joint_filter = value.clone();
                    let names = runtime.0.lock().unwrap().body.joint_names.clone();
                    let filtered = filtered_joints(&names, value);
                    if !filtered.contains(&state.body.joint)
                        && let Some(index) = filtered.first()
                    {
                        state.body.joint = *index;
                    }
                }
                "export_path" => state.export_path = value.clone(),
                "pose_path" => state.body.export_path = value.clone(),
                "seed" => {
                    if let Ok(seed) = value.parse() {
                        state.request.seed = seed;
                    }
                }
                _ => {}
            },
            Value::Point(point) => {
                if id == "image_crop" {
                    state.body.crop.center = [
                        point.x * state.body.image_size[0] as f32,
                        point.y * state.body.image_size[1] as f32,
                    ];
                }
            }
            Value::Choice(value) => match id {
                "tab" => {
                    state.tab = *value;
                    state.place_waypoints = false;
                    display.visible = state.is_body_tab();
                    if state.is_manual_tab() {
                        player.active = false;
                        player.playing = false;
                    }
                }
                "joint" => {
                    let names = runtime.0.lock().unwrap().body.joint_names.clone();
                    if let Some(index) =
                        filtered_joints(&names, &state.body.joint_filter).get(*value)
                    {
                        state.body.joint = *index;
                    }
                }
                "component" => state.body.component = *value,
                "scale" => state.body.scale = *value,
                "play_joint" => {
                    player.joint = *value;
                    player.joint_euler = [0.; 3];
                }
                "waypoint" => state.selected_waypoint = Some(*value),
                _ => {}
            },
            Value::Toggle(value) => match id {
                "advanced" => state.advanced = *value,
                "generation" => state.generation = *value,
                "path_edit" => state.place_waypoints = *value,
                "path_show" => player.show_path = *value,
                "active" => player.active = *value,
                "playing" => player.playing = *value,
                "loop" => player.looping = *value,
                "skeleton" => player.show_skeleton = *value,
                "fit_height" => player.fit_root_height = *value,
                "body_visible" => display.visible = *value,
                "body_skeleton" => display.skeleton = *value,
                "live" => state.body.live = *value,
                "correctives" => state.body.pose.apply_correctives = *value,
                "waypoint_face" => {
                    if let Some(i) = state.selected_waypoint
                        && let Some(w) = state.request.waypoints.get_mut(i)
                    {
                        w.heading = if *value { Some(0.) } else { None };
                    }
                }
                "waypoint_height" => {
                    if let Some(i) = state.selected_waypoint
                        && let Some(w) = state.request.waypoints.get_mut(i)
                    {
                        w.constrain_height = *value;
                    }
                }
                _ => {}
            },
            Value::Number(value) => set_number(id, *value, &mut state, &mut player),
            Value::Action => {
                let busy = runtime.0.lock().unwrap().status.busy();
                match id {
                    "home" => view.requested = Some(View::Home),
                    "front" => view.requested = Some(View::Front),
                    "side" => view.requested = Some(View::Side),
                    "top" => view.requested = Some(View::Top),
                    "frame" => {
                        if state.is_body_tab() {
                            display.frame_view = true;
                        } else {
                            player.frame_view = true;
                        }
                    }
                    "walk" => state.request.prompt = "A person walks forward.".into(),
                    "run" => state.request.prompt = "A person runs forward.".into(),
                    "dance" => {
                        state.request.prompt = "A person dances with their arms moving.".into()
                    }
                    "wave" => {
                        state.request.prompt = "A person stands and waves their right hand.".into()
                    }
                    "straight" | "turn" | "path_loop" => {
                        let points: &[[f32; 2]] = match id {
                            "straight" => &[[0., 0.], [0., 3.]],
                            "turn" => &[[0., 0.], [0., 1.5], [1.5, 2.5]],
                            _ => &[[0., 0.], [-1., 1.], [0., 2.], [1., 1.], [0., 0.]],
                        };
                        state.request.waypoints.clear();
                        for [x, z] in points {
                            super::trajectory::add(&mut state, Vec3::new(*x, 0.02, *z));
                        }
                        state.selected_waypoint = None;
                        player.show_path = true;
                        player.frame_view = true;
                    }
                    "clear_path" => {
                        state.request.waypoints.clear();
                        state.selected_waypoint = None;
                    }
                    "remove_point" => {
                        if let Some(i) = state.selected_waypoint.take()
                            && i < state.request.waypoints.len()
                        {
                            state.request.waypoints.remove(i);
                            super::trajectory::retime(&mut state.request);
                        }
                    }
                    "add_point" => super::trajectory::add(&mut state, Vec3::new(0., 0.02, 0.)),
                    "restart" => player.time = 0.,
                    "reset_offset" => player.joint_euler = [0.; 3],
                    "cancel" => runtime
                        .0
                        .lock()
                        .unwrap()
                        .cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed),
                    "native_identity" => {
                        state.body.identity = BodyIdentity::Native(Default::default());
                        state.body.pose = Default::default();
                    }
                    "reset_identity" => match &mut state.body.identity {
                        BodyIdentity::Native(p) => p.coefficients.fill(0.),
                        BodyIdentity::Image(p) => p.coefficients.fill(0.),
                    },
                    "t_pose" | "relaxed" | "body_wave" => {
                        let names = runtime.0.lock().unwrap().body.joint_names.clone();
                        let preset = match id {
                            "t_pose" => "T pose",
                            "relaxed" => "Relaxed",
                            _ => "Wave",
                        };
                        state.body.pose = super::body_ui::pose_preset(
                            preset,
                            &names,
                            matches!(state.body.identity, BodyIdentity::Native(_)),
                        );
                    }
                    "reset_crop" => {
                        state.body.image_size = [0; 2];
                        super::body_ui::sync(&mut state.body, &runtime);
                    }
                    _ if busy => {}
                    "load_motion" => runtime::prepare_motion(
                        &runtime,
                        (state.bundle.clone(), state.digest.clone()),
                        (state.text_bundle.clone(), state.text_digest.clone()),
                    ),
                    "load_ardy" => {
                        runtime::load(&runtime, state.bundle.clone(), state.digest.clone())
                    }
                    "load_text" => runtime::load_text(
                        &runtime,
                        state.text_bundle.clone(),
                        state.text_digest.clone(),
                    ),
                    "encode" => runtime::encode_prompt(&runtime, state.request.prompt.clone()),
                    "generate" => {
                        let shared = runtime.0.lock().unwrap();
                        let ready = shared.model.is_some()
                            && (shared.text_model.is_some()
                                || shared
                                    .embedding
                                    .as_ref()
                                    .is_some_and(|e| e.prompt == state.request.prompt));
                        drop(shared);
                        if ready && state.request.validate().is_ok() {
                            runtime::generate(&runtime, state.request.clone());
                        }
                    }
                    "pick_embedding" => runtime::pick(&runtime, InputKind::Embedding),
                    "import_embedding" => {
                        runtime::import(&runtime, state.embedding.clone(), InputKind::Embedding)
                    }
                    "pick_clip" => runtime::pick(&runtime, InputKind::Clip),
                    "import_clip" => runtime::import(&runtime, state.clip.clone(), InputKind::Clip),
                    "export_clip" => {
                        if let Some(clip) = &player.clip {
                            runtime::export(&runtime, clip, &state.export_path);
                        }
                    }
                    "load_soma" => body::load_soma(
                        &runtime,
                        state.body.soma_bundle.clone(),
                        state.body.soma_digest.clone(),
                    ),
                    "apply_body" => {
                        super::body_ui::apply_if_changed(&mut state.body, &runtime, true)
                    }
                    "load_gem" => body::load_gem(&runtime, state.body.gem_suite.clone()),
                    "pick_image" => runtime::pick(&runtime, InputKind::Image),
                    "import_image" => {
                        runtime::import(&runtime, state.image.clone(), InputKind::Image)
                    }
                    "estimate" => {
                        let shared = runtime.0.lock().unwrap();
                        let ready = shared.body.gem.is_some() && shared.image.is_some();
                        drop(shared);
                        if ready {
                            body::estimate(&runtime, state.body.crop, state.body.camera);
                        }
                    }
                    "export_pose" => {
                        let b = &state.body;
                        let value = serde_json::json!({"schema":1,"identity":b.identity,"pose":b.pose,"camera":b.camera,"crop":b.crop});
                        runtime::export_json(&runtime, &value, &b.export_path, "pose.json");
                    }
                    _ => {}
                }
            }
        }
    }
    if state.tab == 1 {
        super::body_ui::apply_if_changed(&mut state.body, &runtime, false);
    }
    let shared = runtime.0.lock().unwrap();
    if shared.image_revision != state.image_revision {
        state.image_revision = shared.image_revision;
        state.image_preview = None;
        if let Some(image) = &shared.image {
            state.image_preview = Some(images.add(Image::new(
                Extent3d {
                    width: image.width,
                    height: image.height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                image.rgba.clone(),
                TextureFormat::Rgba8UnormSrgb,
                bevy::asset::RenderAssetUsages::default(),
            )));
        }
    }
    let busy = shared.status.busy();
    let loaded = shared.model.is_some();
    let text_loaded = shared.text_model.is_some();
    let body_loaded = !shared.body.joint_names.is_empty();
    let gem_loaded = shared.body.gem.is_some();
    let has_image = shared.image.is_some();
    let joint_names = shared.body.joint_names.clone();
    let scale_names = shared.body.scale_names.clone();
    let status = match &shared.status {
        MotionStatus::Unloaded => "Models load on demand; choose a mode to explore.".into(),
        MotionStatus::Ready => "Ready".into(),
        MotionStatus::Loading(i, n) => format!("Loading weights: {i}/{n}"),
        MotionStatus::Generating(i, n) => format!("Generating frames: {i}/{n}"),
        MotionStatus::Working(message) | MotionStatus::Failed(message) => message.clone(),
    };
    let embedded = shared
        .embedding
        .as_ref()
        .is_some_and(|e| e.prompt == state.request.prompt);
    let current_estimate =
        shared
            .body
            .estimate_input
            .as_ref()
            .is_some_and(|(rev, crop, camera)| {
                *rev == state.image_revision
                    && crop.center == state.body.crop.center
                    && crop.size == state.body.crop.size
                    && camera.focal == state.body.camera.focal
                    && camera.center == state.body.camera.center
            });
    let keypoints = if current_estimate {
        shared
            .body
            .estimate
            .as_ref()
            .map(|e| e.keypoints_2d.clone())
            .unwrap_or_default()
    } else {
        vec![]
    };
    drop(shared);
    let mut rows = vec![
        Row::Label("burn_human studio".into()),
        Row::select(
            "motion.tab",
            "Explore",
            [
                "Motion · ARDY",
                "SOMA controls",
                "Image pose · GEM-X",
                "Anny body & pose",
            ]
            .map(str::to_owned)
            .to_vec(),
            state.tab,
        ),
    ];
    rows.push(buttons(
        &[
            ("home", "Home"),
            ("front", "Front"),
            ("side", "Side"),
            ("top", "Top"),
        ],
        true,
    ));
    rows.push(button("frame", "Frame subject / path (F)", true));
    rows.push(Row::Label(if state.place_waypoints{"Left: add / drag point · Middle: orbit · Right: pan\nScroll: zoom · Esc: finish path · Delete: remove point"}else{"Left: orbit · Right: pan · Scroll: zoom\nF: frame subject / path · Home: reset view"}.into()));
    rows.push(Row::Label(status));
    match state.tab {
        0 => {
            rows.push(Row::Label(
                "Describe an action, optionally sketch its path, then generate and play.".into(),
            ));
            rows.push(button(
                "load_motion",
                if loaded && text_loaded {
                    "Reload motion models"
                } else {
                    "Load motion models"
                },
                !busy,
            ));
            rows.push(Row::Text {
                id: "motion.prompt".into(),
                label: "Describe the motion".into(),
                value: state.request.prompt.clone(),
                multiline: true,
            });
            rows.push(buttons(
                &[
                    ("walk", "Walk"),
                    ("run", "Run"),
                    ("dance", "Dance"),
                    ("wave", "Wave"),
                ],
                true,
            ));
            rows.push(button("encode", "Encode prompt", !busy && text_loaded));
            rows.push(slider(
                "duration",
                "Duration (seconds)",
                state.request.frames as f32 / 20.,
                2.,
                600.,
                0.2,
            ));
            if let Err(e) = state.request.validate() {
                rows.push(Row::Label(e.to_string()));
            }
            rows.push(button(
                "generate",
                "Generate motion",
                !busy && loaded && (embedded || text_loaded) && state.request.validate().is_ok(),
            ));
            rows.push(button(
                "cancel",
                "Cancel generation",
                matches!(
                    runtime.0.lock().unwrap().status,
                    MotionStatus::Generating(..)
                ),
            ));
            playback_rows(&mut rows, &state, &player, busy);
            trajectory_rows(&mut rows, &state, &player);
            rows.push(toggle(
                "generation",
                "Generation settings",
                state.generation,
            ));
            if state.generation {
                rows.push(text(
                    "seed",
                    "Seed (integer)",
                    &state.request.seed.to_string(),
                ));
                rows.push(slider(
                    "steps",
                    "Diffusion steps",
                    state.request.diffusion_steps as f32,
                    1.,
                    10.,
                    1.,
                ));
                rows.push(slider(
                    "text_guidance",
                    "Text guidance",
                    state.request.text_guidance,
                    0.,
                    5.,
                    0.1,
                ));
                rows.push(slider(
                    "path_guidance",
                    "Path guidance",
                    state.request.trajectory_guidance,
                    0.,
                    5.,
                    0.1,
                ));
                rows.push(slider(
                    "history",
                    "History frames",
                    state.request.history_frames as f32,
                    0.,
                    160.,
                    4.,
                ));
            }
            rows.push(toggle(
                "advanced",
                "Sources, imports and embeddings",
                state.advanced,
            ));
            if state.advanced {
                rows.push(Row::Label("Import motion / SOMA animation".into()));
                rows.push(button("pick_clip", "Choose motion JSON…", !busy));
                rows.push(text("clip", "Clip JSON URL / native path", &state.clip));
                rows.push(button(
                    "import_clip",
                    "Import motion",
                    !busy && !state.clip.is_empty(),
                ));
                rows.push(Row::Label("Advanced model sources".into()));
                for (id, title, value) in [
                    ("bundle", "ARDY bundle URL / directory", &state.bundle),
                    ("digest", "ARDY manifest SHA-256", &state.digest),
                    (
                        "text_bundle",
                        "Llama bundle URL / directory",
                        &state.text_bundle,
                    ),
                    ("text_digest", "Llama manifest SHA-256", &state.text_digest),
                    ("embedding", "Embedding JSON URL / file", &state.embedding),
                ] {
                    rows.push(text(id, title, value));
                }
                rows.push(button("load_ardy", "Load ARDY", !busy));
                rows.push(button("load_text", "Load Llama", !busy));
                rows.push(button("pick_embedding", "Choose embedding JSON…", !busy));
                rows.push(button(
                    "import_embedding",
                    "Import embedding",
                    !busy && !state.embedding.is_empty(),
                ));
            }
        }
        1 => body_rows(
            &mut rows,
            &state,
            &display,
            &joint_names,
            &scale_names,
            body_loaded,
            busy,
        ),
        2 => {
            rows.push(Row::Label(
                "Choose one person, center the crop, and estimate locally.".into(),
            ));
            rows.push(button(
                "load_gem",
                if gem_loaded {
                    "Reload image pose models"
                } else {
                    "Load image pose models"
                },
                !busy,
            ));
            rows.push(button("pick_image", "Choose image…", !busy));
            rows.push(text("image", "Image URL / native path", &state.image));
            rows.push(button(
                "import_image",
                "Load image",
                !busy && !state.image.is_empty(),
            ));
            rows.push(Row::Label("PNG / JPEG · up to 8 MiB, 4096 × 4096 pixels. Drag the image to center the person crop.".into()));
            if let Some(image) = &state.image_preview {
                let [w, h] = state.body.image_size.map(|n| n.max(1) as f32);
                let crop = state.body.crop;
                rows.push(Row::Image {
                    id: "motion.image_crop".into(),
                    image: image.clone(),
                    aspect: w / h,
                    crop: [
                        (crop.center[0] - crop.size / 2.) / w,
                        (crop.center[1] - crop.size / 2.) / h,
                        crop.size / w,
                        crop.size / h,
                    ],
                    points: keypoints
                        .iter()
                        .filter(|p| p[2] > 0.1)
                        .map(|p| [p[0] / w, p[1] / h])
                        .collect(),
                });
            }
            rows.push(slider(
                "crop_x",
                "Crop center X (pixels)",
                state.body.crop.center[0],
                0.,
                state.body.image_size[0].max(1) as f32,
                1.,
            ));
            rows.push(slider(
                "crop_y",
                "Crop center Y (pixels)",
                state.body.crop.center[1],
                0.,
                state.body.image_size[1].max(1) as f32,
                1.,
            ));
            rows.push(slider(
                "crop_size",
                "Crop size (pixels)",
                state.body.crop.size,
                16.,
                4096.,
                1.,
            ));
            rows.push(slider(
                "focal",
                "Camera focal length (pixels)",
                state.body.camera.focal[0],
                10.,
                20000.,
                5.,
            ));
            rows.push(button("reset_crop", "Reset crop and camera", has_image));
            rows.push(button(
                "estimate",
                "Estimate pose",
                !busy && gem_loaded && has_image,
            ));
            if has_image && !current_estimate {
                rows.push(Row::Label(
                    "Estimate again after changing the image, crop or camera.".into(),
                ));
            }
            rows.push(toggle(
                "body_visible",
                "Show estimated body",
                display.visible,
            ));
            rows.push(toggle("body_skeleton", "Show skeleton", display.skeleton));
            rows.push(button("frame", "Frame estimated body", true));
            rows.push(Row::Label(
                "Open SOMA controls to adjust inferred identity and joints.".into(),
            ));
            rows.push(text(
                "pose_path",
                "Pose export filename",
                &state.body.export_path,
            ));
            rows.push(button(
                "export_pose",
                "Export pose and identity",
                !busy && current_estimate,
            ));
            rows.push(toggle("advanced", "Advanced model source", state.advanced));
            if state.advanced {
                rows.push(text(
                    "gem_suite",
                    "Model suite JSON URL / file",
                    &state.body.gem_suite,
                ));
            }
        }
        _ => rows.push(Row::Label(
            "Shape the bundled body, pose joints, or try procedural motion below.".into(),
        )),
    }
    forms.0.insert("motion", rows);
}
fn buttons(items: &[(&str, &str)], enabled: bool) -> Row {
    Row::Buttons(
        items
            .iter()
            .map(|(id, label)| (format!("motion.{id}"), (*label).into(), enabled))
            .collect(),
    )
}
fn button(id: &str, title: &str, enabled: bool) -> Row {
    Row::button(&format!("motion.{id}"), title, enabled)
}
fn text(id: &str, title: &str, value: &str) -> Row {
    Row::text(&format!("motion.{id}"), title, value)
}
fn slider(id: &str, title: &str, value: f32, min: f32, max: f32, step: f32) -> Row {
    Row::slider(&format!("motion.{id}"), title, value, min, max, step)
}
fn toggle(id: &str, title: &str, value: bool) -> Row {
    Row::toggle(&format!("motion.{id}"), title, value)
}
fn select(id: &str, title: &str, options: Vec<String>, value: usize) -> Row {
    Row::select(&format!("motion.{id}"), title, options, value)
}
fn playback_rows(rows: &mut Vec<Row>, state: &MotionUi, player: &MotionPlayback, busy: bool) {
    let Some(clip) = &player.clip else {
        return;
    };
    rows.push(Row::Label("Playback".into()));
    for (id, title, value) in [
        ("active", "Animate Anny from motion", player.active),
        ("playing", "Play", player.playing),
        ("loop", "Loop", player.looping),
        ("skeleton", "Show source rig", player.show_skeleton),
        (
            "fit_height",
            "Fit root height to Anny",
            player.fit_root_height,
        ),
    ] {
        rows.push(toggle(id, title, value));
    }
    rows.push(button("restart", "Restart", true));
    rows.push(slider(
        "time",
        "Playback time (seconds)",
        player.time,
        0.,
        clip.duration(),
        0.05,
    ));
    rows.push(slider(
        "speed",
        "Playback speed",
        player.speed,
        0.1,
        2.,
        0.1,
    ));
    rows.push(select(
        "play_joint",
        "Motion joint offset",
        clip.rig.joints.iter().map(|j| j.name.clone()).collect(),
        player.joint,
    ));
    for (i, title) in [
        "Offset X (degrees)",
        "Offset Y (degrees)",
        "Offset Z (degrees)",
    ]
    .iter()
    .enumerate()
    {
        rows.push(slider(
            &format!("offset_{i}"),
            title,
            player.joint_euler[i],
            -90.,
            90.,
            1.,
        ));
    }
    rows.push(button("reset_offset", "Reset joint offset", true));
    rows.push(text(
        "export_path",
        "Clip export filename",
        &state.export_path,
    ));
    rows.push(button("export_clip", "Export clip JSON", !busy));
}
fn trajectory_rows(rows: &mut Vec<Row>, state: &MotionUi, player: &MotionPlayback) {
    rows.push(Row::Label(
        "Trajectory · optional · gold: requested, blue: generated".into(),
    ));
    rows.push(buttons(
        &[
            ("straight", "Straight"),
            ("turn", "Turn"),
            ("path_loop", "Loop"),
        ],
        true,
    ));
    rows.push(buttons(
        &[("add_point", "Add point"), ("clear_path", "Clear path")],
        true,
    ));
    rows.push(toggle(
        "path_edit",
        "Edit path in scene",
        state.place_waypoints,
    ));
    rows.push(toggle("path_show", "Show paths", player.show_path));
    if !state.request.waypoints.is_empty() {
        rows.push(select(
            "waypoint",
            "Selected waypoint",
            state
                .request
                .waypoints
                .iter()
                .enumerate()
                .map(|(i, _)| format!("Point {}", i + 1))
                .collect(),
            state.selected_waypoint.unwrap_or(0),
        ));
    }
    if let Some(i) = state.selected_waypoint
        && let Some(w) = state.request.waypoints.get(i)
    {
        rows.push(slider(
            "waypoint_frame",
            "Frame",
            w.frame as f32,
            0.,
            (state.request.frames - 1) as f32,
            1.,
        ));
        for (axis, title) in ["World X (meters)", "World Y (meters)", "World Z (meters)"]
            .iter()
            .enumerate()
        {
            rows.push(slider(
                &format!("waypoint_{axis}"),
                title,
                w.position[axis],
                -100.,
                100.,
                0.05,
            ));
        }
        rows.push(toggle(
            "waypoint_height",
            "Constrain height",
            w.constrain_height,
        ));
        rows.push(toggle(
            "waypoint_face",
            "Constrain heading",
            w.heading.is_some(),
        ));
        if let Some(heading) = w.heading {
            rows.push(slider(
                "waypoint_heading",
                "Heading (degrees)",
                heading.to_degrees(),
                -180.,
                180.,
                1.,
            ));
        }
        rows.push(button("remove_point", "Remove selected point", true));
    }
}
fn body_rows(
    rows: &mut Vec<Row>,
    state: &MotionUi,
    display: &BodyDisplay,
    names: &[String],
    scales: &[String],
    loaded: bool,
    busy: bool,
) {
    rows.push(Row::Label(
        "SOMA identity, bone proportions, local joint controls and correctives".into(),
    ));
    rows.push(button("load_soma", "Load SOMA body", !busy));
    rows.push(toggle("advanced", "Advanced model source", state.advanced));
    if state.advanced {
        rows.push(text(
            "soma_bundle",
            "Advanced: SOMA bundle URL / directory",
            &state.body.soma_bundle,
        ));
        rows.push(text(
            "soma_digest",
            "Manifest SHA-256",
            &state.body.soma_digest,
        ));
    }
    if !loaded {
        return;
    }
    rows.push(toggle("body_visible", "Show body", display.visible));
    rows.push(toggle("body_skeleton", "Show skeleton", display.skeleton));
    rows.push(button("frame", "Frame body", true));
    let (coeff, scales_values, global, native) = match &state.body.identity {
        BodyIdentity::Native(p) => (&p.coefficients, &p.bone_scales, p.global_scale, true),
        BodyIdentity::Image(p) => (&p.coefficients, &p.scales, p.global_scale, false),
    };
    if !native {
        rows.push(Row::Label("Identity estimated by GEM-X (MHR)".into()));
        rows.push(button(
            "native_identity",
            "Start a native SOMA identity",
            true,
        ));
    }
    rows.push(buttons(
        &[
            ("t_pose", "T pose"),
            ("relaxed", "Relaxed"),
            ("body_wave", "Wave"),
        ],
        true,
    ));
    rows.push(toggle("live", "Live preview", state.body.live));
    rows.push(slider("body_scale", "Body scale", global, 0.5, 1.5, 0.01));
    rows.push(select(
        "component",
        "Identity component",
        (0..coeff.len()).map(|i| format!("Component {i}")).collect(),
        state.body.component,
    ));
    if let Some(value) = coeff.get(state.body.component) {
        rows.push(slider(
            "coefficient",
            "Identity weight",
            *value,
            -3.,
            3.,
            0.05,
        ));
    }
    rows.push(button("reset_identity", "Reset identity weights", true));
    rows.push(select(
        "scale",
        "Bone proportion",
        (0..scales_values.len())
            .map(|i| {
                if native {
                    scales
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("Scale {i}"))
                } else {
                    format!("MHR scale {i}")
                }
            })
            .collect(),
        state.body.scale,
    ));
    if let Some(value) = scales_values.get(state.body.scale) {
        rows.push(slider(
            "bone_scale",
            if native {
                "Length ratio"
            } else {
                "Scale offset"
            },
            *value,
            if native { 0.5 } else { -0.5 },
            if native { 1.5 } else { 0.5 },
            0.01,
        ));
    }
    rows.push(text(
        "joint_filter",
        "Find a joint: hand, head, leg…",
        &state.body.joint_filter,
    ));
    let filtered = filtered_joints(names, &state.body.joint_filter);
    if filtered.is_empty() {
        rows.push(Row::Label(
            "No matching joints. Clear the filter to see all joints.".into(),
        ));
    } else {
        rows.push(select(
            "joint",
            "Local joint",
            filtered.iter().map(|i| names[*i].clone()).collect(),
            filtered
                .iter()
                .position(|i| *i == state.body.joint)
                .unwrap_or(0),
        ));
    }
    if let Some(rotation) = state.body.pose.rotations.get(state.body.joint) {
        for (i, title) in [
            "Axis-angle X (degrees)",
            "Axis-angle Y (degrees)",
            "Axis-angle Z (degrees)",
        ]
        .iter()
        .enumerate()
        {
            rows.push(slider(
                &format!("rotation_{i}"),
                title,
                rotation[i].to_degrees(),
                -180.,
                180.,
                1.,
            ));
        }
    }
    rows.push(Row::Label(
        "Rotations are local rotation-vector components, in SOMA's joint convention.".into(),
    ));
    if native {
        rows.push(toggle(
            "correctives",
            "Pose correctives",
            state.body.pose.apply_correctives,
        ));
    }
    rows.push(button("apply_body", "Apply body controls", !busy));
    rows.push(text(
        "pose_path",
        "Pose export filename",
        &state.body.export_path,
    ));
    rows.push(button("export_pose", "Export pose and identity", !busy));
}
fn set_number(id: &str, value: f32, state: &mut MotionUi, player: &mut MotionPlayback) {
    if !value.is_finite() {
        return;
    }
    match id {
        "duration" => {
            state.request.frames = ((value * 5.).round() as usize * 4).clamp(40, 12000);
            super::trajectory::retime(&mut state.request);
        }
        "steps" => state.request.diffusion_steps = value.round() as usize,
        "text_guidance" => state.request.text_guidance = value,
        "path_guidance" => state.request.trajectory_guidance = value,
        "history" => state.request.history_frames = (value as usize / 4) * 4,
        "time" => {
            player.time = value;
            player.playing = false;
        }
        "speed" => player.speed = value,
        "crop_x" => state.body.crop.center[0] = value,
        "crop_y" => state.body.crop.center[1] = value,
        "crop_size" => state.body.crop.size = value,
        "focal" => state.body.camera.focal = [value; 2],
        "body_scale" => match &mut state.body.identity {
            BodyIdentity::Native(p) => p.global_scale = value,
            BodyIdentity::Image(p) => p.global_scale = value,
        },
        "coefficient" => {
            let index = state.body.component;
            let weights = match &mut state.body.identity {
                BodyIdentity::Native(p) => &mut p.coefficients,
                BodyIdentity::Image(p) => &mut p.coefficients,
            };
            if let Some(slot) = weights.get_mut(index) {
                *slot = value;
            }
        }
        "bone_scale" => {
            let index = state.body.scale;
            let weights = match &mut state.body.identity {
                BodyIdentity::Native(p) => &mut p.bone_scales,
                BodyIdentity::Image(p) => &mut p.scales,
            };
            if let Some(slot) = weights.get_mut(index) {
                *slot = value;
            }
        }
        _ if id.starts_with("rotation_") => {
            if let Ok(i) = id[9..].parse::<usize>()
                && i < 3
            {
                let joint = state.body.joint;
                if let Some(rotation) = state.body.pose.rotations.get_mut(joint) {
                    rotation[i] = value.to_radians();
                }
            }
        }
        _ if id.starts_with("offset_") => {
            if let Ok(i) = id[7..].parse::<usize>()
                && i < 3
            {
                player.joint_euler[i] = value;
            }
        }
        _ if id.starts_with("waypoint_") => {
            if let Some(i) = state.selected_waypoint
                && let Some(w) = state.request.waypoints.get_mut(i)
            {
                match &id[9..] {
                    "frame" => w.frame = value as usize,
                    "heading" => w.heading = Some(value.to_radians()),
                    axis => {
                        if let Ok(axis) = axis.parse::<usize>()
                            && axis < 3
                        {
                            w.position[axis] = value;
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn filtered_joints(names: &[String], filter: &str) -> Vec<usize> {
    let filter = filter.trim().to_lowercase();
    names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| name.to_lowercase().contains(&filter).then_some(i))
        .collect()
}
