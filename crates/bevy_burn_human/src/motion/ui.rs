use super::{
    MotionPlayback, MotionRuntime, MotionStatus,
    runtime::{self, InputKind},
};
use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{EguiContexts, egui};
use burn_human_motion::{MotionRequest, Waypoint};

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
    pub(super) image_preview: Option<egui::TextureHandle>,
    image_revision: u64,
    pub(super) body: super::body_ui::BodyUi,
    tab: usize,
    selected_waypoint: Option<usize>,
}

impl Default for MotionUi {
    fn default() -> Self {
        Self {
            request: MotionRequest::default(),
            place_waypoints: false,
            bundle: if cfg!(target_arch = "wasm32") {
                String::new()
            } else {
                ".cache/ardy-bundle".into()
            },
            digest: String::new(),
            text_bundle: if cfg!(target_arch = "wasm32") {
                String::new()
            } else {
                ".cache/ardy-text-bundle".into()
            },
            text_digest: String::new(),
            embedding: String::new(),
            clip: String::new(),
            image: String::new(),
            export_path: "motion.json".into(),
            image_preview: None,
            image_revision: 0,
            body: Default::default(),
            tab: 0,
            selected_waypoint: None,
        }
    }
}

impl MotionUi {
    pub fn is_motion_tab(&self) -> bool {
        self.tab == 0
    }
}

pub(super) fn controls(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<MotionUi>,
    runtime: Res<MotionRuntime>,
    mut playback: ResMut<MotionPlayback>,
    mut display: ResMut<super::body_view::BodyDisplay>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let state = &mut *ui_state;
    let shared = runtime.0.lock().unwrap();
    let status = shared.status.clone();
    let busy = status.busy();
    let loaded = shared.model.is_some();
    let text_loaded = shared.text_model.is_some();
    let embedded = shared
        .embedding
        .as_ref()
        .is_some_and(|e| e.prompt == state.request.prompt);
    let adapter = shared.adapter.clone();
    if shared.image_revision != state.image_revision {
        state.image_revision = shared.image_revision;
        state.image_preview = None;
        state.body.image_size = [0; 2];
    }
    if let Some(image) = &shared.image
        && state.image_preview.is_none()
    {
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [image.width as usize, image.height as usize],
            &image.rgba,
        );
        state.image_preview = Some(ctx.load_texture("pose input", color, Default::default()));
    }
    drop(shared);
    super::body_ui::sync(&mut state.body, &runtime);
    egui::Window::new("Motion studio")
        .default_width(360.0)
        .default_height(620.0)
        .default_pos([12.0, 12.0])
        .vscroll(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_value(&mut state.tab, 0, "Motion").clicked() {
                    display.visible = false;
                }
                let body_tab = ui
                    .selectable_value(&mut state.tab, 1, "SOMA controls")
                    .clicked()
                    | ui.selectable_value(&mut state.tab, 2, "Image pose")
                        .clicked();
                if body_tab {
                    state.place_waypoints = false;
                    display.visible = true;
                }
            });
            match &status {
                MotionStatus::Unloaded => {
                    ui.label("Load a model bundle or import a motion clip to begin.");
                }
                MotionStatus::Ready => {
                    ui.label("Ready");
                }
                MotionStatus::Loading(n, total) => {
                    ui.add(
                        egui::ProgressBar::new(*n as f32 / *total as f32)
                            .text(format!("Loading weights {n}/{total}")),
                    );
                }
                MotionStatus::Generating(n, total) => {
                    ui.add(
                        egui::ProgressBar::new(*n as f32 / *total as f32)
                            .text(format!("Generating {n}/{total} frames")),
                    );
                }
                MotionStatus::Working(message) => {
                    ui.spinner();
                    ui.label(message);
                }
                MotionStatus::Failed(message) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, message);
                }
            }
            if state.tab == 1 {
                super::body_ui::soma(ui, &mut state.body, &runtime, busy, &mut display);
                return;
            }
            if state.tab == 2 {
                super::body_ui::image(ui, state, &runtime, busy, &mut display);
                return;
            }
            ui.label("ARDY Core · 20 FPS · Anny rig");
            model_controls(ui, state, &runtime, busy, &adapter);
            ui.separator();
            ui.label("Describe the motion");
            ui.text_edit_multiline(&mut state.request.prompt);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!busy && text_loaded, egui::Button::new("Encode prompt"))
                    .clicked()
                {
                    runtime::encode_prompt(&runtime, state.request.prompt.clone());
                }
                ui.label(if embedded {
                    "Embedding matches prompt"
                } else {
                    "Encodes automatically when generating"
                });
            });
            generation_controls(ui, state);
            trajectory_controls(ui, state, &mut playback);
            if let Err(e) = state.request.validate() {
                ui.colored_label(egui::Color32::YELLOW, e.to_string());
            }
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !busy
                            && loaded
                            && (embedded || text_loaded)
                            && state.request.validate().is_ok(),
                        egui::Button::new("Generate motion"),
                    )
                    .clicked()
                {
                    runtime::generate(&runtime, state.request.clone());
                }
                if matches!(status, MotionStatus::Generating(..)) && ui.button("Cancel").clicked() {
                    runtime
                        .0
                        .lock()
                        .unwrap()
                        .cancel
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });
            playback_controls(ui, state, &runtime, &mut playback);
            import_controls(ui, state, &runtime, busy);
        });
}

fn model_controls(
    ui: &mut egui::Ui,
    state: &mut MotionUi,
    runtime: &MotionRuntime,
    busy: bool,
    adapter: &str,
) {
    ui.collapsing("Model and text encoder", |ui| {
        ui.label("Model bundle URL or native directory");
        ui.text_edit_singleline(&mut state.bundle);
        ui.label("Pinned manifest digest (optional)");
        ui.text_edit_singleline(&mut state.digest);
        if ui
            .add_enabled(
                !busy && !state.bundle.is_empty(),
                egui::Button::new("Load model"),
            )
            .clicked()
        {
            runtime::load(runtime, state.bundle.clone(), state.digest.clone());
        }
        ui.small(format!("Shared graphics device: {adapter}"));
        ui.separator();
        ui.label("Llama text bundle URL or native directory");
        ui.text_edit_singleline(&mut state.text_bundle);
        ui.label("Text manifest digest (optional)");
        ui.text_edit_singleline(&mut state.text_digest);
        if ui
            .add_enabled(
                !busy && !state.text_bundle.is_empty(),
                egui::Button::new("Load text encoder"),
            )
            .clicked()
        {
            runtime::load_text(
                runtime,
                state.text_bundle.clone(),
                state.text_digest.clone(),
            );
        }
        ui.small("Prompts run locally on the shared GPU. Built with Meta Llama 3.");
        #[cfg(target_arch = "wasm32")]
        if ui
            .add_enabled(!busy, egui::Button::new("Choose embedding JSON…"))
            .clicked()
        {
            runtime::pick(runtime, InputKind::Embedding);
        }
        ui.label("Or import a prompt embedding (URL / native file)");
        ui.text_edit_singleline(&mut state.embedding);
        if ui
            .add_enabled(
                !busy && !state.embedding.is_empty(),
                egui::Button::new("Import embedding"),
            )
            .clicked()
        {
            runtime::import(runtime, state.embedding.clone(), InputKind::Embedding);
        }
    });
}

fn generation_controls(ui: &mut egui::Ui, state: &mut MotionUi) {
    ui.collapsing("Generation settings", |ui| {
        ui.horizontal(|ui| {
            ui.label("Frames");
            ui.add(
                egui::DragValue::new(&mut state.request.frames)
                    .range(40..=12000)
                    .speed(4),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Seed");
            ui.add(egui::DragValue::new(&mut state.request.seed));
        });
        ui.add(
            egui::Slider::new(&mut state.request.diffusion_steps, 1..=10).text("Diffusion steps"),
        );
        ui.add(
            egui::Slider::new(&mut state.request.text_guidance, 0.0..=5.0).text("Text guidance"),
        );
        ui.add(
            egui::Slider::new(&mut state.request.trajectory_guidance, 0.0..=5.0)
                .text("Path guidance"),
        );
        ui.add(
            egui::Slider::new(&mut state.request.history_frames, 0..=160)
                .step_by(4.0)
                .text("History frames"),
        );
        ui.small(
            "Fewer steps trade fidelity for speed. History and duration must be multiples of four \
             frames.",
        );
    });
}

fn trajectory_controls(ui: &mut egui::Ui, state: &mut MotionUi, playback: &mut MotionPlayback) {
    ui.collapsing("World trajectory", |ui| {
        ui.checkbox(
            &mut state.place_waypoints,
            "Place / move waypoints in scene",
        );
        ui.small(
            "Click the ground to add a waypoint. Select one below, then drag it in the scene. \
             Positions are metres; frame / 20 is time in seconds.",
        );
        ui.checkbox(
            &mut state.request.dense_trajectory,
            "Interpolate path between waypoints",
        );
        let mut remove = None;
        for (i, w) in state.request.waypoints.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(state.selected_waypoint == Some(i), format!("{}", i + 1))
                    .clicked()
                {
                    state.selected_waypoint = Some(i);
                }
                ui.add(egui::DragValue::new(&mut w.frame).prefix("frame ").speed(1));
                ui.add(
                    egui::DragValue::new(&mut w.position.x)
                        .prefix("X ")
                        .speed(0.05),
                );
                ui.add(
                    egui::DragValue::new(&mut w.position.z)
                        .prefix("Z ")
                        .speed(0.05),
                );
                if ui.small_button("×").clicked() {
                    remove = Some(i);
                }
            });
            if state.selected_waypoint == Some(i) {
                let mut orient = w.heading.is_some();
                if ui.checkbox(&mut orient, "Heading").changed() {
                    w.heading = orient.then_some(0.0);
                }
                if let Some(h) = &mut w.heading {
                    let mut degrees = h.to_degrees();
                    ui.add(egui::Slider::new(&mut degrees, -180.0..=180.0).text("Heading °"));
                    *h = degrees.to_radians();
                }
                ui.checkbox(&mut w.constrain_height, "Constrain root height");
                if w.constrain_height {
                    ui.add(
                        egui::DragValue::new(&mut w.position.y)
                            .prefix("Y ")
                            .speed(0.02),
                    );
                }
            }
        }
        if let Some(i) = remove {
            state.request.waypoints.remove(i);
            state.selected_waypoint = None;
        }
        ui.horizontal(|ui| {
            if ui.button("Add point").clicked() {
                add_waypoint(state, Vec3::ZERO);
            }
            if ui.button("Deselect").clicked() {
                state.selected_waypoint = None;
            }
            if ui.button("Clear path").clicked() {
                state.request.waypoints.clear();
                state.selected_waypoint = None;
            }
        });
        ui.checkbox(
            &mut playback.show_path,
            "Show requested (gold) and generated (blue) paths",
        );
        if ui.button("Frame trajectory").clicked() {
            playback.frame_view = true;
        }
    });
}

fn rig_controls(ui: &mut egui::Ui, playback: &mut MotionPlayback) {
    ui.collapsing("Rig controls (Core / SOMA)", |ui| {
        let names: Vec<_> = playback
            .clip
            .as_ref()
            .unwrap()
            .rig
            .joints
            .iter()
            .map(|j| j.name.clone())
            .collect();
        egui::ComboBox::from_id_salt("motion_joint")
            .selected_text(&names[playback.joint])
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    if ui.selectable_value(&mut playback.joint, i, name).changed() {
                        playback.joint_euler = [0.0; 3];
                    }
                }
            });
        for (i, label) in ["X °", "Y °", "Z °"].iter().enumerate() {
            ui.add(egui::Slider::new(&mut playback.joint_euler[i], -90.0..=90.0).text(*label));
        }
        if ui.button("Reset joint offset").clicked() {
            playback.joint_euler = [0.0; 3];
        }
    });
}

fn import_controls(ui: &mut egui::Ui, state: &mut MotionUi, runtime: &MotionRuntime, busy: bool) {
    ui.collapsing("Import motion / SOMA animation", |ui| {
        #[cfg(target_arch = "wasm32")]
        if ui
            .add_enabled(!busy, egui::Button::new("Choose motion JSON…"))
            .clicked()
        {
            runtime::pick(runtime, InputKind::Clip);
        }
        ui.label("Clip JSON URL or native file");
        ui.text_edit_singleline(&mut state.clip);
        if ui
            .add_enabled(
                !busy && !state.clip.is_empty(),
                egui::Button::new("Import motion"),
            )
            .clicked()
        {
            runtime::import(runtime, state.clip.clone(), InputKind::Clip);
        }
        ui.small(
            "SOMA animation imports carry their own bind rig. Use SOMA controls for body mesh evaluation.",
        );
    });
}

fn playback_controls(
    ui: &mut egui::Ui,
    state: &mut MotionUi,
    runtime: &MotionRuntime,
    playback: &mut MotionPlayback,
) {
    if playback.clip.is_none() {
        return;
    }
    ui.separator();
    #[cfg(not(target_arch = "wasm32"))]
    ui.text_edit_singleline(&mut state.export_path);
    if ui.button("Export clip JSON").clicked() {
        runtime::export(runtime, playback.clip.as_ref().unwrap(), &state.export_path);
    }
    ui.checkbox(
        &mut playback.active,
        "Animate Anny from motion (disable for manual posing)",
    );
    ui.horizontal(|ui| {
        if ui
            .button(if playback.playing { "Pause" } else { "Play" })
            .clicked()
        {
            playback.playing = !playback.playing;
        }
        if ui.button("Restart").clicked() {
            playback.time = 0.0;
        }
        ui.checkbox(&mut playback.looping, "Loop");
    });
    let duration = playback.clip.as_ref().unwrap().duration();
    ui.add(egui::Slider::new(&mut playback.time, 0.0..=duration).text("Seconds"));
    ui.add(egui::Slider::new(&mut playback.speed, 0.1..=2.0).text("Playback speed"));
    ui.checkbox(&mut playback.show_skeleton, "Show source rig");
    ui.checkbox(
        &mut playback.fit_root_height,
        "Fit root height to Anny proportions",
    )
    .on_hover_text("Preserves the X/Z path. Disable to use the source rig's exact root height.");
    rig_controls(ui, playback);
}

fn add_waypoint(state: &mut MotionUi, position: Vec3) {
    let frame = state.request.waypoints.last().map_or(0, |w| w.frame + 40);
    if frame < state.request.frames {
        state.request.waypoints.push(Waypoint {
            frame,
            position,
            heading: None,
            constrain_height: false,
        });
        state.selected_waypoint = None;
    }
}

pub(super) fn place_waypoints(
    mut state: ResMut<MotionUi>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut contexts: EguiContexts,
) {
    if state.tab != 0 || !state.place_waypoints || !buttons.pressed(MouseButton::Left) {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    if ctx.is_pointer_over_egui() || ctx.egui_wants_pointer_input() {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(transform, cursor) else {
        return;
    };
    if ray.direction.y.abs() < 1e-6 {
        return;
    }
    let distance = -ray.origin.y / ray.direction.y;
    if distance < 0.0 {
        return;
    }
    let mut point = ray.origin + ray.direction * distance;
    point.y = 0.02;
    if let Some(i) = state.selected_waypoint {
        if let Some(w) = state.request.waypoints.get_mut(i) {
            w.position.x = point.x;
            w.position.z = point.z;
        }
    } else if buttons.just_pressed(MouseButton::Left) {
        add_waypoint(&mut state, point);
    }
}
