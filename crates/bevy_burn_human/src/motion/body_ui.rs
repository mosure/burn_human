use super::{
    MotionRuntime,
    body::{self, BodyIdentity},
    body_view::BodyDisplay,
    runtime::{self, InputKind},
    ui::MotionUi,
};
use bevy_egui::egui;
use burn_gem::camera::{Camera, Crop};
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
}
impl Default for BodyUi {
    fn default() -> Self {
        Self {
            soma_bundle: if cfg!(target_arch = "wasm32") {
                String::new()
            } else {
                ".cache/soma-bundle-v2".into()
            },
            soma_digest: String::new(),
            gem_suite: if cfg!(target_arch = "wasm32") {
                String::new()
            } else {
                ".cache/gem-suite.json".into()
            },
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
pub(super) fn soma(
    ui: &mut egui::Ui,
    state: &mut BodyUi,
    runtime: &MotionRuntime,
    busy: bool,
    display: &mut BodyDisplay,
) {
    let shared = runtime.0.lock().unwrap();
    let loaded = shared.body.soma.is_some() || shared.body.gem.is_some();
    let names = shared.body.joint_names.clone();
    let scales = shared.body.scale_names.clone();
    drop(shared);
    ui.label("SOMA identity and rig");
    ui.small("Shape, bone lengths, hands, procedural twists and mesh correctives.");
    ui.collapsing("SOMA model", |ui| {
        ui.label("Bundle directory or URL");
        ui.text_edit_singleline(&mut state.soma_bundle);
        ui.label("Manifest digest (optional)");
        ui.text_edit_singleline(&mut state.soma_digest);
        if ui
            .add_enabled(
                !busy && !state.soma_bundle.is_empty(),
                egui::Button::new("Load SOMA"),
            )
            .clicked()
        {
            body::load_soma(
                runtime,
                state.soma_bundle.clone(),
                state.soma_digest.clone(),
            );
        }
    });
    if !loaded {
        ui.label("Load SOMA to edit a body, or estimate one from an image.");
        return;
    }
    ui.checkbox(&mut display.visible, "Show SOMA body");
    ui.checkbox(&mut display.skeleton, "Show SOMA skeleton");
    if ui.button("Frame body").clicked() {
        display.frame_view = true;
    }
    ui.separator();
    if matches!(state.identity, BodyIdentity::Image(_)) {
        ui.label("Identity from GEM-X (MHR)");
        if ui.button("Start a native SOMA identity").clicked() {
            state.identity = BodyIdentity::Native(IdentityParameters::default());
            state.pose = SomaPose::default();
            state.scale = 0;
        }
    }
    let (coeff, scales_values, global, native) = match &mut state.identity {
        BodyIdentity::Native(p) => (
            &mut p.coefficients,
            &mut p.bone_scales,
            &mut p.global_scale,
            true,
        ),
        BodyIdentity::Image(p) => (
            &mut p.coefficients,
            &mut p.scales,
            &mut p.global_scale,
            false,
        ),
    };
    ui.add(egui::Slider::new(global, 0.5..=1.5).text("Body scale"));
    ui.collapsing("Identity components", |ui| {
        ui.add(
            egui::DragValue::new(&mut state.component)
                .range(0..=coeff.len() - 1)
                .prefix("Component "),
        );
        ui.add(egui::Slider::new(&mut coeff[state.component], -3.0..=3.0).text("Weight"));
        if ui.button("Reset identity weights").clicked() {
            coeff.fill(0.0);
        }
    });
    ui.collapsing("Bone proportions", |ui| {
        state.scale = state.scale.min(scales_values.len() - 1);
        egui::ComboBox::from_id_salt("soma_scale")
            .selected_text(if native {
                scales.get(state.scale).cloned().unwrap_or_default()
            } else {
                format!("MHR scale {}", state.scale)
            })
            .show_ui(ui, |ui| {
                for i in 0..scales_values.len() {
                    let label = if native {
                        scales.get(i).cloned().unwrap_or_default()
                    } else {
                        format!("MHR scale {i}")
                    };
                    ui.selectable_value(&mut state.scale, i, label);
                }
            });
        ui.add(
            egui::Slider::new(
                &mut scales_values[state.scale],
                if native { 0.5..=1.5 } else { -0.5..=0.5 },
            )
            .text(if native {
                "Length ratio"
            } else {
                "Scale offset"
            }),
        );
    });
    ui.separator();
    if !names.is_empty() {
        state.joint = state.joint.min(names.len() - 1);
        egui::ComboBox::from_id_salt("soma_joint")
            .selected_text(&names[state.joint])
            .show_ui(ui, |ui| {
                for (i, name) in names.iter().enumerate() {
                    ui.selectable_value(&mut state.joint, i, name);
                }
            });
        for (axis, label) in ["X rotation °", "Y rotation °", "Z rotation °"]
            .iter()
            .enumerate()
        {
            let mut degrees = state.pose.rotations[state.joint][axis].to_degrees();
            if ui
                .add(egui::Slider::new(&mut degrees, -180.0..=180.0).text(*label))
                .changed()
            {
                state.pose.rotations[state.joint][axis] = degrees.to_radians();
            }
        }
    }
    if native {
        ui.checkbox(&mut state.pose.apply_correctives, "Pose correctives");
    } else {
        state.pose.apply_correctives = false;
    }
    ui.horizontal(|ui| {
        if ui.button("Reset pose").clicked() {
            state.pose = SomaPose {
                apply_correctives: native,
                ..Default::default()
            };
        }
        if ui
            .add_enabled(!busy, egui::Button::new("Apply body controls"))
            .clicked()
        {
            body::apply(runtime, state.identity.clone(), state.pose.clone());
        }
    });
    ui.small("Rotation sliders are axis-angle components in the SOMA joint convention.");
    export(ui, state, runtime, busy);
}
pub(super) fn image(
    ui: &mut egui::Ui,
    state: &mut MotionUi,
    runtime: &MotionRuntime,
    busy: bool,
    display: &mut BodyDisplay,
) {
    let shared = runtime.0.lock().unwrap();
    let loaded = shared.body.gem.is_some();
    let has_image = shared.image.is_some();
    let keypoints = shared
        .body
        .estimate
        .as_ref()
        .map(|p| p.keypoints_2d.clone());
    drop(shared);
    ui.label("Image to SOMA pose");
    ui.small("Choose one person, adjust the crop, and estimate their pose locally.");
    ui.collapsing("GEM-X models", |ui| {
        ui.label("Model suite JSON (file or URL)");
        ui.text_edit_singleline(&mut state.body.gem_suite);
        if ui
            .add_enabled(
                !busy && !state.body.gem_suite.is_empty(),
                egui::Button::new("Load GEM-X"),
            )
            .clicked()
        {
            body::load_gem(runtime, state.body.gem_suite.clone());
        }
        ui.small(if loaded {
            "Image models ready"
        } else {
            "Vision, body decoder and SOMA weights load on demand."
        });
    });
    #[cfg(target_arch = "wasm32")]
    if ui
        .add_enabled(!busy, egui::Button::new("Choose image…"))
        .clicked()
    {
        runtime::pick(runtime, InputKind::Image);
    }
    ui.text_edit_singleline(&mut state.image);
    if ui
        .add_enabled(
            !busy && !state.image.is_empty(),
            egui::Button::new("Load image"),
        )
        .clicked()
    {
        runtime::import(runtime, state.image.clone(), InputKind::Image);
    }
    if let Some(texture) = &state.image_preview {
        let response = ui.add(
            egui::Image::new(texture)
                .max_width(330.0)
                .max_height(280.0)
                .sense(egui::Sense::click_and_drag()),
        );
        let rect = response.rect;
        let image_size = egui::vec2(
            state.body.image_size[0] as f32,
            state.body.image_size[1] as f32,
        );
        let scale = rect.size() / image_size;
        if (response.dragged() || response.clicked())
            && let Some(p) = response.interact_pointer_pos()
        {
            let p = (p - rect.min) / scale;
            state.body.crop.center = [p.x, p.y];
        }
        let crop = state.body.crop;
        let center = rect.min + egui::vec2(crop.center[0], crop.center[1]) * scale;
        let size = egui::vec2(crop.size, crop.size) * scale;
        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_stroke(
            egui::Rect::from_center_size(center, size),
            0.0,
            egui::Stroke::new(2.0, egui::Color32::GOLD),
            egui::StrokeKind::Inside,
        );
        if let Some(points) = keypoints {
            for p in points {
                if p[2] > 0.5 {
                    painter.circle_filled(
                        rect.min + egui::vec2(p[0], p[1]) * scale,
                        2.0,
                        egui::Color32::LIGHT_GREEN,
                    );
                }
            }
        }
        ui.small("Drag in the image to center the person crop.");
    }
    ui.add(egui::Slider::new(&mut state.body.crop.size, 16.0..=4096.0).text("Crop size (px)"));
    ui.collapsing("Camera", |ui| {
        let mut focal = state.body.camera.focal[0];
        if ui
            .add(
                egui::DragValue::new(&mut focal)
                    .range(10.0..=20000.0)
                    .prefix("Focal length (px) ")
                    .speed(5.0),
            )
            .changed()
        {
            state.body.camera.focal = [focal; 2];
        }
        ui.small("Defaults to the image diagonal. Set calibrated focal length when available.");
    });
    if ui
        .add_enabled(
            !busy && loaded && has_image,
            egui::Button::new("Estimate pose"),
        )
        .clicked()
    {
        body::estimate(runtime, state.body.crop, state.body.camera);
    }
    ui.checkbox(&mut display.visible, "Show estimated SOMA body");
    ui.checkbox(&mut display.skeleton, "Show skeleton");
    ui.small("Open SOMA controls to adjust the inferred identity and joints.");
    export(ui, &mut state.body, runtime, busy);
}
fn export(ui: &mut egui::Ui, state: &mut BodyUi, runtime: &MotionRuntime, busy: bool) {
    #[cfg(not(target_arch = "wasm32"))]
    ui.text_edit_singleline(&mut state.export_path);
    if ui
        .add_enabled(!busy, egui::Button::new("Export pose and identity"))
        .clicked()
    {
        let value = serde_json::json!({"schema":1,"identity":state.identity,"pose":state.pose,"camera":state.camera,"crop":state.crop});
        runtime::export_json(runtime, &value, &state.export_path, "pose.json");
    }
}
