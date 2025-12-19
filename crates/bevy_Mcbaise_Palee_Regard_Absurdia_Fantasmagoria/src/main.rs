#![cfg_attr(target_arch = "wasm32", no_main)]

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::pbr::{DistanceFog, FogFalloff, Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::reflect::TypePath;
use bevy::shader::ShaderRef;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
#[cfg(not(target_arch = "wasm32"))]
use bevy::window::PrimaryWindow;

#[cfg(target_arch = "wasm32")]
use bevy_burn_human::BurnHumanSource;
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanPlugin};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[cfg(target_arch = "wasm32")]
const TENSOR_BYTES: &[u8] = include_bytes!("../../../assets/model/fullbody_default.safetensors");
#[cfg(target_arch = "wasm32")]
const META_BYTES: &[u8] = include_bytes!("../../../assets/model/fullbody_default.meta.json");

const VIDEO_ID: &str = "v2hcW03gcus";

const TUBE_RADIUS: f32 = 3.4;
const SUBJECT_RADIUS: f32 = 0.78;
const WALL_R: f32 = TUBE_RADIUS - SUBJECT_RADIUS;
const SUBJECT_INSET: f32 = 0.18;

const FRAMES_SAMPLES: usize = 3200;

#[cfg(not(target_arch = "wasm32"))]
const TUBULAR_SEGMENTS: usize = 2600;
#[cfg(not(target_arch = "wasm32"))]
const RADIAL_SEGMENTS: usize = 96;

#[cfg(target_arch = "wasm32")]
const TUBULAR_SEGMENTS: usize = 1800;
#[cfg(target_arch = "wasm32")]
const RADIAL_SEGMENTS: usize = 64;

#[derive(Component)]
struct TubeTag;

#[derive(Component)]
struct SubjectTag;

#[derive(Component)]
struct MainCamera;

#[derive(Resource, Clone)]
struct Playback {
    time_sec: f32,
    playing: bool,
    speed: f32,
}

#[derive(Resource, Clone)]
struct TubeSettings {
    scheme: u32,
    pattern: u32,
}

#[derive(Resource)]
struct TubeScene {
    curve: CatmullRomCurve,
    frames: Frames,
    tube_material: Handle<TubeMaterial>,
}

#[derive(Resource)]
struct OverlayState {
    last_credit_idx: i32,
    last_caption_idx: i32,
}

#[derive(Resource, Clone, Default)]
struct OverlayText {
    credit: String,
    caption: String,
    caption_is_meta: bool,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            last_credit_idx: -1,
            last_caption_idx: -1,
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static JS_INPUT: std::cell::RefCell<JsInput> = const { std::cell::RefCell::new(JsInput::new()) };
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct JsInput {
    has_time: bool,
    time_sec: f32,
    playing: bool,
    toggle_scheme: bool,
    toggle_texture: bool,
    speed_delta: i32,
}

#[cfg(target_arch = "wasm32")]
impl JsInput {
    const fn new() -> Self {
        Self {
            has_time: false,
            time_sec: 0.0,
            playing: false,
            toggle_scheme: false,
            toggle_texture: false,
            speed_delta: 0,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_video_time(time_sec: f32, playing: bool) {
    JS_INPUT.with(|s| {
        let mut st = s.borrow_mut();
        st.has_time = true;
        st.time_sec = time_sec.max(0.0);
        st.playing = playing;
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn toggle_color_scheme() {
    JS_INPUT.with(|s| s.borrow_mut().toggle_scheme = true);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn toggle_texture() {
    JS_INPUT.with(|s| s.borrow_mut().toggle_texture = true);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn speed_up() {
    JS_INPUT.with(|s| s.borrow_mut().speed_delta += 1);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn slow_down() {
    JS_INPUT.with(|s| s.borrow_mut().speed_delta -= 1);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = mcbaise_request_playing)]
    fn mcbaise_request_playing(playing: bool);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let burn_plugin = {
        #[cfg(target_arch = "wasm32")]
        {
            BurnHumanPlugin {
                source: BurnHumanSource::Bytes {
                    tensor: TENSOR_BYTES,
                    meta: META_BYTES,
                },
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            BurnHumanPlugin::default()
        }
    };

    let plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: format!("{VIDEO_ID} • tube ride"),
            #[cfg(target_arch = "wasm32")]
            canvas: Some("#bevy-canvas".to_string()),
            #[cfg(target_arch = "wasm32")]
            fit_canvas_to_parent: true,
            ..default()
        }),
        ..default()
    });

    // On wasm, Bevy's HTTP asset reader will try to fetch `<asset>.meta` as well.
    // We don't ship meta files for this demo, so disable meta-checking to avoid 404 spam.
    #[cfg(target_arch = "wasm32")]
    let plugins = plugins.set(bevy::asset::AssetPlugin {
        meta_check: bevy::asset::AssetMetaCheck::Never,
        ..default()
    });

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Playback {
            time_sec: 0.0,
            playing: cfg!(not(target_arch = "wasm32")),
            speed: 1.0,
        })
        .insert_resource(TubeSettings {
            scheme: 0,
            pattern: 0,
        })
        .insert_resource(OverlayText::default())
        .add_plugins(plugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(MaterialPlugin::<TubeMaterial>::default())
        .add_plugins(burn_plugin)
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                #[cfg(target_arch = "wasm32")]
                apply_js_input,
                #[cfg(not(target_arch = "wasm32"))]
                advance_time_native,
                #[cfg(not(target_arch = "wasm32"))]
                native_controls,
                update_tube_and_subject,
                update_overlays,
            ),
        )
        .add_systems(EguiPrimaryContextPass, ui_overlay)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tube_materials: ResMut<Assets<TubeMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    assets: Res<BurnHumanAssets>,
) {
    commands.insert_resource(AmbientLight {
        color: Color::srgb(1.0, 1.0, 1.0),
        brightness: 0.25,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(10.0, 18.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let curve = make_random_loop_curve(1337);
    let frames = build_frames(&curve, FRAMES_SAMPLES);

    let tube_mesh = meshes.add(build_tube_mesh(
        &curve,
        &frames,
        TUBULAR_SEGMENTS,
        RADIAL_SEGMENTS,
        TUBE_RADIUS,
    ));

    let tube_mat = tube_materials.add(TubeMaterial::default());

    commands.spawn((
        Mesh3d(tube_mesh),
        MeshMaterial3d(tube_mat.clone()),
        Transform::default(),
        TubeTag,
        Name::new("tube"),
    ));

    let phenotype_len = assets.body.metadata().metadata.phenotype_labels.len();
    let selected_case = assets
        .body
        .metadata()
        .cases
        .iter()
        .position(|c| c.pose_parameters.shape[0] == 1)
        .unwrap_or(0usize);

    commands.spawn((
        BurnHumanInput {
            case_name: assets
                .body
                .metadata()
                .metadata
                .case_names
                .get(selected_case)
                .cloned(),
            phenotype_inputs: Some(vec![0.5; phenotype_len]),
            ..Default::default()
        },
        MeshMaterial3d(std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.7, 0.68),
            metallic: 0.0,
            reflectance: 0.5,
            perceptual_roughness: 0.6,
            emissive: Color::srgb(0.14, 0.12, 0.10).into(),
            cull_mode: None,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(1.35)),
        Visibility::default(),
        SubjectTag,
        Name::new("burn_human_subject"),
    ));

    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 0.02,
            far: 3000.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 0.0, -8.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
        DistanceFog {
            color: Color::srgb_u8(0x12, 0x00, 0x00),
            falloff: FogFalloff::Linear {
                start: 10.0,
                end: 260.0,
            },
            ..default()
        },
    ));

    commands.insert_resource(TubeScene {
        curve,
        frames,
        tube_material: tube_mat,
    });

    commands.insert_resource(OverlayState::default());
}

#[cfg(not(target_arch = "wasm32"))]
fn advance_time_native(time: Res<Time>, mut playback: ResMut<Playback>) {
    if playback.playing {
        playback.time_sec += time.delta_secs() * playback.speed;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_controls(keys: Res<ButtonInput<KeyCode>>, mut playback: ResMut<Playback>, mut settings: ResMut<TubeSettings>) {
    if keys.just_pressed(KeyCode::Space) {
        playback.playing = !playback.playing;
    }
    if keys.just_pressed(KeyCode::Digit1) {
        settings.scheme = (settings.scheme + 1) % 2;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        settings.pattern = (settings.pattern + 1) % 2;
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        playback.speed = (playback.speed + 0.25).clamp(0.25, 3.0);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        playback.speed = (playback.speed - 0.25).clamp(0.25, 3.0);
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_js_input(mut playback: ResMut<Playback>, mut settings: ResMut<TubeSettings>) {
    JS_INPUT.with(|s| {
        let mut st = s.borrow_mut();
        if st.has_time {
            playback.time_sec = st.time_sec;
            playback.playing = st.playing;
            st.has_time = false;
        }
        if st.toggle_scheme {
            settings.scheme = (settings.scheme + 1) % 2;
            st.toggle_scheme = false;
        }
        if st.toggle_texture {
            settings.pattern = (settings.pattern + 1) % 2;
            st.toggle_texture = false;
        }
        if st.speed_delta != 0 {
            // +/- 0.25 per click, clamped.
            playback.speed = (playback.speed + st.speed_delta as f32 * 0.25).clamp(0.25, 3.0);
            st.speed_delta = 0;
        }
    });
}

fn update_tube_and_subject(
    playback: Res<Playback>,
    settings: Res<TubeSettings>,
    tube_scene: Res<TubeScene>,
    mut tube_materials: ResMut<Assets<TubeMaterial>>,
    mut subject: Query<&mut Transform, With<SubjectTag>>,
    mut cam: Query<&mut Transform, (With<MainCamera>, Without<SubjectTag>)>,
) {
    let t = playback.time_sec;

    // Update tube shader params.
    if let Some(mat) = tube_materials.get_mut(&tube_scene.tube_material) {
        mat.set_time(t);
        mat.set_scheme(settings.scheme);
        mat.set_pattern(settings.pattern);
    }

    // Drive along curve.
    let progress = progress_from_video_time(t);

    let cam_center = tube_scene.curve.point_at(progress);
    let f = tube_scene.frames.frame_at(progress);

    let cam_tangent = f.tan;
    let cam_n = f.nor;
    let cam_b = f.bin;

    let look_ahead = tube_scene.curve.point_at((progress + 0.003).min(0.99));

    // Subject position on wall.
    let ball_ahead = if camera_mode(t) == CameraMode::BallChase {
        0.020
    } else {
        0.010
    };
    let s = (progress + ball_ahead).min(0.99);
    let center = tube_scene.curve.point_at(s);
    let bf = tube_scene.frames.frame_at(s);

    let theta = theta_from_time(t);
    let offset = bf.nor * theta.cos() + bf.bin * theta.sin();
    let subject_pos = center + offset * (WALL_R - SUBJECT_INSET);

    if let Ok(mut tr) = subject.single_mut() {
        // Align forward to tangent, up to radial.
        let up = (subject_pos - center).normalize_or_zero();
        let forward = bf.tan.normalize_or_zero();
        let right = forward.cross(up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let rot = Quat::from_mat3(&Mat3::from_cols(right, up, forward));

        tr.translation = subject_pos;
        tr.rotation = rot;
    }

    if let Ok(mut cam_tr) = cam.single_mut() {
        let (pos, look, up) =
            camera_pose(t, cam_center, look_ahead, cam_tangent, cam_n, cam_b, center, subject_pos);

        // Smooth like the codepen.
        cam_tr.translation = cam_tr.translation.lerp(pos, 0.20);
        let desired = Transform::from_translation(pos).looking_at(look, up).rotation;
        cam_tr.rotation = cam_tr.rotation.slerp(desired, 0.20);
    }
}

fn update_overlays(
    playback: Res<Playback>,
    mut state: ResMut<OverlayState>,
    mut overlay_text: ResMut<OverlayText>,
    #[cfg(not(target_arch = "wasm32"))] mut window: Query<&mut Window, With<PrimaryWindow>>,
) {
    let t = playback.time_sec;

    let c_idx = find_opening_credit(t);
    if c_idx != state.last_credit_idx {
        state.last_credit_idx = c_idx;
        let credit = if c_idx < 0 {
            ""
        } else {
            opening_credit_plain(c_idx as usize)
        };

        overlay_text.credit = credit.to_string();

        #[cfg(not(target_arch = "wasm32"))]
        {
            if !credit.is_empty() {
                println!("[credit] {credit}");
            }

            if let Ok(mut w) = window.single_mut() {
                if credit.is_empty() {
                    w.title = format!("{VIDEO_ID} • tube ride");
                } else {
                    w.title = format!("{VIDEO_ID} • tube ride — {credit}");
                }
            }
        }
    }

    let cues = lyric_cues();
    let l_idx = find_cue_index(&cues, t);
    if l_idx != state.last_caption_idx {
        state.last_caption_idx = l_idx;
        if l_idx < 0 {
            overlay_text.caption.clear();
            overlay_text.caption_is_meta = false;
        } else {
            let cue = &cues[l_idx as usize];
            overlay_text.caption = cue.text.clone();
            overlay_text.caption_is_meta = cue.is_meta;

            #[cfg(not(target_arch = "wasm32"))]
            {
                if cue.is_meta {
                    println!("[caption] ({})", cue.text);
                } else {
                    println!("[caption] {}", cue.text);
                }
            }
        }
    }
}

fn ui_overlay(
    mut egui_contexts: EguiContexts,
    overlay_text: Res<OverlayText>,
    mut playback: ResMut<Playback>,
    mut settings: ResMut<TubeSettings>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("mcbaise_overlay")
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        .show(ctx, |ui| {
            ui.label(format!("{VIDEO_ID} • tube ride"));
            if !overlay_text.credit.is_empty() {
                ui.strong(&overlay_text.credit);
            }
            if !overlay_text.caption.is_empty() {
                if overlay_text.caption_is_meta {
                    ui.label(format!("({})", overlay_text.caption));
                } else {
                    ui.label(&overlay_text.caption);
                }
            }

            ui.separator();

            ui.horizontal(|ui| {
                let desired_playing = !playback.playing;
                let label = if playback.playing { "Pause" } else { "Play" };
                if ui.button(label).clicked() {
                    playback.playing = desired_playing;
                    #[cfg(target_arch = "wasm32")]
                    {
                        mcbaise_request_playing(desired_playing);
                    }
                }

                ui.label(format!("Speed: {:.2}x", playback.speed));
                if ui.button("-").clicked() {
                    playback.speed = (playback.speed - 0.25).clamp(0.25, 3.0);
                }
                if ui.button("+").clicked() {
                    playback.speed = (playback.speed + 0.25).clamp(0.25, 3.0);
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Toggle Colors").clicked() {
                    settings.scheme = (settings.scheme + 1) % 2;
                }
                if ui.button("Toggle Texture").clicked() {
                    settings.pattern = (settings.pattern + 1) % 2;
                }
            });
        });

}

// ---------------------------- time → curve progress ----------------------------

fn progress_from_video_time(video_time_sec: f32) -> f32 {
    let speed = 0.0028;
    (video_time_sec * speed).min(0.985).max(0.0)
}

fn theta_from_time(t: f32) -> f32 {
    // Simple periodic theta (the original code integrates dynamics; this is a stable approximation).
    t * 1.35
}

// ---------------------------- camera ----------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    First,
    Over,
    Back,
    BallChase,
}

fn camera_mode(video_time_sec: f32) -> CameraMode {
    // Without the YouTube panel (native exe), prefer a camera that actually shows the subject.
    if cfg!(not(target_arch = "wasm32")) {
        return CameraMode::BallChase;
    }
    let cycle = 14.0;
    let u = video_time_sec.rem_euclid(cycle);
    if u > 6.0 && u <= 8.5 {
        CameraMode::Over
    } else if u > 8.5 && u <= 11.0 {
        CameraMode::Back
    } else if u > 11.0 {
        CameraMode::BallChase
    } else {
        CameraMode::First
    }
}

fn camera_pose(
    video_time_sec: f32,
    cam_center: Vec3,
    look_ahead: Vec3,
    cam_tangent: Vec3,
    cam_n: Vec3,
    cam_b: Vec3,
    ball_center: Vec3,
    subject_pos: Vec3,
) -> (Vec3, Vec3, Vec3) {
    let (intro_show_tube, intro_dive) = if cfg!(target_arch = "wasm32") {
        (2.2, 1.6)
    } else {
        // Native has no embedded YouTube UI; start inside immediately.
        (0.0, 0.0)
    };
    let in_intro = video_time_sec < (intro_show_tube + intro_dive);

    let cam_inner_pos = cam_center + cam_b * 0.10;

    let first_pos = cam_inner_pos;
    let first_look = look_ahead;
    let first_up = cam_n;

    let over_pos = cam_center + cam_n * 18.0 + cam_b * 14.0;
    let over_look = cam_center;
    let over_up = cam_tangent.cross(cam_n).normalize_or_zero();

    let back_pos = cam_center + cam_tangent * -12.0 + cam_n * 1.2;
    let back_look = cam_center + cam_tangent * 3.0;
    let back_up = cam_n;

    let chase_pos = subject_pos + cam_tangent * -4.8 + cam_n * 0.9;
    let chase_look = subject_pos;
    let chase_up = cam_n;

    let mut pos;
    let mut look;
    let mut up;

    match camera_mode(video_time_sec) {
        CameraMode::First => {
            pos = first_pos;
            look = first_look;
            up = first_up;
        }
        CameraMode::Over => {
            pos = over_pos;
            look = over_look;
            up = over_up;
        }
        CameraMode::Back => {
            pos = back_pos;
            look = back_look;
            up = back_up;
        }
        CameraMode::BallChase => {
            pos = chase_pos;
            look = chase_look;
            up = chase_up;
        }
    }

    if in_intro {
        let _mid_s = 0.18_f32.min(0.99);
        // Approx: use current frame as mid; visually close.
        let mid = ball_center;
        let far_pos = mid + cam_n * 70.0 + cam_b * 55.0 + cam_tangent * -40.0;
        let far_look = mid + cam_tangent * 40.0;
        let far_up = cam_n;

        if intro_dive <= 0.0 || video_time_sec < intro_show_tube {
            pos = far_pos;
            look = far_look;
            up = far_up;
        } else {
            let a = (video_time_sec - intro_show_tube) / intro_dive;
            let blend = a.clamp(0.0, 1.0);
            pos = far_pos.lerp(pos, blend);
            look = far_look.lerp(look, blend);
            up = far_up.lerp(up, blend).normalize_or_zero();
        }
    }

    (pos, look, up)
}

// ---------------------------- curve + frames ----------------------------

#[derive(Clone)]
struct CatmullRomCurve {
    pts: Vec<Vec3>,
    tension: f32,
}

impl CatmullRomCurve {
    fn point_at(&self, u: f32) -> Vec3 {
        let u = u.clamp(0.0, 1.0);
        let segs = (self.pts.len() - 1) as f32;
        let scaled = u * segs;
        let i = scaled.floor() as isize;
        let t = scaled - i as f32;

        let i1 = i.clamp(0, (self.pts.len() - 2) as isize) as usize;
        let i0 = i1.saturating_sub(1);
        let i2 = (i1 + 1).min(self.pts.len() - 1);
        let i3 = (i1 + 2).min(self.pts.len() - 1);

        catmull_rom(
            self.pts[i0],
            self.pts[i1],
            self.pts[i2],
            self.pts[i3],
            t,
            self.tension,
        )
    }

    fn tangent_at(&self, u: f32) -> Vec3 {
        let eps = 0.0005;
        let a = self.point_at((u - eps).max(0.0));
        let b = self.point_at((u + eps).min(1.0));
        (b - a).normalize_or_zero()
    }
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    // Cubic Hermite form.
    let v0 = (p2 - p0) * tension;
    let v1 = (p3 - p1) * tension;

    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    p1 * h00 + v0 * h10 + p2 * h01 + v1 * h11
}

#[derive(Clone)]
struct Frames {
    tangents: Vec<Vec3>,
    normals: Vec<Vec3>,
    binormals: Vec<Vec3>,
    samples: usize,
}

impl Frames {
    fn frame_at(&self, u: f32) -> Frame {
        let u = u.clamp(0.0, 1.0);
        let i_f = u * (self.samples as f32 - 1.0);
        let i = i_f.floor() as usize;
        let i = i.min(self.samples - 2);
        let t = i_f - i as f32;

        let tan = self.tangents[i].lerp(self.tangents[i + 1], t).normalize_or_zero();
        let nor = self.normals[i].lerp(self.normals[i + 1], t).normalize_or_zero();
        let bin = self.binormals[i]
            .lerp(self.binormals[i + 1], t)
            .normalize_or_zero();
        Frame { tan, nor, bin }
    }
}

#[derive(Clone, Copy)]
struct Frame {
    tan: Vec3,
    nor: Vec3,
    bin: Vec3,
}

fn build_frames(curve: &CatmullRomCurve, samples: usize) -> Frames {
    let mut tangents = Vec::with_capacity(samples);
    for i in 0..samples {
        let u = i as f32 / (samples as f32 - 1.0);
        tangents.push(curve.tangent_at(u));
    }

    let mut normals = Vec::with_capacity(samples);
    let mut binormals = Vec::with_capacity(samples);

    let mut n0 = Vec3::Y;
    if n0.dot(tangents[0]).abs() > 0.9 {
        n0 = Vec3::X;
    }
    n0 = (n0 - tangents[0] * n0.dot(tangents[0])).normalize_or_zero();

    normals.push(n0);
    binormals.push(tangents[0].cross(normals[0]).normalize_or_zero());

    for i in 1..samples {
        let t_prev = tangents[i - 1];
        let t_cur = tangents[i];

        let axis = t_prev.cross(t_cur);
        let axis_len = axis.length();

        let mut n_prev = normals[i - 1];
        if axis_len > 1e-8 {
            let axis_n = axis / axis_len;
            let angle = t_prev.dot(t_cur).clamp(-1.0, 1.0).acos();
            let q = Quat::from_axis_angle(axis_n, angle);
            n_prev = q * n_prev;
        }

        let n_cur = (n_prev - t_cur * n_prev.dot(t_cur)).normalize_or_zero();
        let b_cur = t_cur.cross(n_cur).normalize_or_zero();
        normals.push(n_cur);
        binormals.push(b_cur);
    }

    Frames {
        tangents,
        normals,
        binormals,
        samples,
    }
}

fn make_random_loop_curve(seed: u64) -> CatmullRomCurve {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut pts = Vec::with_capacity(1800);

    let total = 1800;
    let step_z = 1.2;

    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;

    let mut loop_countdown: i32 = 0;
    let mut loop_phase: f32 = 0.0;
    let mut loop_radius: f32 = 0.0;
    let mut loop_freq: f32 = 0.0;

    let rand = |rng: &mut fastrand::Rng, a: f32, b: f32| a + (b - a) * rng.f32();
    let rand_sign = |rng: &mut fastrand::Rng| if rng.f32() < 0.5 { -1.0 } else { 1.0 };

    for i in 0..total {
        if loop_countdown <= 0 && rng.f32() < 0.02 && i > 60 {
            loop_countdown = rand(&mut rng, 70.0, 160.0).floor() as i32;
            loop_phase = rand(&mut rng, 0.0, std::f32::consts::TAU);
            loop_radius = rand(&mut rng, 3.5, 7.5);
            loop_freq = rand(&mut rng, 0.20, 0.55) * rand_sign(&mut rng);
        }

        if loop_countdown > 0 {
            loop_phase += loop_freq;
            x += loop_phase.cos() * 0.35 * loop_radius;
            y += loop_phase.sin() * 0.35 * loop_radius;
            loop_countdown -= 1;
        } else {
            vx += rand(&mut rng, -0.08, 0.08);
            vy += rand(&mut rng, -0.08, 0.08);
            vx *= 0.96;
            vy *= 0.96;
            x += vx * 2.2;
            y += vy * 2.2;
            x *= 0.995;
            y *= 0.995;
        }

        z += step_z;
        pts.push(Vec3::new(x, y, z));
    }

    CatmullRomCurve { pts, tension: 0.35 }
}

fn build_tube_mesh(
    curve: &CatmullRomCurve,
    frames: &Frames,
    tubular_segments: usize,
    radial_segments: usize,
    radius: f32,
) -> Mesh {
    let rings = tubular_segments + 1;
    let ring_verts = radial_segments + 1;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(rings * ring_verts);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(rings * ring_verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(rings * ring_verts);

    for j in 0..rings {
        let u = j as f32 / tubular_segments as f32;
        let center = curve.point_at(u);
        let f = frames.frame_at(u);

        for i in 0..ring_verts {
            let v = i as f32 / radial_segments as f32;
            let ang = v * std::f32::consts::TAU;
            let dir = f.nor * ang.cos() + f.bin * ang.sin();
            let p = center + dir * radius;
            positions.push([p.x, p.y, p.z]);
            normals.push([dir.x, dir.y, dir.z]);
            uvs.push([u, v]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(tubular_segments * radial_segments * 6);
    for j in 0..tubular_segments {
        let ring0 = j * ring_verts;
        let ring1 = (j + 1) * ring_verts;
        for i in 0..radial_segments {
            let a = (ring0 + i) as u32;
            let b = (ring1 + i) as u32;
            let c = (ring1 + i + 1) as u32;
            let d = (ring0 + i + 1) as u32;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh
}

// ---------------------------- tube material ----------------------------

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct TubeMaterial {
    #[uniform(0)]
    // Pack everything into a single uniform buffer: WebGPU has a low per-stage uniform-buffer limit.
    // Layout: [params0, params1, orange, white, dark_inside, dark_outside]
    u: [Vec4; 6],
}

impl Default for TubeMaterial {
    fn default() -> Self {
        Self {
            u: [
                Vec4::new(0.0, 5.0, 240.0, -1.35),
                Vec4::new(0.08, 1.70, 0.22, 0.0),
                Color::srgb_u8(0xD2, 0x6B, 0x11).to_linear().to_vec4(),
                Color::srgb_u8(0xFF, 0xF8, 0xF0).to_linear().to_vec4(),
                Color::srgb_u8(0x0A, 0x00, 0x00).to_linear().to_vec4(),
                Color::srgb_u8(0x05, 0x00, 0x00).to_linear().to_vec4(),
            ],
        }
    }
}

impl TubeMaterial {
    fn set_time(&mut self, t: f32) {
        self.u[0].x = t;
    }

    fn set_scheme(&mut self, scheme: u32) {
        match scheme % 2 {
            0 => {
                self.u[2] = Color::srgb_u8(0xD2, 0x6B, 0x11).to_linear().to_vec4();
                self.u[3] = Color::srgb_u8(0xFF, 0xF8, 0xF0).to_linear().to_vec4();
            }
            _ => {
                // Alt scheme: cyan/pink.
                self.u[2] = Color::srgb_u8(0x18, 0xC5, 0xC5).to_linear().to_vec4();
                self.u[3] = Color::srgb_u8(0xFF, 0xC2, 0xF0).to_linear().to_vec4();
            }
        }
    }

    fn set_pattern(&mut self, pattern: u32) {
        self.u[1].w = (pattern % 2) as f32;
    }
}

impl Material for TubeMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/mcbaise_tube.wgsl".into()
    }
}

// ---------------------------- opening credits + captions ----------------------------

struct Credit {
    start: f32,
    end: f32,
    #[allow(dead_code)]
    html: &'static str,
    #[allow(dead_code)]
    plain: &'static str,
}

fn opening_credits() -> &'static [Credit] {
    &[
        Credit {
            start: 0.00,
            end: 2.70,
            html: r#"<span style=\"font-size:3em; letter-spacing:.06em;\">DIRTY<br>MELODY</span><br><span style=\"font-size:3em; letter-spacing:.06em;\">RECORDS</span><br><span style=\"font-size:2em; opacity:.95; letter-spacing:.08em;\">Owns All Rights</span>"#,
            plain: "DIRTY MELODY RECORDS — Owns All Rights",
        },
        Credit {
            start: 2.70,
            end: 4.80,
            html: r#"<span style=\"font-size:3em; letter-spacing:.06em;\">MCBAISE<br><span style=\"font-size:.70em;\">PALE REGARD</span>"#,
            plain: "MCBAISE — PALE REGARD",
        },
        Credit {
            start: 4.80,
            end: 8.40,
            html: r#"<span style=\"font-size:5.25em;\">ABSURDIA</span><br><span style=\"font-size:5.05em;\">FANTASMAGORIA</span>"#,
            plain: "ABSURDIA FANTASMAGORIA",
        },
    ]
}
fn find_opening_credit(t: f32) -> i32 {
    for (i, c) in opening_credits().iter().enumerate() {
        if t >= c.start && t <= c.end {
            return i as i32;
        }
    }
    -1
}
#[allow(dead_code)]
fn opening_credit_html(idx: usize) -> &'static str {
    opening_credits()
        .get(idx)
        .map(|c| c.html)
        .unwrap_or("")
}

#[allow(dead_code)]
fn opening_credit_plain(idx: usize) -> &'static str {
    opening_credits()
        .get(idx)
        .map(|c| c.plain)
        .unwrap_or("")
}
#[derive(Clone)]
struct Cue {
    start: f32,
    end: f32,
    text: String,
    is_meta: bool,
}
fn lyric_srt_text() -> &'static str {
    // Note: This is intentionally minimal; ellipsis blocks from the original snippet are ignored.
    "1\n00:00:00,120 --> 00:00:54,570\n[Music]\n\n2\n00:00:53,970 --> 00:01:04,980\nJ'me lève | I get up / I'm getting up\n"
}
fn lyric_cues() -> Vec<Cue> {
    parse_srt(lyric_srt_text())
}
fn find_cue_index(cues: &[Cue], t: f32) -> i32 {
    for (i, c) in cues.iter().enumerate() {
        if t >= c.start && t <= c.end {
            return i as i32;
        }
    }
    -1
}
fn parse_srt(srt: &str) -> Vec<Cue> {
    let srt = srt.replace('\r', "");
    let blocks: Vec<&str> = srt.split("\n\n").map(str::trim).filter(|b| !b.is_empty()).collect();
    let mut out = Vec::new();

    for block in blocks {
        let lines: Vec<&str> = block
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        if lines.len() < 2 {
            continue;
        }
        let time_line = lines.iter().copied().find(|l| l.contains("-->"));
        let Some(time_line) = time_line else { continue };
        let mut parts = time_line.split("-->").map(str::trim);
        let Some(a) = parts.next() else { continue };
        let Some(b) = parts.next() else { continue };

        let (Some(start), Some(end)) = (parse_timecode(a), parse_timecode(b)) else {
            continue;
        };

        let text_lines: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|l| *l != time_line && l.parse::<u32>().is_err())
            .collect();
        let text = text_lines.join(" ").trim().to_string();
        if text.is_empty() {
            continue;
        }
        let is_meta = matches!(text.to_lowercase().as_str(), "[music]" | "[applause]");
        out.push(Cue {
            start,
            end,
            text,
            is_meta,
        });
    }

    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}
fn parse_timecode(tc: &str) -> Option<f32> {
    let (hms, ms) = tc.split_once(',')?;
    let mut parts = hms.split(':');
    let hh: f32 = parts.next()?.parse::<f32>().ok()?;
    let mm: f32 = parts.next()?.parse::<f32>().ok()?;
    let ss: f32 = parts.next()?.parse::<f32>().ok()?;
    let ms: f32 = ms.parse::<f32>().ok()?;
    Some(hh * 3600.0 + mm * 60.0 + ss + (ms / 1000.0))
}
