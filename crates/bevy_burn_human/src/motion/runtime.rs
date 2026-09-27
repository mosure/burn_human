use anyhow::Result;
use bevy::{
    prelude::*,
    render::{
        RenderApp,
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
    },
};
use burn::backend::wgpu::{WgpuDevice, WgpuSetup, init_device};
use burn_ardy::{Ardy, transport::read_bounded};
use burn_human_inference::gpu::WgpuBackend as Wgpu;
use burn_human_motion::{
    ImageCondition, MotionClip, MotionRequest, TextEmbedding, soma::SomaAnimation,
};
use burn_llama::TextEncoder;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug)]
pub enum MotionStatus {
    Unloaded,
    Ready,
    Loading(usize, usize),
    Generating(usize, usize),
    Working(String),
    Failed(String),
}
impl MotionStatus {
    pub fn busy(&self) -> bool {
        matches!(
            self,
            Self::Loading(..) | Self::Generating(..) | Self::Working(..)
        )
    }
}

pub(super) struct RuntimeState {
    pub status: MotionStatus,
    pub device: Option<WgpuDevice>,
    pub model: Option<Ardy<Wgpu>>,
    pub text_model: Option<TextEncoder<Wgpu>>,
    pub embedding: Option<TextEmbedding>,
    pub clip: Option<MotionClip>,
    pub image: Option<ImageCondition>,
    pub image_revision: u64,
    pub cancel: Arc<AtomicBool>,
    pub adapter: String,
    pub body: super::body::BodyState,
}
#[derive(Resource, Clone)]
pub struct MotionRuntime(pub(super) Arc<Mutex<RuntimeState>>);
impl Default for MotionRuntime {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(RuntimeState {
            status: MotionStatus::Unloaded,
            device: None,
            model: None,
            text_model: None,
            embedding: None,
            clip: None,
            image: None,
            image_revision: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            adapter: String::new(),
            body: Default::default(),
        })))
    }
}

pub(super) fn initialize_device(app: &mut App) {
    let setup = (|| -> Option<WgpuSetup> {
        let render = app.get_sub_app(RenderApp)?;
        let world = render.world();
        let instance = world.get_resource::<RenderInstance>()?;
        let adapter = world.get_resource::<RenderAdapter>()?;
        let device = world.get_resource::<RenderDevice>()?;
        let queue = world.get_resource::<RenderQueue>()?;
        Some(WgpuSetup {
            instance: (****instance).clone(),
            adapter: (****adapter).clone(),
            device: device.wgpu_device().clone(),
            queue: (****queue).clone(),
            backend: adapter.get_info().backend,
        })
    })();
    let runtime = app.world().resource::<MotionRuntime>();
    let mut state = runtime.0.lock().unwrap();
    if let Some(setup) = setup {
        state.body.processor = Some(Arc::new(super::surface::SurfaceProcessor::new(
            setup.device.clone(),
            setup.queue.clone(),
        )));
        state.adapter = setup.adapter.get_info().name;
        if state.adapter.is_empty() {
            state.adapter = format!("{:?}", setup.backend);
        }
        state.device = Some(init_device(setup, Default::default()));
    } else {
        state.status = MotionStatus::Failed(
            "Bevy GPU device unavailable; motion inference requires WebGPU/WGPU".into(),
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn spawn_job<F, Fu>(runtime: MotionRuntime, job: F)
where
    F: FnOnce() -> Fu + Send + 'static,
    Fu: Future<Output = Result<()>> + 'static,
{
    // One serial worker matches the runtime's single-flight policy. Live rig
    // edits reuse the thread instead of starting an OS thread for every pose.
    type Job = Box<dyn FnOnce() + Send>;
    static WORKER: std::sync::OnceLock<std::sync::mpsc::Sender<Job>> = std::sync::OnceLock::new();
    let worker = WORKER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("human-inference".into())
            .spawn(move || {
                for job in receiver {
                    job();
                }
            })
            .expect("start inference worker");
        sender
    });
    worker
        .send(Box::new(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || pollster::block_on(job()),
                ));
            let error = match result {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(e.to_string()),
                Err(_) => Some("GPU job failed; see the console for device diagnostics".into()),
            };
            if let Some(e) = error {
                runtime.0.lock().unwrap().status = MotionStatus::Failed(e);
            }
        }))
        .expect("inference worker is alive");
}
#[cfg(target_arch = "wasm32")]
pub(super) fn spawn_job<F, Fu>(runtime: MotionRuntime, job: F)
where
    F: FnOnce() -> Fu + 'static,
    Fu: Future<Output = Result<()>> + 'static,
{
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = job().await {
            runtime.0.lock().unwrap().status = MotionStatus::Failed(e.to_string());
        }
    });
}

pub(super) fn load(runtime: &MotionRuntime, base: String, digest: String) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(device) = state.device.clone() else {
        state.status = MotionStatus::Failed("No shared GPU device".into());
        return;
    };
    state.status = MotionStatus::Loading(0, 1);
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let mut artifact = burn_ardy::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest.trim().into();
        let model = Ardy::load_artifact(&artifact, &device, |i, n| {
            shared.lock().unwrap().status = MotionStatus::Loading(i, n)
        })
        .await?;
        let mut state = shared.lock().unwrap();
        state.model = Some(model);
        state.status = MotionStatus::Ready;
        Ok(())
    });
}

/// One discoverable action loads both prerequisites, retaining a successful
/// first stage if the second download needs to be retried.
pub(super) fn prepare_motion(
    runtime: &MotionRuntime,
    model: (String, String),
    text: (String, String),
) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(device) = state.device.clone() else {
        state.status = MotionStatus::Failed("No shared GPU device".into());
        return;
    };
    let need_model = state.model.is_none();
    let need_text = state.text_model.is_none();
    state.status = MotionStatus::Working("Preparing motion models".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        if need_model {
            let mut artifact = burn_ardy::pretrained::DEFAULT.at_base(model.0);
            artifact.sha256 = model.1;
            let model = Ardy::load_artifact(&artifact, &device, |i, n| {
                shared.lock().unwrap().status =
                    MotionStatus::Working(format!("Loading ARDY {i}/{n}"));
            })
            .await?;
            shared.lock().unwrap().model = Some(model);
        }
        if need_text {
            let mut artifact = burn_llama::pretrained::DEFAULT.at_base(text.0);
            artifact.sha256 = text.1;
            let encoder = TextEncoder::load_artifact(&artifact, &device, |i, n| {
                shared.lock().unwrap().status =
                    MotionStatus::Working(format!("Loading Llama {i}/{n}"));
            })
            .await?;
            shared.lock().unwrap().text_model = Some(encoder);
        }
        shared.lock().unwrap().status = MotionStatus::Ready;
        Ok(())
    });
}

pub(super) fn generate(runtime: &MotionRuntime, request: MotionRequest) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    if let Err(e) = request.validate() {
        state.status = MotionStatus::Failed(e.to_string());
        return;
    }
    let embedding = state
        .embedding
        .clone()
        .filter(|e| e.validate(&request.prompt, 4096).is_ok());
    if embedding.is_none() && state.text_model.is_none() {
        state.status =
            MotionStatus::Failed("Load the Llama text encoder to generate from a prompt".into());
        return;
    }
    let Some(model) = state.model.take() else {
        state.status = MotionStatus::Failed("Load the ARDY bundle first".into());
        return;
    };
    let mut encoder = if embedding.is_none() {
        state.text_model.take()
    } else {
        None
    };
    state.status = MotionStatus::Working("Preparing prompt".into());
    state.cancel.store(false, Ordering::Relaxed);
    let cancel = state.cancel.clone();
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let result = async {
            let embedding = if let Some(e) = embedding {
                e
            } else {
                let e = encoder
                    .as_mut()
                    .unwrap()
                    .encode(&request.prompt, |i, n| {
                        shared.lock().unwrap().status =
                            MotionStatus::Working(format!("Encoding prompt {i}/{n}"));
                    })
                    .await?;
                shared.lock().unwrap().embedding = Some(e.clone());
                e
            };
            model
                .generate(&request, &embedding, |n| {
                    shared.lock().unwrap().status = MotionStatus::Generating(n, request.frames);
                    !cancel.load(Ordering::Relaxed)
                })
                .await
        }
        .await;
        let mut state = shared.lock().unwrap();
        state.model = Some(model);
        if let Some(encoder) = encoder {
            state.text_model = Some(encoder);
        }
        match result {
            Ok(clip) => {
                state.clip = Some(clip);
                state.status = MotionStatus::Ready;
            }
            Err(e) => state.status = MotionStatus::Failed(e.to_string()),
        }
        Ok(())
    });
}

#[derive(Clone, Copy)]
pub(super) enum InputKind {
    Clip,
    Embedding,
    Image,
}
pub(super) fn import(runtime: &MotionRuntime, location: String, kind: InputKind) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    state.status = MotionStatus::Working("Reading input".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let limit = match kind {
            InputKind::Clip => 32 * 1024 * 1024,
            InputKind::Embedding => 256 * 1024,
            InputKind::Image => 8 * 1024 * 1024,
        };
        let bytes = read_bounded(&location, limit).await?;
        apply_input(&shared, bytes, kind)?;
        shared.lock().unwrap().status = MotionStatus::Ready;
        Ok(())
    });
}

fn apply_input(shared: &Arc<Mutex<RuntimeState>>, bytes: Vec<u8>, kind: InputKind) -> Result<()> {
    match kind {
        InputKind::Clip => {
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            let clip: MotionClip = if value.get("poses").is_some() {
                serde_json::from_value::<SomaAnimation>(value)?.into_clip()?
            } else {
                serde_json::from_value(value)?
            };
            clip.validate()?;
            shared.lock().unwrap().clip = Some(clip);
        }
        InputKind::Embedding => {
            let embedding: TextEmbedding = serde_json::from_slice(&bytes)?;
            embedding.validate(&embedding.prompt, 4096)?;
            shared.lock().unwrap().embedding = Some(embedding);
        }
        InputKind::Image => {
            let mut reader =
                image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
            let mut limits = image::Limits::default();
            limits.max_image_width = Some(4096);
            limits.max_image_height = Some(4096);
            limits.max_alloc = Some(64 * 1024 * 1024);
            reader.limits(limits);
            let rgba = reader.decode()?.into_rgba8();
            let input = ImageCondition {
                width: rgba.width(),
                height: rgba.height(),
                rgba: rgba.into_raw(),
                crop_xywh: None,
                focal_length_px: None,
            };
            input.validate()?;
            let mut state = shared.lock().unwrap();
            state.image = Some(input);
            state.image_revision += 1;
            state.body.estimate = None;
            state.body.estimate_input = None;
        }
    }
    Ok(())
}

pub(super) fn pick(runtime: &MotionRuntime, kind: InputKind) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    state.status = MotionStatus::Working("Choose a file".into());
    drop(state);
    let shared = runtime.0.clone();
    #[cfg(not(target_arch = "wasm32"))]
    let selection = {
        let dialog = rfd::AsyncFileDialog::new();
        match kind {
            InputKind::Image => dialog.add_filter("PNG or JPEG image", &["png", "jpg", "jpeg"]),
            _ => dialog.add_filter("JSON", &["json"]),
        }
        .pick_file()
    };
    spawn_job(runtime.clone(), move || async move {
        let (accept, limit) = match kind {
            InputKind::Clip => (".json", 32 * 1024 * 1024),
            InputKind::Embedding => (".json", 256 * 1024),
            InputKind::Image => ("image/png,image/jpeg", 8 * 1024 * 1024),
        };
        #[cfg(target_arch = "wasm32")]
        {
            let value = super::browser_io::select_motion_file(accept, limit)
                .await
                .map_err(|e| anyhow::anyhow!("file selection: {e:?}"))?;
            if !value.is_null() {
                apply_input(&shared, js_sys::Uint8Array::new(&value).to_vec(), kind)?;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = accept;
            if let Some(file) = selection.await {
                let bytes = read_bounded(&file.path().to_string_lossy(), limit).await?;
                apply_input(&shared, bytes, kind)?;
            }
        }
        shared.lock().unwrap().status = MotionStatus::Ready;
        Ok(())
    });
}

pub(super) fn export(runtime: &MotionRuntime, clip: &MotionClip, path: &str) {
    // Keep long clips compact enough to round-trip through the bounded importer.
    save_bytes(runtime, serde_json::to_vec(clip), path, "motion.json");
}

pub(super) fn export_json(
    runtime: &MotionRuntime,
    value: &serde_json::Value,
    path: &str,
    name: &str,
) {
    save_bytes(runtime, serde_json::to_vec_pretty(value), path, name);
}

fn save_bytes(runtime: &MotionRuntime, bytes: serde_json::Result<Vec<u8>>, path: &str, name: &str) {
    let bytes = match bytes {
        Ok(bytes) => bytes,
        Err(e) => {
            runtime.0.lock().unwrap().status = MotionStatus::Failed(e.to_string());
            return;
        }
    };
    #[cfg(target_arch = "wasm32")]
    {
        let _ = path;
        super::browser_io::download_motion_artifact(&bytes, name);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = name;
        let mut state = runtime.0.lock().unwrap();
        if state.status.busy() {
            return;
        }
        state.status = MotionStatus::Working("Choose where to save".into());
        drop(state);
        let selection = rfd::AsyncFileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name(path)
            .save_file();
        let shared = runtime.0.clone();
        spawn_job(runtime.clone(), move || async move {
            if let Some(file) = selection.await {
                file.write(&bytes).await?;
            }
            shared.lock().unwrap().status = MotionStatus::Ready;
            Ok(())
        });
    }
}

pub(super) fn load_text(runtime: &MotionRuntime, base: String, digest: String) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(device) = state.device.clone() else {
        return;
    };
    state.status = MotionStatus::Working("Loading Llama text encoder".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let mut artifact = burn_llama::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest.trim().into();
        let encoder = TextEncoder::load_artifact(&artifact, &device, |i, n| {
            shared.lock().unwrap().status =
                MotionStatus::Working(format!("Loading Llama weights {i}/{n}"));
        })
        .await?;
        let mut state = shared.lock().unwrap();
        state.text_model = Some(encoder);
        state.embedding = None;
        state.status = MotionStatus::Ready;
        Ok(())
    });
}

pub(super) fn encode_prompt(runtime: &MotionRuntime, prompt: String) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(mut encoder) = state.text_model.take() else {
        state.status = MotionStatus::Failed("Load the Llama text bundle first".into());
        return;
    };
    state.status = MotionStatus::Working("Encoding prompt on the GPU".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let result = encoder
            .encode(&prompt, |i, n| {
                shared.lock().unwrap().status =
                    MotionStatus::Working(format!("Encoding prompt: layer {i}/{n}"));
            })
            .await;
        let mut state = shared.lock().unwrap();
        state.text_model = Some(encoder);
        match result {
            Ok(embedding) => {
                state.embedding = Some(embedding);
                state.status = MotionStatus::Ready;
            }
            Err(e) => state.status = MotionStatus::Failed(e.to_string()),
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_replacement_invalidates_preview_even_for_identical_dimensions() {
        let runtime = MotionRuntime::default();
        for (revision, red) in [(1, 0), (2, 255)] {
            let image = image::RgbaImage::from_pixel(2, 2, image::Rgba([red, 0, 0, 255]));
            let mut bytes = std::io::Cursor::new(Vec::new());
            image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
            apply_input(&runtime.0, bytes.into_inner(), InputKind::Image).unwrap();
            let state = runtime.0.lock().unwrap();
            assert_eq!(state.image_revision, revision);
            assert_eq!(state.image.as_ref().unwrap().rgba[0], red);
        }
        assert!(apply_input(&runtime.0, vec![0; 20], InputKind::Image).is_err());
        let state = runtime.0.lock().unwrap();
        assert_eq!(state.image_revision, 2);
        assert_eq!(state.image.as_ref().unwrap().rgba[0], 255);
    }
}
