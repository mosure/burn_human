use bevy::asset::{AssetLoader, LoadContext, RenderAssetUsages, io::Reader};
use bevy::log::warn;
use bevy::mesh::{
    Indices, Mesh, PrimitiveTopology, VertexAttributeValues,
    skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
};
use bevy::prelude::*;
use burn_human::data::reference::TensorData;
use burn_human::model::{blendshape, kinematics, skinning};
use burn_human::util::math::invert_rigid_mat4;
use burn_human::{AnnyBody, AnnyInput};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::{Context, anyhow, bail, Result};

/// How to load the reference data used by the Bevy plugin.
#[derive(Clone)]
pub enum BurnHumanSource {
    Paths {
        tensor: PathBuf,
        meta: PathBuf,
    },
    Bytes {
        tensor: &'static [u8],
        meta: &'static [u8],
    },
    Preloaded(Arc<AnnyBody>),
    /// Load via the Bevy asset pipeline from a meta JSON path; the loader will
    /// pull the sibling `.safetensors` alongside it.
    AssetPath(String),
    /// Load via the Bevy asset pipeline using a prepared handle to the meta JSON asset.
    Asset(Handle<BurnHumanReferenceAsset>),
}

impl BurnHumanSource {
    pub fn default_paths() -> Self {
        Self::Paths {
            tensor: PathBuf::from("assets/model/fullbody_default.safetensors"),
            meta: PathBuf::from("assets/model/fullbody_default.meta.json"),
        }
    }

    pub fn default_asset() -> Self {
        Self::AssetPath("model/fullbody_default.meta.json".to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BurnHumanMeshMode {
    BakedMesh,
    SkinnedMesh,
}

/// Per-entity render mode override.
#[derive(Component, Clone, Copy, Debug)]
pub struct BurnHumanRenderMode(pub BurnHumanMeshMode);

impl Default for BurnHumanRenderMode {
    fn default() -> Self {
        Self(BurnHumanMeshMode::SkinnedMesh)
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct BurnHumanDefaults {
    pub render_mode: BurnHumanMeshMode,
}

impl Default for BurnHumanDefaults {
    fn default() -> Self {
        Self {
            render_mode: BurnHumanMeshMode::SkinnedMesh,
        }
    }
}

pub struct BurnHumanPlugin {
    pub source: BurnHumanSource,
    pub defaults: BurnHumanDefaults,
}

impl Default for BurnHumanPlugin {
    fn default() -> Self {
        Self {
            source: BurnHumanSource::default_asset(),
            defaults: BurnHumanDefaults::default(),
        }
    }
}

impl BurnHumanPlugin {
    pub fn from_paths<T: Into<PathBuf>, M: Into<PathBuf>>(tensor: T, meta: M) -> Self {
        Self {
            source: BurnHumanSource::Paths {
                tensor: tensor.into(),
                meta: meta.into(),
            },
            defaults: BurnHumanDefaults::default(),
        }
    }

    /// Embed the reference bytes directly (useful for wasm).
    pub fn from_bytes(tensor: &'static [u8], meta: &'static [u8]) -> Self {
        Self {
            source: BurnHumanSource::Bytes { tensor, meta },
            defaults: BurnHumanDefaults::default(),
        }
    }

    /// Load reference data through the Bevy asset server from the given meta JSON path.
    /// The loader fetches the sibling `.safetensors` next to the provided meta file.
    pub fn from_asset_path(path: impl Into<String>) -> Self {
        Self {
            source: BurnHumanSource::AssetPath(path.into()),
            defaults: BurnHumanDefaults::default(),
        }
    }

    /// Load reference data through the Bevy asset server using an explicit handle.
    pub fn from_asset(handle: Handle<BurnHumanReferenceAsset>) -> Self {
        Self {
            source: BurnHumanSource::Asset(handle),
            defaults: BurnHumanDefaults::default(),
        }
    }

    pub fn with_render_mode(mut self, mode: BurnHumanMeshMode) -> Self {
        self.defaults.render_mode = mode;
        self
    }
}

/// Plugin that keeps `BurnHumanInput` entities hydrated with a cached `Mesh3d`
/// or a skinned mesh rig, depending on render mode.
impl Plugin for BurnHumanPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.defaults)
            .init_asset::<BurnHumanReferenceAsset>()
            .init_asset_loader::<ReferenceAssetLoader>()
            .add_systems(
                Update,
                (
                    hydrate_reference_asset,
                    (hydrate_burn_humans, hydrate_skinning_bindings, update_burn_humans)
                        .chain()
                        .run_if(resource_exists::<BurnHumanAssets>),
                ),
            );

        match &self.source {
            BurnHumanSource::Paths { tensor, meta } => {
                let body = Arc::new(
                    AnnyBody::from_reference_paths(tensor, meta)
                        .expect("load burn_human reference data"),
                );
                let faces = Arc::new(body.faces_quads().clone());
                let uvs = Arc::new(body.metadata().static_data.texture_coordinates.clone());
                app.insert_resource(BurnHumanAssets { body, faces, uvs });
            }
            BurnHumanSource::Bytes { tensor, meta } => {
                let body = Arc::new(
                    AnnyBody::from_reference_bytes(tensor, meta)
                        .expect("load embedded burn_human reference data"),
                );
                let faces = Arc::new(body.faces_quads().clone());
                let uvs = Arc::new(body.metadata().static_data.texture_coordinates.clone());
                app.insert_resource(BurnHumanAssets { body, faces, uvs });
            }
            BurnHumanSource::Preloaded(body) => {
                let faces = Arc::new(body.faces_quads().clone());
                let uvs = Arc::new(body.metadata().static_data.texture_coordinates.clone());
                app.insert_resource(BurnHumanAssets {
                    body: body.clone(),
                    faces,
                    uvs,
                });
            }
            BurnHumanSource::AssetPath(path) => {
                let asset_server = app
                    .world()
                    .get_resource::<AssetServer>()
                    .expect("AssetServer available")
                    .clone();
                let handle: Handle<BurnHumanReferenceAsset> = asset_server.load(path.clone());
                app.insert_resource(BurnHumanAssetHandle(handle));
            }
            BurnHumanSource::Asset(handle) => {
                app.insert_resource(BurnHumanAssetHandle(handle.clone()));
            }
        }
    }
}

#[derive(Resource, Clone)]
struct BurnHumanAssetHandle(Handle<BurnHumanReferenceAsset>);

#[derive(Asset, TypePath, Clone)]
pub struct BurnHumanReferenceAsset(pub Arc<AnnyBody>);

#[derive(Default)]
struct ReferenceAssetLoader;

impl AssetLoader for ReferenceAssetLoader {
    type Asset = BurnHumanReferenceAsset;
    type Settings = ();
    type Error = anyhow::Error;

    fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> impl bevy::tasks::ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        async move {
            let mut meta_bytes: Vec<u8> = Vec::new();
            reader.read_to_end(&mut meta_bytes).await?;
            let stem = load_context
                .path()
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("meta file missing stem"))?;
            let base = stem.strip_suffix(".meta").unwrap_or(stem);
            let tensor_path = load_context
                .path()
                .with_file_name(format!("{base}.safetensors"));
            let tensor_bytes: Vec<u8> = load_context
                .read_asset_bytes(tensor_path.clone())
                .await
                .with_context(|| format!("reading companion {:?}", tensor_path))?;
            let body = AnnyBody::from_reference_bytes(&tensor_bytes, &meta_bytes)?;
            Ok(BurnHumanReferenceAsset(Arc::new(body)))
        }
    }

    fn extensions(&self) -> &[&str] {
        &["meta.json"]
    }
}

/// Shared reference data + baked topology used by the plugin.
#[derive(Resource)]
pub struct BurnHumanAssets {
    pub body: Arc<AnnyBody>,
    pub faces: Arc<TensorData<i64>>,
    pub uvs: Arc<TensorData<f64>>,
}

#[derive(Resource)]
struct BurnHumanSkinningBindings {
    joint_indices: Arc<Vec<[u16; 4]>>,
    joint_weights: Arc<Vec<[f32; 4]>>,
}

/// Drive the Anny model inside Bevy. Change this component and the mesh updates.
#[derive(Component, Clone, Default)]
pub struct BurnHumanInput {
    pub case_name: Option<String>,
    pub phenotype_inputs: Option<Vec<f64>>,
    pub blendshape_weights: Option<Vec<f64>>,
    pub blendshape_delta: Option<Vec<f64>>,
    pub pose_parameters: Option<Vec<f64>>,
    pub pose_parameters_delta: Option<Vec<f64>>,
    pub root_translation_delta: Option<[f64; 3]>,
}

/// Per-entity mesh generation knobs.
#[derive(Component, Clone, Copy)]
pub struct BurnHumanMeshSettings {
    pub compute_normals: bool,
}

impl Default for BurnHumanMeshSettings {
    fn default() -> Self {
        Self {
            compute_normals: true,
        }
    }
}

/// Hashed version of the current input/settings used for cache lookups.
#[derive(Component, Default)]
pub struct BurnHumanCacheKey(pub u64);

/// Marks that the mesh handle on an entity was allocated by this plugin and can be mutated safely.
#[derive(Component)]
struct BurnHumanOwnedMesh;

#[derive(Component, Default)]
struct BurnHumanPhenotypeCache {
    key: u64,
    rest_vertices: Option<TensorData<f64>>,
    rest_bone_poses: Option<TensorData<f64>>,
}

type HydrateQueryItem<'w> = (
    Entity,
    Option<&'w BurnHumanMeshSettings>,
    Option<&'w BurnHumanRenderMode>,
    Option<&'w BurnHumanCacheKey>,
    Option<&'w BurnHumanPhenotypeCache>,
    Option<&'w Mesh3d>,
);

type MeshUpdateItem<'w> = (
    Entity,
    &'w BurnHumanInput,
    Ref<'w, BurnHumanMeshSettings>,
    Ref<'w, BurnHumanRenderMode>,
    &'w mut Mesh3d,
    &'w mut BurnHumanCacheKey,
    &'w mut BurnHumanPhenotypeCache,
    Option<&'w BurnHumanOwnedMesh>,
    Option<Mut<'w, SkinnedMesh>>,
);

type MeshUpdateFilter = Or<(
    Changed<BurnHumanInput>,
    Changed<BurnHumanMeshSettings>,
    Changed<BurnHumanRenderMode>,
    Added<Mesh3d>,
    Added<BurnHumanInput>,
    Added<BurnHumanMeshSettings>,
    Added<BurnHumanRenderMode>,
)>;

fn hydrate_burn_humans(
    mut commands: Commands,
    defaults: Res<BurnHumanDefaults>,
    query: Query<HydrateQueryItem<'_>, With<BurnHumanInput>>,
) {
    for (entity, settings, render_mode, cache, phenotype_cache, mesh) in query.iter() {
        let mut e = commands.entity(entity);
        if settings.is_none() {
            e.insert(BurnHumanMeshSettings::default());
        }
        if render_mode.is_none() {
            e.insert(BurnHumanRenderMode(defaults.render_mode));
        }
        if cache.is_none() {
            e.insert(BurnHumanCacheKey::default());
        }
        if phenotype_cache.is_none() {
            e.insert(BurnHumanPhenotypeCache::default());
        }
        if mesh.is_none() {
            e.insert((Mesh3d(Handle::<Mesh>::default()), BurnHumanOwnedMesh));
        }
    }
}

fn hydrate_skinning_bindings(
    mut commands: Commands,
    assets: Res<BurnHumanAssets>,
    existing: Option<Res<BurnHumanSkinningBindings>>,
) {
    if existing.is_some() {
        return;
    }
    let (indices, weights) = assets.body.skinning_bindings();
    let bindings =
        build_skinning_bindings(indices, weights).expect("build burn_human skinning bindings");
    commands.insert_resource(bindings);
}

fn update_burn_humans(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut inverse_bindposes_assets: ResMut<Assets<SkinnedMeshInverseBindposes>>,
    assets: Res<BurnHumanAssets>,
    bindings: Option<Res<BurnHumanSkinningBindings>>,
    mut joint_transforms: Query<&mut Transform>,
    mut query: Query<MeshUpdateItem<'_>, MeshUpdateFilter>,
) {
    for (
        entity,
        input,
        settings,
        render_mode,
        mut mesh_handle,
        mut pose_cache,
        mut phenotype_cache,
        owned,
        mut skinned,
    ) in query.iter_mut()
    {
        let resolved = resolve_input(&assets, input).expect("resolve burn_human input");
        let pose_key = hash_tensor_f64(&resolved.pose_parameters);
        let settings_changed = settings.is_changed();
        let mode_changed = render_mode.is_changed();

        let (phenotype_changed, phenotype) =
            ensure_phenotype_cache(&assets, &resolved, &mut phenotype_cache)
                .expect("compute burn_human phenotype");
        let pose_changed = pose_cache.0 != pose_key;

        match render_mode.0 {
            BurnHumanMeshMode::BakedMesh => {
                if let Some(skinned_mesh) = skinned.as_mut() {
                    let joints = skinned_mesh.joints.clone();
                    for joint in joints {
                        commands.entity(joint).despawn();
                    }
                    commands.entity(entity).remove::<SkinnedMesh>();
                }

                let needs_mesh = phenotype_changed
                    || pose_changed
                    || settings_changed
                    || mode_changed
                    || owned.is_none()
                    || meshes.get(&mesh_handle.0).is_none();
                if !needs_mesh {
                    continue;
                }

                let (bone_poses, _) = kinematics::forward_root_relative_world(
                    phenotype.rest_bone_poses,
                    &resolved.pose_parameters,
                    &assets.body.metadata().metadata.bone_parents,
                )
                .expect("burn_human pose");
                let (vertex_indices, vertex_weights) = assets.body.skinning_bindings();
                let posed_vertices = skinning::linear_blend_skinning(
                    phenotype.rest_vertices,
                    &bone_poses,
                    phenotype.rest_bone_poses,
                    vertex_indices,
                    vertex_weights,
                )
                .expect("burn_human skinning");
                let positions = tensor_to_vec3(&posed_vertices);
                let new_mesh = build_mesh_from_positions(
                    &positions,
                    assets.faces.as_ref(),
                    assets.uvs.as_ref(),
                    settings.compute_normals,
                );

                if owned.is_none() {
                    // Take ownership by allocating a fresh mesh asset to avoid mutating shared handles.
                    mesh_handle.0 = meshes.add(new_mesh);
                    commands.entity(entity).insert(BurnHumanOwnedMesh);
                } else if let Some(existing) = meshes.get_mut(&mesh_handle.0) {
                    *existing = new_mesh;
                } else {
                    mesh_handle.0 = meshes.add(new_mesh);
                }
                pose_cache.0 = pose_key;
            }
            BurnHumanMeshMode::SkinnedMesh => {
                let bindings = bindings.as_ref().expect("skinning bindings ready");
                let bone_count = assets.body.metadata().metadata.bone_labels.len();
                let rig_needs_rebuild = mode_changed
                    || skinned
                        .as_ref()
                        .map(|s| s.joints.len() != bone_count)
                        .unwrap_or(true);

                let needs_mesh = phenotype_changed
                    || settings_changed
                    || mode_changed
                    || owned.is_none()
                    || meshes.get(&mesh_handle.0).is_none();
                if needs_mesh {
                    let positions = tensor_to_vec3(phenotype.rest_vertices);
                    let new_mesh = build_skinned_mesh_from_positions(
                        &positions,
                        assets.faces.as_ref(),
                        assets.uvs.as_ref(),
                        bindings.joint_indices.as_ref(),
                        bindings.joint_weights.as_ref(),
                        settings.compute_normals,
                    );
                    if owned.is_none() {
                        mesh_handle.0 = meshes.add(new_mesh);
                        commands.entity(entity).insert(BurnHumanOwnedMesh);
                    } else if let Some(existing) = meshes.get_mut(&mesh_handle.0) {
                        *existing = new_mesh;
                    } else {
                        mesh_handle.0 = meshes.add(new_mesh);
                    }
                }

                let needs_bone_update = pose_changed || phenotype_changed || rig_needs_rebuild;
                if needs_bone_update {
                    let (bone_poses, _) = kinematics::forward_root_relative_world(
                        phenotype.rest_bone_poses,
                        &resolved.pose_parameters,
                        &assets.body.metadata().metadata.bone_parents,
                    )
                    .expect("burn_human pose");
                    let bone_mats = tensor_to_mat4s_first_batch(&bone_poses)
                        .expect("burn_human pose matrices");

                    if rig_needs_rebuild {
                        if let Some(skinned_mesh) = skinned.as_mut() {
                            for joint in skinned_mesh.joints.iter().copied() {
                                commands.entity(joint).despawn();
                            }
                        }
                        let inverse_bindposes =
                            inverse_bindposes_from_rest(phenotype.rest_bone_poses)
                                .expect("burn_human bindposes");
                        let handle =
                            inverse_bindposes_assets.add(SkinnedMeshInverseBindposes::from(
                                inverse_bindposes,
                            ));
                        let joints = spawn_joints(&mut commands, entity, &bone_mats);
                        commands.entity(entity).insert(SkinnedMesh {
                            inverse_bindposes: handle,
                            joints,
                        });
                    } else if let Some(skinned_mesh) = skinned.as_mut() {
                        if phenotype_changed {
                            let inverse_bindposes =
                                inverse_bindposes_from_rest(phenotype.rest_bone_poses)
                                    .expect("burn_human bindposes");
                            if let Some(existing) =
                                inverse_bindposes_assets.get_mut(&skinned_mesh.inverse_bindposes)
                            {
                                *existing = SkinnedMeshInverseBindposes::from(inverse_bindposes);
                            } else {
                                skinned_mesh.inverse_bindposes =
                                    inverse_bindposes_assets.add(SkinnedMeshInverseBindposes::from(
                                        inverse_bindposes,
                                    ));
                            }
                        }
                        for (joint, mat) in skinned_mesh.joints.iter().zip(bone_mats.iter()) {
                            if let Ok(mut transform) = joint_transforms.get_mut(*joint) {
                                *transform = Transform::from_matrix(*mat);
                            }
                        }
                    }
                    pose_cache.0 = pose_key;
                }
            }
        }
    }
}

fn hydrate_reference_asset(
    mut commands: Commands,
    handle: Option<Res<BurnHumanAssetHandle>>,
    assets: Res<Assets<BurnHumanReferenceAsset>>,
    existing: Option<Res<BurnHumanAssets>>,
) {
    let Some(handle) = handle else { return };
    if existing.is_some() {
        return;
    }
    if let Some(asset) = assets.get(&handle.0) {
        let faces = Arc::new(asset.0.faces_quads().clone());
        let uvs = Arc::new(asset.0.metadata().static_data.texture_coordinates.clone());
        commands.insert_resource(BurnHumanAssets {
            body: asset.0.clone(),
            faces,
            uvs,
        });
    }
}

impl BurnHumanInput {
    pub fn as_anny_input(&self) -> AnnyInput<'_> {
        AnnyInput {
            case_name: self.case_name.as_deref(),
            phenotype_inputs: self.phenotype_inputs.as_deref(),
            blendshape_weights: self.blendshape_weights.as_deref(),
            blendshape_delta: self.blendshape_delta.as_deref(),
            pose_parameters: self.pose_parameters.as_deref(),
            pose_parameters_delta: self.pose_parameters_delta.as_deref(),
            root_translation_delta: self.root_translation_delta,
        }
    }
}

struct ResolvedInput {
    blendshape_weights: TensorData<f64>,
    pose_parameters: TensorData<f64>,
}

struct PhenotypeData {
    rest_vertices: TensorData<f64>,
    rest_bone_poses: TensorData<f64>,
}

struct PhenotypeDataRef<'a> {
    rest_vertices: &'a TensorData<f64>,
    rest_bone_poses: &'a TensorData<f64>,
}

fn resolve_input(assets: &BurnHumanAssets, input: &BurnHumanInput) -> Result<ResolvedInput> {
    let bundle = assets.body.metadata();
    let case = match input.case_name.as_deref() {
        Some(name) => Some(
            bundle
                .cases
                .iter()
                .find(|c| c.name == name)
                .context("reference case not found")?,
        ),
        None => None,
    };

    let blend_count = bundle.static_data.blendshapes.shape[0];
    let bone_count = bundle.static_data.template_bone_heads.shape[0];
    let phenotype_len = bundle.metadata.phenotype_labels.len();

    let phenotype_inputs = if let Some(vals) = input.phenotype_inputs.as_deref() {
        ensure_len(vals.len(), phenotype_len, "phenotype_inputs")?;
        TensorData {
            shape: vec![1, phenotype_len],
            data: vals.to_vec(),
        }
    } else if let Some(case) = case {
        first_batch_2d(&case.phenotype_inputs)
    } else {
        TensorData {
            shape: vec![1, phenotype_len],
            data: vec![0.5; phenotype_len],
        }
    };

    let mut blendshape_weights = if let Some(weights) = input.blendshape_weights.as_deref() {
        ensure_len(weights.len(), blend_count, "blendshape_weights")?;
        TensorData {
            shape: vec![1, blend_count],
            data: weights.to_vec(),
        }
    } else {
        match (&input.phenotype_inputs, case) {
            (_, None) | (Some(_), _) => assets.body.phenotype_evaluator().weights(&phenotype_inputs)?,
            (None, Some(case)) => first_batch_2d(&case.blendshape_coeffs),
        }
    };
    if let Some(delta) = input.blendshape_delta.as_deref() {
        ensure_len(
            delta.len(),
            blendshape_weights.data.len(),
            "blendshape_delta",
        )?;
        for (w, d) in blendshape_weights.data.iter_mut().zip(delta.iter()) {
            *w = (*w + d).clamp(0.0, 1.0);
        }
    }

    let mut pose_parameters = if let Some(pose) = input.pose_parameters.as_deref() {
        ensure_len(pose.len(), bone_count * 16, "pose_parameters")?;
        TensorData {
            shape: vec![1, bone_count, 4, 4],
            data: pose.to_vec(),
        }
    } else if let Some(case) = case {
        first_batch_4d(&case.pose_parameters)
    } else {
        identity_pose_parameters(bone_count)
    };
    if let Some(delta) = input.pose_parameters_delta.as_deref() {
        ensure_len(
            delta.len(),
            pose_parameters.data.len(),
            "pose_parameters_delta",
        )?;
        for (p, d) in pose_parameters.data.iter_mut().zip(delta.iter()) {
            *p += d;
        }
    }
    if let Some(delta_t) = input.root_translation_delta
        && pose_parameters.data.len() >= 12
    {
        pose_parameters.data[3] += delta_t[0];
        pose_parameters.data[7] += delta_t[1];
        pose_parameters.data[11] += delta_t[2];
    }

    Ok(ResolvedInput {
        blendshape_weights,
        pose_parameters,
    })
}

fn ensure_len(actual: usize, expected: usize, label: &str) -> Result<()> {
    if actual != expected {
        bail!(
            "{} length mismatch: expected {}, got {}",
            label,
            expected,
            actual
        );
    }
    Ok(())
}

fn first_batch_2d(data: &TensorData<f64>) -> TensorData<f64> {
    if data.shape.len() != 2 || data.shape[0] <= 1 {
        return data.clone();
    }
    let cols = data.shape[1];
    TensorData {
        shape: vec![1, cols],
        data: data.data.iter().take(cols).copied().collect(),
    }
}

fn first_batch_4d(data: &TensorData<f64>) -> TensorData<f64> {
    if data.shape.len() != 4 || data.shape[0] <= 1 {
        return data.clone();
    }
    let bones = data.shape[1];
    let len = bones * 16;
    TensorData {
        shape: vec![1, bones, 4, 4],
        data: data.data.iter().take(len).copied().collect(),
    }
}

fn identity_pose_parameters(bones: usize) -> TensorData<f64> {
    let mut data = vec![0.0; bones * 16];
    for bone in 0..bones {
        let base = bone * 16;
        data[base] = 1.0;
        data[base + 5] = 1.0;
        data[base + 10] = 1.0;
        data[base + 15] = 1.0;
    }
    TensorData {
        shape: vec![1, bones, 4, 4],
        data,
    }
}

fn build_phenotype(assets: &BurnHumanAssets, resolved: &ResolvedInput) -> Result<PhenotypeData> {
    let static_data = &assets.body.metadata().static_data;
    let rest_vertices = blendshape::apply_blendshapes(
        &static_data.template_vertices,
        &static_data.blendshapes,
        &resolved.blendshape_weights,
    )?;
    let rest_bone_heads = blendshape::apply_bone_blendshapes(
        &static_data.template_bone_heads,
        &static_data.bone_heads_blendshapes,
        &resolved.blendshape_weights,
    )?;
    let rest_bone_tails = blendshape::apply_bone_blendshapes(
        &static_data.template_bone_tails,
        &static_data.bone_tails_blendshapes,
        &resolved.blendshape_weights,
    )?;
    let rest_bone_poses = kinematics::rest_bone_poses_from_heads_tails(
        &rest_bone_heads,
        &rest_bone_tails,
        &static_data.bone_rolls_rotmat,
    )?;
    Ok(PhenotypeData {
        rest_vertices,
        rest_bone_poses,
    })
}

fn ensure_phenotype_cache<'a>(
    assets: &BurnHumanAssets,
    resolved: &ResolvedInput,
    cache: &'a mut BurnHumanPhenotypeCache,
) -> Result<(bool, PhenotypeDataRef<'a>)> {
    let key = hash_tensor_f64(&resolved.blendshape_weights);
    let needs_update = cache.key != key
        || cache.rest_vertices.is_none()
        || cache.rest_bone_poses.is_none();
    if needs_update {
        let phenotype = build_phenotype(assets, resolved)?;
        cache.rest_vertices = Some(phenotype.rest_vertices);
        cache.rest_bone_poses = Some(phenotype.rest_bone_poses);
        cache.key = key;
    }
    let rest_vertices = cache
        .rest_vertices
        .as_ref()
        .expect("rest_vertices cached");
    let rest_bone_poses = cache
        .rest_bone_poses
        .as_ref()
        .expect("rest_bone_poses cached");
    Ok((
        needs_update,
        PhenotypeDataRef {
            rest_vertices,
            rest_bone_poses,
        },
    ))
}

fn hash_tensor_f64(data: &TensorData<f64>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    data.shape.hash(&mut hasher);
    for v in &data.data {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn build_mesh_from_positions(
    positions: &[Vec3],
    faces: &TensorData<i64>,
    uvs: &TensorData<f64>,
    compute_normals: bool,
) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_indices(Indices::U32(triangulate_quads(faces)));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, to_position_attribute(positions));
    if let Some(uvs) = tensor_to_uv_attribute(uvs, positions.len()) {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    if compute_normals {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            compute_normals_from_quads(positions, faces),
        );
    }
    mesh
}

fn build_skinned_mesh_from_positions(
    positions: &[Vec3],
    faces: &TensorData<i64>,
    uvs: &TensorData<f64>,
    joint_indices: &[[u16; 4]],
    joint_weights: &[[f32; 4]],
    compute_normals: bool,
) -> Mesh {
    let mut mesh = build_mesh_from_positions(positions, faces, uvs, compute_normals);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(joint_indices.to_vec()),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, joint_weights.to_vec());
    mesh
}

fn build_skinning_bindings(
    indices: &TensorData<i64>,
    weights: &TensorData<f64>,
) -> Result<BurnHumanSkinningBindings> {
    if indices.shape.len() != 2 || weights.shape.len() != 2 {
        bail!("skinning bindings must be [V,K]");
    }
    if indices.shape != weights.shape {
        bail!("skinning index/weight shapes must match");
    }
    let verts = indices.shape[0];
    let influences = indices.shape[1];
    if influences == 0 {
        bail!("skinning bindings have zero influences");
    }
    if influences > 4 {
        warn!(
            "burn_human: clamping {} bone influences to 4 for bevy skinning",
            influences
        );
    }

    let mut joint_indices = Vec::with_capacity(verts);
    let mut joint_weights = Vec::with_capacity(verts);
    for v in 0..verts {
        let base = v * influences;
        let mut slots: Vec<(f64, usize)> = Vec::with_capacity(influences);
        for i in 0..influences {
            let weight = weights.data[base + i];
            if weight <= 0.0 {
                continue;
            }
            let idx = indices.data[base + i];
            if idx < 0 {
                continue;
            }
            slots.push((weight, idx as usize));
        }
        slots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        let mut idxs = [0u16; 4];
        let mut wts = [0f32; 4];
        let mut sum = 0.0f64;
        for (slot, (weight, idx)) in slots.into_iter().take(4).enumerate() {
            if idx > u16::MAX as usize {
                bail!("bone index {} exceeds u16 range", idx);
            }
            idxs[slot] = idx as u16;
            wts[slot] = weight as f32;
            sum += weight;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for w in &mut wts {
                *w = (*w as f64 * inv) as f32;
            }
        } else {
            wts[0] = 1.0;
        }
        joint_indices.push(idxs);
        joint_weights.push(wts);
    }

    Ok(BurnHumanSkinningBindings {
        joint_indices: Arc::new(joint_indices),
        joint_weights: Arc::new(joint_weights),
    })
}

fn tensor_to_vec3(data: &TensorData<f64>) -> Vec<Vec3> {
    match data.shape.as_slice() {
        [n, 3] => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        // batched shape [B,N,3]; take the first batch (current demo renders one body)
        [b, n, 3] if *b >= 1 => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        other => panic!("expected [N,3] or [B,N,3] tensor, got shape {:?}", other),
    }
}

fn to_position_attribute(points: &[Vec3]) -> Vec<[f32; 3]> {
    points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

fn tensor_to_uv_attribute(data: &TensorData<f64>, vertex_count: usize) -> Option<Vec<[f32; 2]>> {
    let to_uvs = |n: usize, data: &[f64]| -> Vec<[f32; 2]> {
        data.chunks_exact(2)
            .take(n.min(vertex_count))
            .map(|c| [c[0] as f32, c[1] as f32])
            .collect()
    };
    match data.shape.as_slice() {
        [n, 2] => Some(to_uvs(*n, &data.data)),
        [1, n, 2] => Some(to_uvs(*n, &data.data)),
        _ => None,
    }
}

fn spawn_joints(commands: &mut Commands, entity: Entity, bone_mats: &[Mat4]) -> Vec<Entity> {
    let mut joints = Vec::with_capacity(bone_mats.len());
    commands.entity(entity).with_children(|parent| {
        for mat in bone_mats {
            let joint = parent
                .spawn((Transform::from_matrix(*mat), GlobalTransform::IDENTITY))
                .id();
            joints.push(joint);
        }
    });
    joints
}

fn tensor_to_mat4s_first_batch(data: &TensorData<f64>) -> Result<Vec<Mat4>> {
    let joints = match data.shape.as_slice() {
        [batch, joints, 4, 4] => {
            if *batch < 1 {
                bail!("bone pose batch is empty");
            }
            *joints
        }
        [joints, 4, 4] => *joints,
        other => bail!("expected [B,J,4,4] tensor, got shape {:?}", other),
    };

    let mut mats = Vec::with_capacity(joints);
    for joint in 0..joints {
        let row = slice_mat4_first_batch(data, joint)?;
        mats.push(mat4_from_row_major(&row));
    }
    Ok(mats)
}

fn inverse_bindposes_from_rest(rest_bone_poses: &TensorData<f64>) -> Result<Vec<Mat4>> {
    let joints = match rest_bone_poses.shape.as_slice() {
        [batch, joints, 4, 4] => {
            if *batch < 1 {
                bail!("rest bone pose batch is empty");
            }
            *joints
        }
        [joints, 4, 4] => *joints,
        other => bail!("expected [B,J,4,4] tensor, got shape {:?}", other),
    };

    let mut mats = Vec::with_capacity(joints);
    for joint in 0..joints {
        let row = slice_mat4_first_batch(rest_bone_poses, joint)?;
        let inv = invert_rigid_mat4(&row);
        mats.push(mat4_from_row_major(&inv));
    }
    Ok(mats)
}

fn slice_mat4_first_batch(data: &TensorData<f64>, joint: usize) -> Result<[f64; 16]> {
    let joints = match data.shape.as_slice() {
        [_, joints, 4, 4] => *joints,
        [joints, 4, 4] => *joints,
        other => bail!("expected [B,J,4,4] tensor, got shape {:?}", other),
    };
    if joint >= joints {
        bail!("bone index out of range: {}", joint);
    }
    let idx = joint * 16;
    let slice = data
        .data
        .get(idx..idx + 16)
        .ok_or_else(|| anyhow!("mat4 slice OOB"))?;
    let mut out = [0.0; 16];
    out.copy_from_slice(slice);
    Ok(out)
}

fn mat4_from_row_major(m: &[f64; 16]) -> Mat4 {
    Mat4::from_cols_array(&[
        m[0] as f32,
        m[4] as f32,
        m[8] as f32,
        m[12] as f32,
        m[1] as f32,
        m[5] as f32,
        m[9] as f32,
        m[13] as f32,
        m[2] as f32,
        m[6] as f32,
        m[10] as f32,
        m[14] as f32,
        m[3] as f32,
        m[7] as f32,
        m[11] as f32,
        m[15] as f32,
    ])
}

fn triangulate_quads(quads: &TensorData<i64>) -> Vec<u32> {
    assert_eq!(quads.shape.len(), 2, "faces tensor should be [F,4]");
    assert_eq!(quads.shape[1], 4, "faces tensor should be [F,4]");
    let mut indices = Vec::with_capacity(quads.shape[0] * 6);
    for face in quads.data.chunks_exact(4) {
        let (a, b, c, d) = (
            face[0] as u32,
            face[1] as u32,
            face[2] as u32,
            face[3] as u32,
        );
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }
    indices
}

fn compute_normals_from_quads(positions: &[Vec3], quads: &TensorData<i64>) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for face in quads.data.chunks_exact(4) {
        let a = face[0] as usize;
        let b = face[1] as usize;
        let c = face[2] as usize;
        let d = face[3] as usize;
        let pa = positions[a];
        let pb = positions[b];
        let pc = positions[c];
        let pd = positions[d];
        let n0 = (pb - pa).cross(pc - pa);
        let n1 = (pc - pa).cross(pd - pa);
        let normal = (n0 + n1).normalize_or_zero();
        for idx in [a, b, c, d] {
            normals[idx] += normal;
        }
    }
    normals
        .into_iter()
        .map(|n| n.normalize_or_zero())
        .map(|n| [n.x, n.y, n.z])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_assets() -> Result<BurnHumanAssets> {
        let body = Arc::new(AnnyBody::from_reference_paths(
            "assets/model/fullbody_default.safetensors",
            "assets/model/fullbody_default.meta.json",
        )?);
        let faces = Arc::new(body.faces_quads().clone());
        let uvs = Arc::new(body.metadata().static_data.texture_coordinates.clone());
        Ok(BurnHumanAssets { body, faces, uvs })
    }

    #[test]
    fn skinning_bindings_clamp_to_four() -> Result<()> {
        let indices = TensorData {
            shape: vec![2, 6],
            data: vec![0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0],
        };
        let weights = TensorData {
            shape: vec![2, 6],
            data: vec![0.1, 0.2, 0.3, 0.15, 0.15, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        };
        let bindings = build_skinning_bindings(&indices, &weights)?;
        assert_eq!(bindings.joint_indices.len(), 2);
        assert_eq!(bindings.joint_weights.len(), 2);
        for w in bindings.joint_weights.iter() {
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
        let used = bindings.joint_indices[0];
        for idx in used.iter() {
            assert!(!(*idx == 0 || *idx == 5));
        }
        Ok(())
    }

    #[test]
    fn skinned_mode_spawns_joints() -> Result<()> {
        let mut app = App::new();
        app.insert_resource(BurnHumanDefaults {
            render_mode: BurnHumanMeshMode::SkinnedMesh,
        });
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<SkinnedMeshInverseBindposes>::default());
        app.insert_resource(load_assets()?);
        app.add_systems(
            Update,
            (hydrate_burn_humans, hydrate_skinning_bindings, update_burn_humans)
                .chain()
                .run_if(resource_exists::<BurnHumanAssets>),
        );

        let entity = app
            .world_mut()
            .spawn((
                BurnHumanInput::default(),
                BurnHumanRenderMode(BurnHumanMeshMode::SkinnedMesh),
            ))
            .id();

        app.update();

        let world = app.world();
        let skinned = world.get::<SkinnedMesh>(entity);
        assert!(skinned.is_some());
        let skinned = skinned.unwrap();
        let bone_count = world
            .resource::<BurnHumanAssets>()
            .body
            .metadata()
            .metadata
            .bone_labels
            .len();
        assert_eq!(skinned.joints.len(), bone_count);

        let mesh_handle = world.get::<Mesh3d>(entity).expect("mesh handle");
        let meshes = world.resource::<Assets<Mesh>>();
        let mesh = meshes.get(&mesh_handle.0).expect("mesh asset");
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT).is_some());

        Ok(())
    }

    #[test]
    fn baked_mode_skips_skinning() -> Result<()> {
        let mut app = App::new();
        app.insert_resource(BurnHumanDefaults {
            render_mode: BurnHumanMeshMode::BakedMesh,
        });
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<SkinnedMeshInverseBindposes>::default());
        app.insert_resource(load_assets()?);
        app.add_systems(
            Update,
            (hydrate_burn_humans, hydrate_skinning_bindings, update_burn_humans)
                .chain()
                .run_if(resource_exists::<BurnHumanAssets>),
        );

        let entity = app
            .world_mut()
            .spawn((
                BurnHumanInput::default(),
                BurnHumanRenderMode(BurnHumanMeshMode::BakedMesh),
            ))
            .id();

        app.update();

        let world = app.world();
        assert!(world.get::<SkinnedMesh>(entity).is_none());
        let mesh_handle = world.get::<Mesh3d>(entity).expect("mesh handle");
        let meshes = world.resource::<Assets<Mesh>>();
        let mesh = meshes.get(&mesh_handle.0).expect("mesh asset");
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_none());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT).is_none());
        Ok(())
    }
}
