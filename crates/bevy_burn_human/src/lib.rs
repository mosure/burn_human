use bevy::asset::{AssetLoader, LoadContext, RenderAssetUsages, io::Reader};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use burn_human::data::reference::TensorData;
use burn_human::{AnnyBody, AnnyInput};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::{Context, anyhow};

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

pub struct BurnHumanPlugin {
    pub source: BurnHumanSource,
}

impl Default for BurnHumanPlugin {
    fn default() -> Self {
        Self {
            source: BurnHumanSource::default_asset(),
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
        }
    }

    /// Embed the reference bytes directly (useful for wasm).
    pub fn from_bytes(tensor: &'static [u8], meta: &'static [u8]) -> Self {
        Self {
            source: BurnHumanSource::Bytes { tensor, meta },
        }
    }

    /// Load reference data through the Bevy asset server from the given meta JSON path.
    /// The loader fetches the sibling `.safetensors` next to the provided meta file.
    pub fn from_asset_path(path: impl Into<String>) -> Self {
        Self {
            source: BurnHumanSource::AssetPath(path.into()),
        }
    }

    /// Load reference data through the Bevy asset server using an explicit handle.
    pub fn from_asset(handle: Handle<BurnHumanReferenceAsset>) -> Self {
        Self {
            source: BurnHumanSource::Asset(handle),
        }
    }
}

/// Plugin that keeps `BurnHumanInput` entities hydrated with a cached `Mesh3d`.
impl Plugin for BurnHumanPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<BurnHumanReferenceAsset>()
            .init_asset_loader::<ReferenceAssetLoader>()
            .add_systems(
                Update,
                (
                    hydrate_reference_asset,
                    (hydrate_burn_humans, update_burn_human_meshes)
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

type HydrateQueryItem<'w> = (
    Entity,
    Option<&'w BurnHumanMeshSettings>,
    Option<&'w BurnHumanCacheKey>,
    Option<&'w Mesh3d>,
);

type MeshUpdateItem<'w> = (
    Entity,
    &'w BurnHumanInput,
    &'w BurnHumanMeshSettings,
    &'w mut Mesh3d,
    &'w mut BurnHumanCacheKey,
    Option<&'w BurnHumanOwnedMesh>,
);

type MeshUpdateFilter = Or<(
    Changed<BurnHumanInput>,
    Changed<BurnHumanMeshSettings>,
    Added<Mesh3d>,
    Added<BurnHumanInput>,
    Added<BurnHumanMeshSettings>,
)>;

fn hydrate_burn_humans(
    mut commands: Commands,
    query: Query<HydrateQueryItem<'_>, With<BurnHumanInput>>,
) {
    for (entity, settings, cache, mesh) in query.iter() {
        let mut e = commands.entity(entity);
        if settings.is_none() {
            e.insert(BurnHumanMeshSettings::default());
        }
        if cache.is_none() {
            e.insert(BurnHumanCacheKey::default());
        }
        if mesh.is_none() {
            e.insert((Mesh3d(Handle::<Mesh>::default()), BurnHumanOwnedMesh));
        }
    }
}

fn update_burn_human_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<BurnHumanAssets>,
    mut query: Query<MeshUpdateItem<'_>, MeshUpdateFilter>,
) {
    for (entity, input, settings, mut mesh_handle, mut cached, owned) in query.iter_mut() {
        let key = cache_key(input, settings.compute_normals);
        if cached.0 == key {
            continue;
        }

        let output = assets
            .body
            .forward(input.as_anny_input())
            .expect("burn_human forward");
        let new_mesh = build_mesh(
            &output,
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
        cached.0 = key;
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

fn cache_key(input: &BurnHumanInput, compute_normals: bool) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.case_name.hash(&mut hasher);
    hash_opt_slice(&mut hasher, input.phenotype_inputs.as_deref());
    hash_opt_slice(&mut hasher, input.blendshape_weights.as_deref());
    hash_opt_slice(&mut hasher, input.blendshape_delta.as_deref());
    hash_opt_slice(&mut hasher, input.pose_parameters.as_deref());
    hash_opt_slice(&mut hasher, input.pose_parameters_delta.as_deref());
    if let Some(delta) = input.root_translation_delta {
        for f in delta {
            f.to_bits().hash(&mut hasher);
        }
    }
    compute_normals.hash(&mut hasher);
    hasher.finish()
}

fn hash_opt_slice(hasher: &mut std::collections::hash_map::DefaultHasher, data: Option<&[f64]>) {
    if let Some(slice) = data {
        for v in slice {
            v.to_bits().hash(hasher);
        }
    }
}

fn build_mesh(
    output: &burn_human::AnnyOutput,
    faces: &TensorData<i64>,
    uvs: &TensorData<f64>,
    compute_normals: bool,
) -> Mesh {
    let positions = tensor_to_vec3(&output.posed_vertices);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_indices(Indices::U32(triangulate_quads(faces)));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, to_position_attribute(&positions));
    if let Some(uvs) = tensor_to_uv_attribute(uvs, positions.len()) {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    if compute_normals {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            compute_normals_from_quads(&positions, faces),
        );
    }
    mesh
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
