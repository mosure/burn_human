//! GPU surface handoff. Topology uploads once; vertex positions never visit the
//! host. A 24-byte bounds readback supports grounding, gizmos and camera framing.
use anyhow::{Result, ensure};
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    render::render_resource::{
        AsBindGroup, Buffer, RenderPipelineDescriptor, SpecializedMeshPipelineError, VertexFormat,
    },
    shader::ShaderRef,
};
use burn::tensor::Tensor;
use burn_human_inference::gpu::{self, WgpuBackend};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub(super) const VERTEX_INDEX: MeshVertexAttribute =
    MeshVertexAttribute::new("SurfaceIndex", 1_305_191_502, VertexFormat::Uint32);
const SHADER_PATH: &str = "embedded://bevy_burn_human/motion/surface_vertex.wgsl";

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub(super) struct SurfaceMaterial {
    #[storage(100, read_only, buffer)]
    pub vertices: Buffer,
}
pub(super) type BodyMaterial = ExtendedMaterial<StandardMaterial, SurfaceMaterial>;
impl MaterialExtension for SurfaceMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_PATH.into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        SHADER_PATH.into()
    }
    fn deferred_vertex_shader() -> ShaderRef {
        SHADER_PATH.into()
    }
    fn specialize(
        _: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialExtensionKey<Self>,
    ) -> std::result::Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.buffers =
            vec![layout.0.get_layout(&[VERTEX_INDEX.at_shader_location(8)])?];
        Ok(())
    }
}
pub(super) fn plugin(app: &mut App) {
    bevy::asset::embedded_asset!(app, "surface_vertex.wgsl");
    app.add_plugins(MaterialPlugin::<BodyMaterial>::default());
}

pub(super) struct Topology {
    pub faces: Vec<[u32; 3]>,
    pub count: usize,
    ranges: wgpu::Buffer,
    neighbors: wgpu::Buffer,
}
pub(super) struct GpuSurface {
    pub vertices: Buffer,
    pub topology: Arc<Topology>,
    pub low: Vec3,
    pub high: Vec3,
}
pub(super) struct SurfaceProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
struct Adjacency {
    ranges: Vec<[u32; 2]>,
    pairs: Vec<[u32; 2]>,
}
fn adjacency(count: usize, faces: &[[u32; 3]]) -> Result<Adjacency> {
    ensure!(count > 0 && !faces.is_empty(), "Empty surface topology");
    let mut around = vec![Vec::new(); count];
    for &[a, b, c] in faces {
        ensure!(
            [a, b, c].iter().all(|i| (*i as usize) < count),
            "Surface index out of range"
        );
        around[a as usize].push([b, c]);
        around[b as usize].push([c, a]);
        around[c as usize].push([a, b]);
    }
    let mut pairs = Vec::with_capacity(faces.len() * 3);
    let ranges = around
        .into_iter()
        .map(|list| {
            let start = pairs.len() as u32;
            pairs.extend(list);
            [start, pairs.len() as u32]
        })
        .collect();
    Ok(Adjacency { ranges, pairs })
}
impl SurfaceProcessor {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding != 3,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SOMA surface layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SOMA surface pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SOMA area-weighted normals"),
            source: wgpu::ShaderSource::Wgsl(include_str!("surface_normals.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SOMA surface normals"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("normals"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            layout,
            pipeline,
        }
    }
    pub fn topology(&self, count: usize, faces: &[[u32; 3]]) -> Result<Arc<Topology>> {
        let Adjacency {
            ranges,
            pairs: neighbors,
        } = adjacency(count, faces)?;
        let buffer = |label, data: &[[u32; 2]]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        Ok(Arc::new(Topology {
            faces: faces.to_vec(),
            count,
            ranges: buffer("SOMA adjacency ranges", &ranges),
            neighbors: buffer("SOMA adjacency pairs", &neighbors),
        }))
    }
    pub async fn prepare(
        &self,
        positions: Tensor<WgpuBackend, 3>,
        topology: Arc<Topology>,
    ) -> Result<GpuSurface> {
        ensure!(
            positions.dims() == [1, topology.count, 3],
            "Surface tensor dimensions"
        );
        let flat = positions.clone().reshape([topology.count, 3]);
        let bounds = Tensor::cat(vec![flat.clone().min_dim(0), flat.max_dim(0)], 0);
        let lease = gpu::export(positions)?;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SOMA resident vertices and normals"),
            size: topology.count as u64 * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SOMA surface inputs"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lease.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: topology.ranges.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: topology.neighbors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SOMA surface handoff"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SOMA parallel normals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((topology.count as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        // This async read is queued after the normals pass. Its completion also
        // makes it safe to release Burn's allocation lease. No device.poll wait
        // occurs in the Bevy update system, and no vertex array is read back.
        let data = bounds
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Surface bounds: {e}"))?;
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "Non-finite surface bounds"
        );
        drop(lease);
        Ok(GpuSurface {
            vertices: output.into(),
            topology,
            low: Vec3::from_slice(&data[..3]),
            high: Vec3::from_slice(&data[3..]),
        })
    }
}
pub(super) fn mesh(topology: &Topology) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3]; topology.count]);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![[0.0, 1.0, 0.0]; topology.count],
    );
    mesh.insert_attribute(VERTEX_INDEX, (0..topology.count as u32).collect::<Vec<_>>());
    mesh.insert_indices(Indices::U32(
        topology.faces.iter().flatten().copied().collect(),
    ));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adjacency_preserves_winding_and_shared_vertices() {
        let Adjacency { ranges, pairs } = adjacency(4, &[[0, 1, 2], [0, 2, 3]]).unwrap();
        assert_eq!(
            &pairs[ranges[0][0] as usize..ranges[0][1] as usize],
            &[[1, 2], [2, 3]]
        );
        assert_eq!(
            &pairs[ranges[2][0] as usize..ranges[2][1] as usize],
            &[[0, 1], [3, 0]]
        );
        assert!(adjacency(3, &[[0, 1, 3]]).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires a hardware WGPU adapter"]
    fn gpu_handoff_preserves_positions_normals_and_bounds() {
        use burn::{
            backend::wgpu::{graphics::AutoGraphicsApi, init_setup},
            tensor::TensorData,
        };
        let device = Default::default();
        let setup = init_setup::<AutoGraphicsApi>(&device, Default::default());
        let processor = SurfaceProcessor::new(setup.device.clone(), setup.queue.clone());
        let topology = processor.topology(4, &[[0, 1, 2]]).unwrap();
        // Millimetre triangles, an isolated vertex, a sliced allocation and a
        // strided tensor all exercise cases a raw buffer clone cannot handle.
        let values = vec![
            0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 1.0, 1.0, 1.0,
        ];
        let plain = Tensor::<WgpuBackend, 3>::from_data(
            TensorData::new(values.clone(), [1, 4, 3]),
            &device,
        );
        let sliced = Tensor::cat(vec![Tensor::zeros([1, 4, 3], &device), plain.clone()], 1)
            .slice([0..1, 4..8, 0..3]);
        let transposed: Vec<_> = (0..3)
            .flat_map(|axis| values.as_chunks::<3>().0.iter().map(move |v| v[axis]))
            .collect();
        let strided =
            Tensor::<WgpuBackend, 3>::from_data(TensorData::new(transposed, [1, 3, 4]), &device)
                .swap_dims(1, 2);
        for tensor in [plain, sliced, strided] {
            let surface = pollster::block_on(processor.prepare(tensor, topology.clone())).unwrap();
            assert_eq!(surface.low, Vec3::ZERO);
            assert_eq!(surface.high, Vec3::ONE);
            let staging = setup.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4 * 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = setup.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&surface.vertices, 0, &staging, 0, 4 * 32);
            setup.queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            setup
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range();
            let data: &[f32] = bytemuck::cast_slice(&mapped);
            for (i, vertex) in data.as_chunks::<8>().0.iter().enumerate() {
                assert_eq!(&vertex[..3], &values[i * 3..i * 3 + 3]);
                let expected = if i == 3 {
                    [0.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                };
                for (actual, expected) in vertex[4..7].iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
            super::super::body_view::check_changed_topology_keeps_gpu_bounds(
                surface,
                processor.topology(4, &[[0, 1, 2]]).unwrap(),
            );
        }
    }
}
