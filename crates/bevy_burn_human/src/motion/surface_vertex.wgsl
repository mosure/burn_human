#import bevy_pbr::{mesh_functions, view_transformations::position_world_to_clip}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::VertexOutput
#else
#import bevy_pbr::forward_io::VertexOutput
#endif
struct SurfaceVertex { position: vec4<f32>, normal: vec4<f32> }
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<storage, read> surface: array<SurfaceVertex>;
struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(8) surface_index: u32,
}
@vertex
fn vertex(input: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let data = surface[input.surface_index];
    let world_from_local = mesh_functions::get_world_from_local(input.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, data.position);
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef PREPASS_PIPELINE
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(data.normal.xyz, input.instance_index);
#endif
#ifdef MOTION_VECTOR_PREPASS
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        mesh_functions::get_previous_world_from_local(input.instance_index), data.position);
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#else
    out.world_normal = mesh_functions::mesh_normal_local_to_world(data.normal.xyz, input.instance_index);
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = input.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(input.instance_index, world_from_local[3]);
#endif
    return out;
}
