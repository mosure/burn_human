struct SurfaceVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> neighbors: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> vertices: array<SurfaceVertex>;
fn point(i: u32) -> vec3<f32> {
    return vec3(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
}
@compute @workgroup_size(64)
fn normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&ranges) { return; }
    let p = point(i);
    var normal = vec3(0.0);
    for (var j = ranges[i].x; j < ranges[i].y; j++) {
        let pair = neighbors[j];
        normal += cross(point(pair.x) - p, point(pair.y) - p);
    }
    let length2 = dot(normal, normal);
    if length2 > 1e-30 { normal *= inverseSqrt(length2); }
    vertices[i].position = vec4(p, 1.0);
    vertices[i].normal = vec4(normal, 0.0);
}
