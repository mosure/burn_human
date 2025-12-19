#import bevy_pbr::forward_io::VertexOutput

// Pack everything into a single uniform buffer: WebGPU has a low per-stage uniform-buffer limit.
// Layout matches Rust `TubeMaterial.u: [Vec4; 6]`:
// [params0, params1, orange, white, dark_inside, dark_outside]
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> u: array<vec4<f32>, 6>;

const TAU: f32 = 6.28318530718;

fn aa_band(phase: f32, aa_mul: f32) -> f32 {
    let s = 0.5 + 0.5 * sin(phase);
    let w = fwidth(phase) * aa_mul;
    return smoothstep(0.5 - w, 0.5 + w, s);
}

@fragment
fn fragment(mesh: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let params0 = u[0];
    let params1 = u[1];
    let orange = u[2];
    let white = u[3];
    let dark_inside = u[4];
    let dark_outside = u[5];

    let time = params0.x;
    let bands = params0.y;
    let turns = params0.z;
    let spin = params0.w;

    let flow = params1.x;
    let aa = params1.y;
    let white_bias = params1.z;
    let pattern = params1.w;

    let ang = mesh.uv.y * TAU;
    let s = mesh.uv.x;

    let s_warp = pow(s, 1.18);

    // pattern 0: swirl (codepen)
    // pattern 1: stripes (simple variant)
    var phase: f32;
    if (pattern < 0.5) {
        let theta = ang + time * spin + s_warp * TAU * turns;
        phase = theta * bands + time * flow;
    } else {
        // stripes: fewer turns, more axial flow
        let theta = ang + s_warp * TAU * 16.0;
        phase = theta * (bands * 0.75) + time * (flow * 3.0);
    }

    let band = aa_band(phase, aa);

    let t = smoothstep(white_bias, 1.0, band);
    var col = mix(white.rgb, orange.rgb, t);

    let depth = smoothstep(0.0, 1.0, s);
    let dark = select(dark_inside.rgb, dark_outside.rgb, is_front);
    col = mix(col, dark, depth * 0.40);

    return vec4<f32>(col, 1.0);
}
