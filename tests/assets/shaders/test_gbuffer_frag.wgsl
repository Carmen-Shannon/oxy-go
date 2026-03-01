// Minimal G-Buffer fragment shader for renderer integration tests.
// Outputs to two color targets: normal (rgba16float) and albedo (rgba8unorm).

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
};

struct GBufferOutput {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> GBufferOutput {
    var out: GBufferOutput;
    out.normal = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    out.albedo = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return out;
}
