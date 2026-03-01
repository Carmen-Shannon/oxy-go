// Minimal composition fragment shader for renderer integration tests.
// Outputs a single color for the surface target.

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.uv, 0.0, 1.0);
}
