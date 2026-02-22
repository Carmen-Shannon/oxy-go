// Minimal fragment shader for renderer integration tests.

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
