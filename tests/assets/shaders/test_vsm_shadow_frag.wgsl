// Minimal VSM shadow depth fragment shader for renderer integration tests.
// Outputs the first two depth moments for variance shadow mapping.

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) light_depth: f32,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec2<f32> {
    let depth = in.light_depth;
    let dx = dpdx(depth);
    let dy = dpdy(depth);
    let moment2 = depth * depth + 0.25 * (dx * dx + dy * dy);
    return vec2<f32>(depth, moment2);
}
