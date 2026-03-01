// VSM depth fragment shader
//
// Outputs the first two moments of the depth distribution for variance shadow
// mapping. Uses a linear depth metric (light-space Z for directional lights)
// and the partial-derivative bias from Lauritzen §8.4.2 to encode per-texel
// depth extent into the second moment.
//
// Bind group layout: none (uses only interpolated vertex output)

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) light_depth: f32,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec2<f32> {
    let depth = in.light_depth;

    // Partial derivatives of depth across the fragment quad.
    let dx = dpdx(depth);
    let dy = dpdy(depth);

    // Second moment: depth² + bias term encoding the texel's depth extent.
    // This eliminates shadow acne without requiring manual depth bias tuning.
    // Reference: Lauritzen §8.4.2, Listing 8-2.
    let moment2 = depth * depth + 0.25 * (dx * dx + dy * dy);

    return vec2<f32>(depth, moment2);
}
