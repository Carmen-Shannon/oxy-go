// Rainbow fragment shader
//
// Receives per-vertex color interpolated across the triangle by the
// rasterizer and outputs it directly. No textures or lighting — pure
// vertex-color pass-through for rainbow-style rendering.

// ── Input from vertex shader (interpolated) ────────────────────────
struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return in.color;
}
