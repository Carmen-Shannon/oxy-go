// Tint overlay fragment shader
//
// Renders a semi-transparent color pass over existing geometry for a tint
// effect. Used with alpha blending and no depth write so it composites on
// top of the base material without occluding it.
//
// The tint uniform uses RGB for color and alpha for intensity:
//   - alpha 0.0 = fully transparent (no visible tint)
//   - alpha 0.5 = 50% blend toward tint color
//   - alpha 1.0 = fully opaque tint color

// ── Input from vertex shader (must match all vertex output locations) ──
struct FragmentInput {
    @location(0) uv:             vec2<f32>,
    @location(1) normal:         vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

//@oxy:include effect_params

//@oxy:provider 2 0 material
//@oxy:group 2 0 storage_uniform effect_tint effect_params

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return effect_tint.tint_color;
}
