// Textured fragment shader
//
// Samples a diffuse texture using interpolated UV coordinates from the vertex
// stage. The bind group for the texture and sampler is at @group(2) so it does
// not conflict with the camera (@group(0)) or instance (@group(1)) groups in
// the vertex shader. Per-binding provider annotations declare each
// binding's role so the Loader can wire material textures from Declarations.

// ── Fragment input (from vertex shader) ────────────────────────────
struct FragmentInput {
    @location(0) uv:    vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color:  vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

// ── Material bind group ────────────────────────────────────────────
//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;

// ── Effect uniform ─────────────────────────────────────────────────
// Tint overlay: rgb = tint color, a = intensity (0.0 = no tint).
// Written by the application at runtime for damage flashes, highlights, etc.
//@oxy:include effect_params
// struct EffectParams {
//     tint_color: vec4<f32>,
// };
//@oxy:group 3 0 storage_uniform effect_tint effect_params
// @group(3) @binding(0) var<uniform> effect_tint: EffectParams;

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    // Discard fully transparent fragments
    if tex_color.a < 0.01 {
        discard;
    }

    // Apply tint overlay: mix base color toward tint based on intensity (alpha)
    let tinted = mix(tex_color.rgb, effect_tint.tint_color.rgb, effect_tint.tint_color.a);
    return vec4<f32>(tinted, tex_color.a);
}
