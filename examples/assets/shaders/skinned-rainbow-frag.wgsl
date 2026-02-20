// Skinned rainbow fragment shader
//
// Generates rainbow coloring from interpolated world-space position. The
// colors are baked onto the model's surface, so they stay fixed as the
// camera moves. Hue cycles along a diagonal in world XYZ to reveal mesh
// structure and animate visibly with skeletal motion.

// ── Input from skinned vertex shader (interpolated) ────────────────
struct FragmentInput {
    @location(0) uv:    vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color:  vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

// ── Effect uniform ─────────────────────────────────────────────────
// Tint overlay: rgb = tint color, a = intensity (0.0 = no tint).
// Placed at @group(2) since this shader has no texture/sampler bindings.
//@oxy:include effect_params
// struct EffectParams {
//     tint_color: vec4<f32>,
// };
//@oxy:group 2 0 storage_uniform effect_tint effect_params
// @group(2) @binding(0) var<uniform> effect_tint: EffectParams;

// HSV to RGB conversion.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if hp < 1.0       { rgb = vec3<f32>(c, x, 0.0); }
    else if hp < 2.0  { rgb = vec3<f32>(x, c, 0.0); }
    else if hp < 3.0  { rgb = vec3<f32>(0.0, c, x); }
    else if hp < 4.0  { rgb = vec3<f32>(0.0, x, c); }
    else if hp < 5.0  { rgb = vec3<f32>(x, 0.0, c); }
    else              { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m);
}

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Use world-space position to create a rainbow gradient baked to the
    // model surface. Sum all three axes so the hue varies across the full
    // mesh and stays fixed relative to the geometry regardless of camera.
    let hue = fract((in.world_position.x + in.world_position.y + in.world_position.z) * 0.02);
    let color = hsv_to_rgb(hue, 0.85, 1.0);

    // Apply tint overlay: mix base color toward tint based on intensity (alpha)
    let tinted = mix(color, effect_tint.tint_color.rgb, effect_tint.tint_color.a);
    return vec4<f32>(tinted, 1.0);
}
