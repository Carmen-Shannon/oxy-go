// Outline fragment shader
//
// Outputs a solid opaque color for the inverted-hull outline effect.
// Used with the outline vertex shader and front-face culling, so only
// back faces of the inflated mesh are visible — creating a silhouette
// outline around the model. The outline color is controlled by a
// uniform so it can be changed at runtime.

// ── Input from vertex shader (must match all vertex output locations) ──
struct FragmentInput {
    @location(0) uv:             vec2<f32>,
    @location(1) normal:         vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

//@oxy:include overlay_params

//@oxy:provider 2 0 material
//@oxy:group 2 0 storage_uniform overlay_material overlay_params

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return overlay_material.overlay_color;
}
