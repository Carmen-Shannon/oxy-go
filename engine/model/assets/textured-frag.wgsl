// Textured fragment shader
//
// Samples a diffuse texture using interpolated UV coordinates from the vertex
// stage. The bind group for the texture and sampler is at @group(2) so it does
// not conflict with the camera (@group(0)) or instance (@group(1)) groups in
// the vertex shader. Per-binding provider annotations declare each
// binding's role so the Loader can wire material textures from Declarations.

struct FragmentInput {
    @location(0) uv:    vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color:  vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    // Discard fully transparent fragments
    if tex_color.a < 0.01 {
        discard;
    }

    // Combine texture color with per-vertex color. For textured models the
    // vertex color is typically (1,1,1,1) so this is a no-op multiply. For
    // hand-built models with per-vertex colors and no texture, the fallback
    // 1×1 white texture makes the multiply pass the vertex color through.
    return tex_color * in.color;
}
