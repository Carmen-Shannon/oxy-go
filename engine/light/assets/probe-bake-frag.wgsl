// Probe bake fragment shader
//
// Outputs the surface albedo color to a single RGBA8Unorm render target.
// This is an unlit bake — direct lighting is not evaluated. The resulting
// radiance approximation is sufficient for low-frequency indirect illumination
// encoded as L2 spherical harmonics through the SH projection compute shader.
//
// Bind group layout:
//   @group(0) bake_camera — ProbeBakeCamera (consumed by vertex shader)
//   @group(1) instances   — per-instance InstanceData (consumed by vertex shader)
//   @group(2) material    — diffuse texture + sampler, normal map, metallic-roughness map
//                           (all 6 entries declared for bind group layout compat with lit materials)

struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv:             vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) world_tangent:  vec4<f32>,
};

//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;
//@oxy:provider 2 2 material normal_texture
@group(2) @binding(2) var normal_texture: texture_2d<f32>;
//@oxy:provider 2 3 material normal_sampler
@group(2) @binding(3) var normal_sampler: sampler;
//@oxy:provider 2 4 material metallic_roughness_texture
@group(2) @binding(4) var metallic_roughness_texture: texture_2d<f32>;
//@oxy:provider 2 5 material metallic_roughness_sampler
@group(2) @binding(5) var metallic_roughness_sampler: sampler;

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    if tex_color.a < 0.01 {
        discard;
    }

    let albedo = tex_color.rgb * in.color.rgb;
    return vec4<f32>(albedo, tex_color.a);
}
