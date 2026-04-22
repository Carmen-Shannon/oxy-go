// G-Buffer fragment shader (static and skinned models)
//
// Writes per-pixel geometric and material properties into multiple render
// targets (MRTs) for consumption by screen-space GI passes (SSAO, SSR).
// No lighting evaluation is performed — this is a data-only pass.
// World-space position is reconstructed from depth at read time by the
// SSAO/SSR compute shaders, so no position MRT is needed.
//
// MRT layout:
//   @location(0) normal:   RGBA16Float — world normal XYZ (packed to [0,1]) + roughness in W
//   @location(1) albedo:   RGBA8Unorm  — albedo RGB + metallic in A
//
// Bind group layout:
//   @group(0) camera     — CameraUniform (view_proj + camera_position)
//   @group(2) material   — diffuse texture + sampler, normal map, metallic-roughness map

struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) uv:             vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) world_tangent:  vec4<f32>,
};

//@oxy:include gbuffer_output
//@oxy:include camera

struct MaterialParams {
    alpha_cutoff: f32,
}

// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform camera camera
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
//@oxy:provider 2 6 material material_params
@group(2) @binding(6) var<uniform> material_params: MaterialParams;

@fragment
fn fs_main(in: FragmentInput) -> GBufferOutput {
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    if (tex_color.a < material_params.alpha_cutoff) {
        discard;
    }

    let albedo = tex_color.rgb * in.color.rgb;

    // Normal mapping: reconstruct TBN and transform the sampled normal.
    let normal_sample = textureSample(normal_texture, normal_sampler, in.uv).rgb;
    let mapped_normal = normal_sample * 2.0 - 1.0;

    var N = normalize(in.world_normal);
    if !in.front_facing {
        N = -N;
    }
    let T = normalize(in.world_tangent.xyz);
    let B = cross(N, T) * in.world_tangent.w;
    let TBN = mat3x3<f32>(T, B, N);
    var normal = normalize(TBN * mapped_normal);

    // Metallic-roughness sampling.
    let mr_sample = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
    let roughness = mr_sample.g;
    let metallic = mr_sample.b;

    var out: GBufferOutput;
    out.normal = vec4<f32>(normal * 0.5 + 0.5, roughness);
    out.albedo = vec4<f32>(albedo, metallic);
    return out;
}
