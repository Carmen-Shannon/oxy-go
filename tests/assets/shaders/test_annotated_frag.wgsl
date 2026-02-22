//@oxy:include overlay_params
//@oxy:include effect_params

//@oxy:group 0 0 storage_uniform overlay_uniform overlay_params
//@oxy:group 0 1 storage_uniform effect_uniform effect_params
//@oxy:provider 1 0 material diffuse_texture
@group(1) @binding(0) var diffuseTexture: texture_2d<f32>;
//@oxy:provider 1 1 material diffuse_sampler
@group(1) @binding(1) var diffuseSampler: sampler;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let base_color = textureSample(diffuseTexture, diffuseSampler, in.uv);
    return base_color * overlay_uniform.overlay_color + effect_uniform.tint_color;
}
