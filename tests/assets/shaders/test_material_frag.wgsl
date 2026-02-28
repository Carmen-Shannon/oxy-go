// Minimal fragment shader with material provider annotations for RegisterMaterial tests.

//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuseSampler: sampler;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let uv = in.clip_position.xy / vec2<f32>(800.0, 600.0);
    let color = textureSample(diffuseTexture, diffuseSampler, uv);
    return color;
}
