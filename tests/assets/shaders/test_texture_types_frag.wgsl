// Test shader exercising various texture and sampler types for parser coverage.

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var depthTexture: texture_depth_2d;
@group(0) @binding(1) var shadowSampler: sampler_comparison;
@group(0) @binding(2) var cubeTex: texture_cube<f32>;
@group(0) @binding(3) var arrayTex: texture_2d_array<f32>;
@group(0) @binding(4) var storageTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var msTex: texture_multisampled_2d<f32>;

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
