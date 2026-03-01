// composition-frag.wgsl — Full-screen composition fragment shader.
// Samples the HDR lit texture and optional SSR texture, applies ACES tone mapping
// and gamma correction, then writes the final LDR color to the swapchain.

//@oxy:include composition_params

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var ssr_texture: texture_2d<f32>;
@group(0) @binding(3) var ssr_sampler: sampler;
//@oxy:group 0 4 storage_uniform composition_params composition_params

// aces_tonemap applies the ACES filmic tone mapping curve.
// Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_texture, hdr_sampler, uv).rgb;
    let ssr = textureSample(ssr_texture, ssr_sampler, uv);

    // Blend SSR based on alpha (confidence factor from ray march hit quality).
    let combined = hdr + ssr.rgb * ssr.a;

    // Apply exposure scaling before tone mapping so HDR values land in
    // the sweet-spot of the ACES curve.
    var color = combined * composition_params.exposure;

    if (composition_params.tone_mapping_enabled != 0u) {
        color = aces_tonemap(color);
    }

    // NOTE: No manual gamma / sRGB conversion here. The swapchain format is
    // *Srgb, so the GPU hardware converts linear→sRGB automatically on write.
    return vec4<f32>(color, 1.0);
}
