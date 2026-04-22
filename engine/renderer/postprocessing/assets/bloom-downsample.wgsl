//@oxy:include bloom_params

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
//@oxy:group 0 3 storage_uniform bloom_params bloom_params

fn downsample_13tap(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    // center tap
    let a = textureSampleLevel(tex, samp, uv, 0.0).rgb;

    // 4 diagonal taps at ±1 texel
    let b = textureSampleLevel(tex, samp, uv + texel_size * vec2(-1.0, -1.0), 0.0).rgb;
    let c = textureSampleLevel(tex, samp, uv + texel_size * vec2( 1.0, -1.0), 0.0).rgb;
    let d = textureSampleLevel(tex, samp, uv + texel_size * vec2(-1.0,  1.0), 0.0).rgb;
    let e = textureSampleLevel(tex, samp, uv + texel_size * vec2( 1.0,  1.0), 0.0).rgb;

    // 4 axis taps at ±2 texels
    let f = textureSampleLevel(tex, samp, uv + texel_size * vec2(-2.0,  0.0), 0.0).rgb;
    let g = textureSampleLevel(tex, samp, uv + texel_size * vec2( 2.0,  0.0), 0.0).rgb;
    let h = textureSampleLevel(tex, samp, uv + texel_size * vec2( 0.0, -2.0), 0.0).rgb;
    let i = textureSampleLevel(tex, samp, uv + texel_size * vec2( 0.0,  2.0), 0.0).rgb;

    // 4 more diagonal taps at ±2 texels
    let j = textureSampleLevel(tex, samp, uv + texel_size * vec2(-2.0, -2.0), 0.0).rgb;
    let k = textureSampleLevel(tex, samp, uv + texel_size * vec2( 2.0, -2.0), 0.0).rgb;
    let l = textureSampleLevel(tex, samp, uv + texel_size * vec2(-2.0,  2.0), 0.0).rgb;
    let m = textureSampleLevel(tex, samp, uv + texel_size * vec2( 2.0,  2.0), 0.0).rgb;

    // Weighted combination using 5 box filter groups
    var result = vec3<f32>(0.0);
    // Center group
    result += (a + b + c + d + e) * 0.125;
    // Edge groups
    result += (b + f + h + j) * 0.03125;
    result += (c + g + h + k) * 0.03125;
    result += (d + f + i + l) * 0.03125;
    result += (e + g + i + m) * 0.03125;

    return result;
}

fn soft_threshold(color: vec3<f32>, threshold: f32) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    let knee = threshold * 0.5;
    let soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    let contribution = soft * soft / (4.0 * knee + 0.00001);
    let weight = max(contribution, brightness - threshold) / max(brightness, 0.00001);
    return color * clamp(weight, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_texture);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }

    let in_dims = vec2<f32>(textureDimensions(input_texture, 0));
    let texel_size = 1.0 / in_dims;
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(out_dims);

    var color = downsample_13tap(input_texture, input_sampler, uv, texel_size);

    if bloom_params.threshold > 0.0 {
        color = soft_threshold(color, bloom_params.threshold);
    }

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
