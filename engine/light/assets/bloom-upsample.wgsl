@group(0) @binding(0) var lower_texture: texture_2d<f32>;
@group(0) @binding(1) var upsample_sampler: sampler;
@group(0) @binding(2) var current_texture: texture_2d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;

fn upsample_tent(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    let d = texel_size.xyxy * vec4(-1.0, -1.0, 1.0, 1.0);

    var result = vec3<f32>(0.0);
    result += textureSampleLevel(tex, samp, uv + d.xy, 0.0).rgb;                  // bottom-left
    result += textureSampleLevel(tex, samp, uv + vec2(0.0, d.y), 0.0).rgb * 2.0;  // bottom
    result += textureSampleLevel(tex, samp, uv + d.zy, 0.0).rgb;                  // bottom-right
    result += textureSampleLevel(tex, samp, uv + vec2(d.x, 0.0), 0.0).rgb * 2.0;  // left
    result += textureSampleLevel(tex, samp, uv, 0.0).rgb * 4.0;                   // center
    result += textureSampleLevel(tex, samp, uv + vec2(d.z, 0.0), 0.0).rgb * 2.0;  // right
    result += textureSampleLevel(tex, samp, uv + d.xw, 0.0).rgb;                  // top-left
    result += textureSampleLevel(tex, samp, uv + vec2(0.0, d.w), 0.0).rgb * 2.0;  // top
    result += textureSampleLevel(tex, samp, uv + d.zw, 0.0).rgb;                  // top-right
    result *= (1.0 / 16.0);

    return result;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_texture);
    if gid.x >= out_dims.x || gid.y >= out_dims.y { return; }

    let lower_dims = vec2<f32>(textureDimensions(lower_texture, 0));
    let texel_size = 1.0 / lower_dims;
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(out_dims);

    let upsampled = upsample_tent(lower_texture, upsample_sampler, uv, texel_size);
    let current = textureSampleLevel(current_texture, upsample_sampler, uv, 0.0).rgb;
    let blended = current + upsampled;

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(blended, 1.0));
}
