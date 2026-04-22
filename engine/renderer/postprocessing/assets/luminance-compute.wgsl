//@oxy:include luminance_params

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
//@oxy:group 0 1 storage_uniform luminance_params luminance_params
@group(0) @binding(2) var<storage, read_write> exposure_out: f32;

//@oxy:inject LUMINANCE_WORKGROUP_SIZE u32 luminance_workgroup_size

var<workgroup> shared_lum: array<f32, 256>;

@compute @workgroup_size(LUMINANCE_WORKGROUP_SIZE, LUMINANCE_WORKGROUP_SIZE, 1)
fn cs_main(@builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(local_invocation_index) local_index: u32) {

    let step_x = (luminance_params.screen_width  + LUMINANCE_WORKGROUP_SIZE - 1u) / LUMINANCE_WORKGROUP_SIZE;
    let step_y = (luminance_params.screen_height + LUMINANCE_WORKGROUP_SIZE - 1u) / LUMINANCE_WORKGROUP_SIZE;

    let px = local_id.x * step_x + step_x / 2u;
    let py = local_id.y * step_y + step_y / 2u;

    var lum: f32 = 0.0;
    if px < luminance_params.screen_width && py < luminance_params.screen_height {
        let rgb = textureLoad(hdr_texture, vec2<i32>(i32(px), i32(py)), 0).rgb;
        let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        lum = log2(max(luminance, 0.0001));
    }
    shared_lum[local_index] = lum;

    workgroupBarrier();
    if local_index < 128u { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 128u]; }
    workgroupBarrier();
    if local_index < 64u  { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 64u]; }
    workgroupBarrier();
    if local_index < 32u  { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 32u]; }
    workgroupBarrier();
    if local_index < 16u  { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 16u]; }
    workgroupBarrier();
    if local_index < 8u   { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 8u]; }
    workgroupBarrier();
    if local_index < 4u   { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 4u]; }
    workgroupBarrier();
    if local_index < 2u   { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 2u]; }
    workgroupBarrier();
    if local_index < 1u   { shared_lum[local_index] = shared_lum[local_index] + shared_lum[local_index + 1u]; }
    workgroupBarrier();

    if local_index == 0u {
        let avg_log_lum = shared_lum[0] / 256.0;
        let avg_lum     = exp2(avg_log_lum);
        let target_exposure = luminance_params.key_value / max(avg_lum, 0.0001);
        let clamped         = clamp(target_exposure, luminance_params.min_exposure, luminance_params.max_exposure);
        exposure_out    = mix(exposure_out, clamped, saturate(luminance_params.adapt_speed * luminance_params.delta_time));
    }
}
