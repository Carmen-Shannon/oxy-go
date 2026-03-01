struct SSAOParams {
    projection:       mat4x4<f32>,
    inv_view_proj:    mat4x4<f32>,
    radius:           f32,
    bias:             f32,
    power:            f32,
    sample_count:     u32,
    screen_width:     f32,
    screen_height:    f32,
    gbuffer_scale:    f32,
    _pad:             f32,
    camera_position:  vec3<f32>,
    _pad2:            f32,
};
