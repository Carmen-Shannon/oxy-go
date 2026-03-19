struct ContactShadowParams {
    view_proj:        mat4x4<f32>,
    inv_view_proj:    mat4x4<f32>,
    light_direction:  vec3<f32>,
    step_count:       u32,
    max_distance:     f32,
    thickness:        f32,
    screen_width:     f32,
    screen_height:    f32,
    camera_position:  vec3<f32>,
    _pad:             f32,
};
