struct SSRParams {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    max_distance: f32,
    thickness: f32,
    stride: f32,
    max_steps: u32,
    screen_width: f32,
    screen_height: f32,
    roughness_cutoff: f32,
    hiz_mip_count: u32,
}
