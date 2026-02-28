struct LightCullUniforms {
    inv_proj:       mat4x4<f32>,
    view_matrix:    mat4x4<f32>,
    tile_count_x:   u32,
    tile_count_y:   u32,
    screen_width:   u32,
    screen_height:  u32,
    light_count:    u32,
    near:           f32,
    far:            f32,
    _pad:           u32,
};
