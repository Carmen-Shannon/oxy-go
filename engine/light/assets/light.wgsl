struct Light {
    position:      vec3<f32>,
    light_type:    u32,
    color:         vec3<f32>,
    intensity:     f32,
    direction:     vec3<f32>,
    light_range:   f32,
    inner_cone:    f32,
    outer_cone:    f32,
    casts_shadows: u32,
    _pad:          u32,
};
