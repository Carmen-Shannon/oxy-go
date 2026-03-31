struct CSMCascade {
    light_vp:    mat4x4<f32>,
    shadow_near: f32,
    shadow_far:  f32,
    cam_far:     f32,
    normal_bias: f32,
};

struct CSMData {
    texel_size:          vec2<f32>,
    bias:                f32,
    inner_radius:        f32,
    pcf_radius:          f32,
    shadow_max_distance: f32,
    _pad0:               f32,
    _pad1:               f32,
    cascades:            array<CSMCascade, 2>,
};