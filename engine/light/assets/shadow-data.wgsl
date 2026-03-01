struct ShadowData {
    light_vp:              mat4x4<f32>,
    light_view:            mat4x4<f32>,
    texel_size:            vec2<f32>,
    bias:                  f32,
    normal_bias:           f32,
    shadow_near:           f32,
    shadow_far:            f32,
    min_variance:          f32,
    light_bleed_reduction: f32,
    light_size:            f32,
    shadow_half_extent:    f32,
    _pad:                  vec2<f32>,
};
