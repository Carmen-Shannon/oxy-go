struct ShadowUniform {
    light_vp:    mat4x4<f32>,
    light_view:  mat4x4<f32>,
    shadow_near: f32,
    shadow_far:  f32,
};
