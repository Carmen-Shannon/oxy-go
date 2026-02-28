struct BoneInfo {
    inverse_bind_matrix: mat4x4<f32>,
    local_translation:   vec3<f32>,
    parent_index:        i32,
    local_scale:         vec3<f32>,
    _pad_scale:          f32,
    local_rotation:      vec4<f32>,
}
