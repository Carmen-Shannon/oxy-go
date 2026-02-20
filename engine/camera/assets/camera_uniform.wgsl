struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
};
