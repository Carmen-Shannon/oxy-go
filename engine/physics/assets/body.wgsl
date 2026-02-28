struct Body {
    position:                vec4<f32>,
    quaternion:              vec4<f32>,
    linear_momentum:         vec4<f32>,
    angular_momentum:        vec4<f32>,
    inv_inertia_tensor_body: mat3x3<f32>,
    inverse_mass:            f32,
    particle_start:          u32,
    particle_count:          u32,
    flags:                   u32,
    external_force:          vec4<f32>,
    external_torque:         vec4<f32>,
};
