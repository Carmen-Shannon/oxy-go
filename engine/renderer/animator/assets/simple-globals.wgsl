struct GlobalData {
    instance_count:  u32,
    delta_time:      f32,
    bounding_radius: f32,
    _padding:        f32,
    planes:          array<FrustumPlane, 6>,
}
