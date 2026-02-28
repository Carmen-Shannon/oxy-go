struct AnimationGlobals {
    instance_count:      u32,
    bone_count:          u32,
    bounding_radius:     f32,
    channel_data_offset: u32,
    keyframe_data_offset: u32,
    _pad1:               u32,
    _pad2:               u32,
    _pad3:               u32,
    planes:              array<FrustumPlane, 6>,
}
